"""MCRQI encoding: FRQI extension to RGB images.

A 2^n × 2^n RGB image is encoded into ``2n + 3`` qubits: 2n position qubits
and 3 color qubits (one per channel R / G / B). Each color qubit independently
carries the channel's pixel intensity as an RY rotation angle.

Compared to FRQI:
- adds 2 qubits (3 color qubits instead of 1);
- per-pixel construction is 3 fully-controlled RY gates instead of 1.

Reference: docs/tesi.pdf §3.1.1 (color extension).
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate

from qimp.encoding.frqi import intensities_to_angles

__all__ = ["McrqiEncoder", "mcrqi_circuit", "mcrqi_decode"]


def _validate_rgb(image: np.ndarray) -> int:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"image must be (H, W, 3) RGB, got shape {image.shape}")
    h, w, _ = image.shape
    if h != w:
        raise ValueError(f"image must be square, got {h}×{w}")
    if h < 2 or (h & (h - 1)) != 0:
        raise ValueError(f"image side must be a power of two ≥ 2, got {h}")
    return int(math.log2(h))


def _infer_normalization(image: np.ndarray) -> float:
    if np.issubdtype(image.dtype, np.integer):
        return float(np.iinfo(image.dtype).max)
    observed = float(np.asarray(image).max())
    return observed if observed > 1.0 else 1.0


def mcrqi_circuit(image: np.ndarray, *, normalization: float | None = None) -> QuantumCircuit:
    """Build an MCRQI circuit for a single RGB image.

    Parameters
    ----------
    image
        ``(2^n, 2^n, 3)`` array of non-negative intensities.
    normalization
        Intensity that maps to angle π. Auto-inferred when None.

    Returns
    -------
    QuantumCircuit
        Circuit with ``2n + 3`` qubits (no classical bits). Qubit layout:
        0..2n-1 = position (low = X, high = Y); 2n = R, 2n+1 = G, 2n+2 = B.
    """
    n = _validate_rgb(image)
    if normalization is None:
        normalization = _infer_normalization(image)
    num_controls = 2 * n
    total = 2 * n + 3
    qc = QuantumCircuit(total)
    qc.h(range(2 * n))
    qc.barrier()

    flat_rgb = image.reshape(-1, 3)  # (4^n, 3)
    angles = np.stack(
        [intensities_to_angles(flat_rgb[:, c].astype(np.float64), normalization) for c in range(3)],
        axis=-1,
    )  # (4^n, 3)

    color_target = [2 * n, 2 * n + 1, 2 * n + 2]
    for pixel_idx in range(1 << (2 * n)):
        pos_bits = format(pixel_idx, f"0{2 * n}b")[::-1]
        flips = [q for q, bit in enumerate(pos_bits) if bit == "0"]
        for q in flips:
            qc.x(q)
        for c in range(3):
            theta = float(angles[pixel_idx, c])
            gate = RYGate(theta).control(num_controls, annotated=True)
            qc.append(gate, [*range(2 * n), color_target[c]])
        for q in flips:
            qc.x(q)
        qc.barrier()

    return qc


def _decode_state_string(state: str, n: int) -> tuple[int, int, int, int]:
    """Return (pixel_idx, R_bit, G_bit, B_bit) from a Qiskit count key.

    Qiskit reports MSB-first. Layout: B G R q[2n-1] … q[0].
    """
    flat = state.replace(" ", "")
    b_bit = int(flat[0])
    g_bit = int(flat[1])
    r_bit = int(flat[2])
    pos_str = flat[3:]
    pixel_idx = int(pos_str, 2) if pos_str else 0
    return pixel_idx, r_bit, g_bit, b_bit


def mcrqi_decode(counts: dict[str, int], n: int, normalization: float = 1.0) -> np.ndarray:
    """Decode MCRQI measurement counts into an RGB image.

    For each channel, intensity at pixel j = ``prob(channel_qubit=1 | pos=j) · normalization``.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    side = 1 << n
    total_per_image = 1 << (2 * n)
    sum_p1 = np.zeros((total_per_image, 3), dtype=np.float64)
    sum_p_total = np.zeros((total_per_image, 3), dtype=np.float64)

    for state, count in counts.items():
        pixel_idx, r_bit, g_bit, b_bit = _decode_state_string(state, n)
        for c, bit in enumerate((r_bit, g_bit, b_bit)):
            sum_p_total[pixel_idx, c] += count
            if bit == 1:
                sum_p1[pixel_idx, c] += count

    with np.errstate(invalid="ignore", divide="ignore"):
        intensities = np.where(
            sum_p_total > 0,
            sum_p1 / sum_p_total * normalization,
            0.0,
        )

    img = np.zeros((side, side, 3), dtype=np.float64)
    for pixel_idx in range(total_per_image):
        bits = format(pixel_idx, f"0{2 * n}b")
        row = int(bits[:n], 2)
        col = int(bits[n:], 2)
        img[row, col, :] = intensities[pixel_idx, :]
    return img


class McrqiEncoder:
    """Stateful MCRQI encoder mirroring `FrqiEncoder`."""

    def __init__(self, *, normalization: float | None = None) -> None:
        self._normalization = normalization
        self.n: int | None = None

    @property
    def normalization(self) -> float:
        if self._normalization is None:
            raise RuntimeError("normalization is set only after encode() is called")
        return self._normalization

    def encode(self, image: np.ndarray) -> QuantumCircuit:
        self.n = _validate_rgb(image)
        if self._normalization is None:
            self._normalization = _infer_normalization(image)
        return mcrqi_circuit(image, normalization=self._normalization)

    def decode(self, counts: dict[str, int]) -> np.ndarray:
        if self.n is None or self._normalization is None:
            raise RuntimeError("call encode() before decode()")
        return mcrqi_decode(counts, n=self.n, normalization=self._normalization)
