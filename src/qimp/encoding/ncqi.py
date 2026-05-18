"""NCQI encoding: NEQR extension to RGB images.

A 2^n × 2^n RGB image with intensity resolution 2^q per channel is encoded
into ``2n + 3q`` qubits. The intensity of each channel sits in its own
q-bit basis-state register; retrieval is exact.

Qubit layout (low → high index):
  - q R-intensity qubits (qubits 0 … q-1)
  - q G-intensity qubits (qubits q … 2q-1)
  - q B-intensity qubits (qubits 2q … 3q-1)
  - n X-position qubits (qubits 3q … 3q+n-1)
  - n Y-position qubits (qubits 3q+n … 3q+2n-1)

Reference: docs/tesi.pdf §3.1.2 (color extension).
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit

__all__ = ["NcqiEncoder", "ncqi_circuit", "ncqi_decode"]


def _validate_image(image: np.ndarray, q: int) -> int:
    if q < 1:
        raise ValueError("q must be >= 1")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"image must be (H, W, 3) RGB, got shape {image.shape}")
    h, w, _ = image.shape
    if h != w:
        raise ValueError(f"image must be square, got {h}×{w}")
    if h < 2 or (h & (h - 1)) != 0:
        raise ValueError(f"image side must be a power of two ≥ 2, got {h}")
    n = int(math.log2(h))
    max_int = (1 << q) - 1
    if image.min() < 0:
        raise ValueError("NCQI requires non-negative intensities")
    if image.max() > max_int:
        raise ValueError(
            f"intensity {image.max()} exceeds 2^q - 1 = {max_int}; increase q or rescale"
        )
    return n


def ncqi_circuit(image: np.ndarray, q: int = 8) -> QuantumCircuit:
    """Build an NCQI circuit for a single RGB image.

    Parameters
    ----------
    image
        ``(2^n, 2^n, 3)`` integer array with values in ``[0, 2^q - 1]``.
    q
        Number of intensity qubits per channel.

    Returns
    -------
    QuantumCircuit
        Circuit with ``2n + 3q`` qubits.
    """
    n = _validate_image(image, q)
    total = 2 * n + 3 * q
    qc = QuantumCircuit(total)
    pos_qubits = list(range(3 * q, 3 * q + 2 * n))
    qc.h(pos_qubits)
    qc.barrier()

    image_int = image.astype(np.int64)
    for row in range(1 << n):
        for col in range(1 << n):
            r_val = int(image_int[row, col, 0])
            g_val = int(image_int[row, col, 1])
            b_val = int(image_int[row, col, 2])
            if r_val == 0 and g_val == 0 and b_val == 0:
                continue

            x_bits = format(col, f"0{n}b")[::-1]
            y_bits = format(row, f"0{n}b")[::-1]
            x_flips = [3 * q + i for i, bit in enumerate(x_bits) if bit == "0"]
            y_flips = [3 * q + n + i for i, bit in enumerate(y_bits) if bit == "0"]
            all_flips = x_flips + y_flips
            for qbit in all_flips:
                qc.x(qbit)

            for channel_offset, intensity in enumerate((r_val, g_val, b_val)):
                if intensity == 0:
                    continue
                int_bits = format(intensity, f"0{q}b")[::-1]
                for i, bit in enumerate(int_bits):
                    if bit == "1":
                        target = channel_offset * q + i
                        qc.mcx(pos_qubits, target)

            for qbit in all_flips:
                qc.x(qbit)
            qc.barrier()

    return qc


def _decode_state_string(state: str, n: int, q: int) -> tuple[int, int, int, int, int]:
    """Return (row, col, R, G, B) from a Qiskit count key.

    Layout (MSB-first): y_{n-1}…y_0 x_{n-1}…x_0 B_{q-1}…B_0 G_{q-1}…G_0 R_{q-1}…R_0.
    """
    flat = state.replace(" ", "")
    if len(flat) != 2 * n + 3 * q:
        raise ValueError(f"unexpected count key length {len(flat)} for n={n}, q={q}")
    y_str = flat[:n]
    x_str = flat[n : 2 * n]
    b_str = flat[2 * n : 2 * n + q]
    g_str = flat[2 * n + q : 2 * n + 2 * q]
    r_str = flat[2 * n + 2 * q : 2 * n + 3 * q]
    return int(y_str, 2), int(x_str, 2), int(r_str, 2), int(g_str, 2), int(b_str, 2)


def ncqi_decode(counts: dict[str, int], n: int, q: int) -> np.ndarray:
    """Decode NCQI counts. Exact retrieval — picks the modal (R, G, B) per pixel."""
    if n < 1 or q < 1:
        raise ValueError("n and q must both be >= 1")
    side = 1 << n
    histogram: dict[tuple[int, int], dict[tuple[int, int, int], int]] = {}
    for state, count in counts.items():
        row, col, r, g, b = _decode_state_string(state, n, q)
        bucket = histogram.setdefault((row, col), {})
        bucket[(r, g, b)] = bucket.get((r, g, b), 0) + count

    img = np.zeros((side, side, 3), dtype=np.int64)
    for (row, col), bucket in histogram.items():
        r, g, b = max(bucket, key=bucket.__getitem__)
        img[row, col, 0] = r
        img[row, col, 1] = g
        img[row, col, 2] = b
    return img


class NcqiEncoder:
    """Stateful NCQI encoder mirroring `NeqrEncoder`."""

    def __init__(self, q: int = 8) -> None:
        if q < 1:
            raise ValueError("q must be >= 1")
        self.q = q
        self.n: int | None = None

    def encode(self, image: np.ndarray) -> QuantumCircuit:
        qc = ncqi_circuit(image, q=self.q)
        self.n = int(math.log2(image.shape[0]))
        return qc

    def decode(self, counts: dict[str, int]) -> np.ndarray:
        if self.n is None:
            raise RuntimeError("call encode() before decode()")
        return ncqi_decode(counts, n=self.n, q=self.q)
