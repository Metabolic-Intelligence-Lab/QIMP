"""NEQR encoding: Novel Enhanced Quantum Representation.

A 2^n × 2^n grayscale image with intensity resolution 2^q is encoded into
``2n + q`` qubits. The intensity is stored in a separate q-bit register
(basis-state encoding), enabling exact retrieval and arithmetic operations.

Qubit layout (low → high index):
  - q intensity qubits (qubits 0 … q-1)
  - n X-position qubits (qubits q … q+n-1)
  - n Y-position qubits (qubits q+n … q+2n-1)

Reference: docs/tesi.pdf §2.2.2, §3.1.2
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit

__all__ = ["NeqrEncoder", "neqr_circuit", "neqr_decode"]


def _validate_image(image: np.ndarray, q: int) -> int:
    if q < 1:
        raise ValueError("q (intensity qubits) must be >= 1")
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError(f"image must be square 2D, got shape {image.shape}")
    side = image.shape[0]
    if side < 2 or (side & (side - 1)) != 0:
        raise ValueError(f"image side must be a power of two ≥ 2, got {side}")
    n = int(math.log2(side))
    max_int = (1 << q) - 1
    if image.min() < 0:
        raise ValueError("NEQR requires non-negative intensities")
    if image.max() > max_int:
        raise ValueError(
            f"intensity {image.max()} exceeds 2^q - 1 = {max_int}; increase q or rescale the image"
        )
    return n


def neqr_circuit(image: np.ndarray, q: int = 8) -> QuantumCircuit:
    """Build a NEQR circuit for a single grayscale image.

    Parameters
    ----------
    image
        Square ``(2^n, 2^n)`` integer array with values in ``[0, 2^q - 1]``.
    q
        Number of intensity qubits. Image values must fit in ``q`` bits.

    Returns
    -------
    QuantumCircuit
        Circuit with ``2n + q`` qubits.

    Notes
    -----
    Complexity: ``O(q · n · 2^(2n))`` gates worst-case. Scales freely with both
    ``n`` and ``q``.
    """
    n = _validate_image(image, q)
    total_qubits = 2 * n + q
    qc = QuantumCircuit(total_qubits)

    # Step 1: superposition over position qubits.
    pos_qubits = list(range(q, q + 2 * n))
    qc.h(pos_qubits)
    qc.barrier()

    # Step 2: per-pixel multi-controlled X gates onto the intensity bits that are 1.
    image_int = image.astype(np.int64)
    for row in range(1 << n):
        for col in range(1 << n):
            intensity = int(image_int[row, col])
            if intensity == 0:
                continue

            # Pixel position |yx⟩ → bit-i of (col) on qubit q+i, bit-i of (row) on q+n+i.
            x_bits = format(col, f"0{n}b")[::-1]  # LSB-first
            y_bits = format(row, f"0{n}b")[::-1]  # LSB-first

            x_flips = [q + i for i, bit in enumerate(x_bits) if bit == "0"]
            y_flips = [q + n + i for i, bit in enumerate(y_bits) if bit == "0"]
            all_flips = x_flips + y_flips

            for qbit in all_flips:
                qc.x(qbit)

            # For every '1' bit of `intensity`, MCX onto the corresponding intensity qubit.
            int_bits = format(intensity, f"0{q}b")[::-1]  # LSB-first
            for i, bit in enumerate(int_bits):
                if bit == "1":
                    qc.mcx(pos_qubits, i)

            for qbit in all_flips:
                qc.x(qbit)
            qc.barrier()

    return qc


def _decode_state_string(state: str, n: int, q: int) -> tuple[int, int, int]:
    """Return (row, col, intensity) from a Qiskit count key.

    Qiskit reports the bitstring high-to-low. Layout: ``y_{n-1}…y_0 x_{n-1}…x_0 b_{q-1}…b_0``.
    """
    flat = state.replace(" ", "")
    if len(flat) != 2 * n + q:
        raise ValueError(f"unexpected count key length {len(flat)} for n={n}, q={q}")
    y_str = flat[:n]
    x_str = flat[n : 2 * n]
    int_str = flat[2 * n :]
    return int(y_str, 2), int(x_str, 2), int(int_str, 2)


def neqr_decode(counts: dict[str, int], n: int, q: int) -> np.ndarray:
    """Decode NEQR measurement counts into the reconstructed image.

    For each position, the most-frequent intensity reading is taken (NEQR is
    exact, so ideal simulations yield a single intensity per pixel).
    """
    if n < 1 or q < 1:
        raise ValueError("n and q must both be >= 1")
    side = 1 << n
    histogram: dict[tuple[int, int], dict[int, int]] = {}
    for state, count in counts.items():
        row, col, intensity = _decode_state_string(state, n, q)
        histogram.setdefault((row, col), {})
        histogram[(row, col)][intensity] = histogram[(row, col)].get(intensity, 0) + count

    img = np.zeros((side, side), dtype=np.int64)
    for (row, col), votes in histogram.items():
        img[row, col] = max(votes, key=votes.__getitem__)
    return img


class NeqrEncoder:
    """Stateful NEQR encoder that records ``n`` and ``q`` for round-trips."""

    def __init__(self, q: int = 8) -> None:
        if q < 1:
            raise ValueError("q must be >= 1")
        self.q = q
        self.n: int | None = None

    def encode(self, image: np.ndarray) -> QuantumCircuit:
        qc = neqr_circuit(image, q=self.q)
        self.n = int(math.log2(image.shape[0]))
        return qc

    def decode(self, counts: dict[str, int]) -> np.ndarray:
        if self.n is None:
            raise RuntimeError("call encode() before decode()")
        return neqr_decode(counts, n=self.n, q=self.q)
