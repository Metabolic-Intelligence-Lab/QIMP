"""QPIE encoding: Quantum Probability Image Encoding.

Pixel intensities are normalized so that their squared values sum to 1, then
loaded as amplitude probabilities into 2n qubits via Qiskit's `initialize`.
Compact (only 2n qubits) but produces an arbitrary state preparation that
transpiles to deep circuits.

Reference: docs/tesi.pdf §2.2.3, §3.1.3
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit

__all__ = ["QpieEncoder", "qpie_circuit", "qpie_decode"]


def _validate_and_normalize(image: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Return (amplitudes, n, rms) for a square 2^n × 2^n image."""
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError(f"image must be square 2D, got shape {image.shape}")
    side = image.shape[0]
    if side < 2 or (side & (side - 1)) != 0:
        raise ValueError(f"image side must be a power of two ≥ 2, got {side}")
    n = int(math.log2(side))
    flat = image.astype(np.float64).flatten()
    if np.any(flat < 0):
        raise ValueError("QPIE requires non-negative intensities")
    norm = float(np.linalg.norm(flat))
    if norm == 0:
        # All-zero image: encode as uniform superposition.
        amplitudes = np.full(flat.size, 1.0 / math.sqrt(flat.size))
        rms = 0.0
    else:
        amplitudes = flat / norm
        rms = norm / math.sqrt(flat.size)  # √(mean(I²))
    return amplitudes, n, rms


def qpie_circuit(image: np.ndarray) -> QuantumCircuit:
    """Build a QPIE circuit for a single grayscale image.

    The circuit has 2n qubits; the state is
    ``|Ψ⟩ = Σ_i (I_i / ‖I‖) |i⟩`` where ``|i⟩`` is the position basis state.

    Scales freely with n. The intensity resolution is bounded only by the number
    of measurement shots (no `q` parameter — QPIE has no intensity register).
    """
    amplitudes, n, _ = _validate_and_normalize(image)
    qc = QuantumCircuit(2 * n)
    qc.initialize(amplitudes, range(2 * n))
    return qc


def qpie_decode(
    counts: dict[str, int],
    n: int,
    rms: float = 1.0,
) -> np.ndarray:
    """Decode QPIE measurement counts into a reconstructed image.

    Parameters
    ----------
    counts
        Histogram from `ideal_simulation(qpie_circuit(image), shots)`.
    n
        Number of spatial qubits per axis (image side = 2^n).
    rms
        Root-mean-square of the original intensities, ``√(mean(I²))``. Returned
        by `QpieEncoder.encode()`; if unknown, leave 1.0 to recover the
        *normalized* image (the absolute scale of QPIE is not preserved).

    Returns
    -------
    np.ndarray
        ``(2^n, 2^n)`` float array; ``I_yx ≈ √(N_yx / N_meas) · RMS·√(2^(2n))``.
        Multiplying by ``√(2^(2n))`` cancels the 1/√M factor implicit in the
        amplitude normalisation.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    side = 1 << n
    total = 1 << (2 * n)
    total_shots = sum(counts.values())
    if total_shots == 0:
        return np.zeros((side, side), dtype=np.float64)

    probs = np.zeros(total, dtype=np.float64)
    for state, count in counts.items():
        flat = state.replace(" ", "")
        # Qiskit reports bitstring high-to-low: q[2n-1] … q[0].
        # Our encode treats bit-i of `i` as qubit i (LSB-first). Therefore the
        # MSB-first bitstring is precisely the canonical big-endian integer
        # representation of the pixel index — same convention as FRQI decode.
        idx = int(flat, 2)
        probs[idx] += count / total_shots

    intensities = np.sqrt(probs) * rms * math.sqrt(total)

    img = np.zeros((side, side), dtype=np.float64)
    for pixel_idx in range(total):
        bits = format(pixel_idx, f"0{2 * n}b")  # MSB-first
        row = int(bits[:n], 2)
        col = int(bits[n:], 2)
        img[row, col] = intensities[pixel_idx]
    return img


class QpieEncoder:
    """Stateful QPIE encoder that records `n` and the RMS for decoding."""

    def __init__(self) -> None:
        self.n: int | None = None
        self.rms: float | None = None

    def encode(self, image: np.ndarray) -> QuantumCircuit:
        amplitudes, n, rms = _validate_and_normalize(image)
        qc = QuantumCircuit(2 * n)
        qc.initialize(amplitudes, range(2 * n))
        self.n = n
        self.rms = rms
        return qc

    def decode(self, counts: dict[str, int]) -> np.ndarray:
        if self.n is None or self.rms is None:
            raise RuntimeError("call encode() before decode()")
        return qpie_decode(counts, n=self.n, rms=self.rms)
