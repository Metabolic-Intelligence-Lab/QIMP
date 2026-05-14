"""Quantum filters for spatial processing.

Currently exposes the Quantum Hadamard Edge Detection (QHED) algorithm for
images encoded with QPIE. QHED is a poly(n) algorithm that uses a single
auxiliary qubit, a Hadamard gate, and a cyclic decrement gate to extract
gradient information from adjacent pixels in a single shot batch.

Reference: docs/tesi.pdf §2.7.3, §3.1.4
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit

from qimp.encoding.qpie import _validate_and_normalize

__all__ = ["qhed_decode", "qhed_filter", "qhed_full_edges"]


def _decrement(qc: QuantumCircuit, qubits: list[int]) -> None:
    """Cyclic decrement on a little-endian register (qubits[0] = LSB).

    Implements ``|k⟩ → |k − 1 mod 2^N⟩``. Inverse of the standard ripple
    incrementer. ``O(N²)`` gates after Toffoli decomposition.
    """
    qc.x(qubits[0])
    for k in range(1, len(qubits)):
        qc.mcx(qubits[:k], qubits[k])


def qhed_filter(image: np.ndarray) -> tuple[QuantumCircuit, int, float]:
    """Build a QHED circuit that computes horizontal gradients of `image`.

    The circuit has ``2n + 1`` qubits:
    - qubit 0: auxiliary qubit (LSB)
    - qubits 1..2n: position register holding the QPIE-encoded image

    After execution, measuring with ``aux = 1`` reveals gradient magnitudes:
    ``P(aux=1, pos=i) = (1/2) (c_i − c_{i+1})²``.

    Returns
    -------
    qc
        The QHED circuit (no measurements appended).
    n
        Spatial qubits per axis.
    rms
        Root-mean-square of the input intensities, ``√(mean(I²))``.
        Forward this to `qhed_decode` to recover gradient units.

    Scales freely with ``n`` (image side = 2^n).
    """
    amplitudes, n, rms = _validate_and_normalize(image)
    total = 1 << (2 * n)

    # Inject aux=|0⟩ at the LSB: full amplitudes = [c_0, 0, c_1, 0, …, c_{N-1}, 0].
    combined = np.zeros(2 * total, dtype=np.float64)
    combined[::2] = amplitudes

    qc = QuantumCircuit(2 * n + 1)
    qc.initialize(combined, range(2 * n + 1))

    aux = 0
    qc.h(aux)
    qc.barrier()
    _decrement(qc, list(range(2 * n + 1)))
    qc.barrier()
    qc.h(aux)
    return qc, n, rms


def qhed_decode(
    counts: dict[str, int],
    n: int,
    rms: float = 1.0,
    *,
    max_gradient: float | None = None,
) -> np.ndarray:
    """Recover the horizontal gradient image from QHED measurement counts.

    Parameters
    ----------
    counts
        Histogram from `ideal_simulation(qc, shots)` with `qc` the QHED filter.
    n
        Number of spatial qubits per axis.
    rms
        RMS scale factor returned by `qhed_filter`. Use 1.0 to return *normalized*
        gradients.
    max_gradient
        If given, clip the output to ``[0, max_gradient]``.

    Returns
    -------
    np.ndarray
        ``(2^n, 2^n)`` float array of horizontal gradient magnitudes.
        Gradients are non-negative; the sign is lost because we measure
        ``P(aux=1) ∝ (c_i − c_{i+1})²``.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    side = 1 << n
    total = 1 << (2 * n)
    total_shots = sum(counts.values())
    if total_shots == 0:
        return np.zeros((side, side), dtype=np.float64)

    grad = np.zeros(total, dtype=np.float64)
    for state, count in counts.items():
        flat = state.replace(" ", "")
        # Qiskit reports MSB-first; qubit 0 (LSB = aux) is the last character.
        if flat[-1] != "1":
            continue
        pos_str = flat[:-1]  # MSB-first position bits
        pos_idx = int(pos_str, 2)
        prob = count / total_shots
        grad[pos_idx] = math.sqrt(2.0 * prob) * rms * math.sqrt(total)

    img = np.zeros((side, side), dtype=np.float64)
    for pixel_idx in range(total):
        bits = format(pixel_idx, f"0{2 * n}b")
        row = int(bits[:n], 2)
        col = int(bits[n:], 2)
        img[row, col] = grad[pixel_idx]

    if max_gradient is not None:
        img = np.clip(img, 0.0, max_gradient)
    return img


def qhed_full_edges(
    image: np.ndarray,
    counts_horizontal: dict[str, int],
    counts_vertical: dict[str, int],
    *,
    max_gradient: float | None = None,
) -> np.ndarray:
    """Combine horizontal and vertical gradients into a single edge map.

    Typical workflow::

        qc_h, n, rms = qhed_filter(image)
        qc_v, _, _   = qhed_filter(image.T)
        counts_h = ideal_simulation(qc_h, shots=N)
        counts_v = ideal_simulation(qc_v, shots=N)
        edges    = qhed_full_edges(image, counts_h, counts_v, max_gradient=5.0)
    """
    amplitudes, n, rms = _validate_and_normalize(image)
    del amplitudes  # only used for n / rms
    edges_h = qhed_decode(counts_horizontal, n=n, rms=rms)
    edges_v = qhed_decode(counts_vertical, n=n, rms=rms).T
    total: np.ndarray = edges_h + edges_v
    if max_gradient is not None:
        total = np.clip(total, 0.0, max_gradient)
    return total
