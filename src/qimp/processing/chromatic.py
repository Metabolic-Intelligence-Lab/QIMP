"""Chromatic transformations: color complement, color change, halving, classify.

Reference: docs/tesi.pdf §2.4
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

__all__ = [
    "frqi_color_change",
    "frqi_color_complement",
    "neqr_classify_complement",
    "neqr_color_complement",
    "neqr_half_intensity",
]


# -------------------------------------------------------------------- FRQI ----


def frqi_color_complement(qc: QuantumCircuit, *, color_qubit: int | None = None) -> QuantumCircuit:
    """Invert FRQI colors by applying an X to the color qubit.

    Maps ``cos(θ)|0⟩ + sin(θ)|1⟩`` → ``sin(θ)|0⟩ + cos(θ)|1⟩``, i.e. ``θ → π−θ``.
    If ``color_qubit`` is ``None`` the highest-indexed qubit is used.
    """
    target = qc.num_qubits - 1 if color_qubit is None else color_qubit
    qc.x(target)
    return qc


def frqi_color_change(
    qc: QuantumCircuit,
    theta: float,
    *,
    color_qubit: int | None = None,
) -> QuantumCircuit:
    """Apply a custom unitary on the FRQI color qubit, parameterised by ``theta``.

    The unitary is ``[[cos(θ/2), sin(θ/2)], [sin(θ/2), -cos(θ/2)]]`` — a reflection
    matrix that maps ``cos(α)|0⟩ + sin(α)|1⟩ → cos(α+θ)|0⟩ + sin(α+θ)|1⟩``,
    shifting every pixel intensity by ``θ`` modulo ``π``.
    """
    target = qc.num_qubits - 1 if color_qubit is None else color_qubit
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    unitary = np.array([[c, s], [s, -c]], dtype=np.complex128)
    qc.unitary(Operator(unitary), [target], label=f"C({theta:.3f})")
    return qc


# -------------------------------------------------------------------- NEQR ----


def neqr_color_complement(qc: QuantumCircuit, q: int) -> QuantumCircuit:
    """Bit-wise NOT of every intensity bit: maps ``i → (2^q − 1) − i`` for each pixel.

    Applies an X gate to each of the ``q`` intensity qubits (assumed to be the
    lowest-indexed ``q`` qubits, per the NEQR layout).
    """
    if q < 1:
        raise ValueError("q must be >= 1")
    for i in range(q):
        qc.x(i)
    return qc


def neqr_half_intensity(qc: QuantumCircuit, q: int) -> QuantumCircuit:
    """Divide every pixel intensity by 2 (integer division) by right-shifting one bit.

    Implemented as a cascade of SWAPs that re-labels intensity qubits and a final
    reset of the freed top bit. Effectively maps ``b_{q-1}…b_1 b_0 → 0 b_{q-1}…b_1``.

    The reset on qubit ``q-1`` brings it back to |0⟩; this only works for ideal
    simulations (resetting a real device qubit involves measurement+conditional X).
    """
    if q < 2:
        raise ValueError("q must be >= 2 for half_intensity to be meaningful")
    # Move bits down: bit i takes the value of bit i+1.
    for i in range(q - 1):
        qc.swap(i, i + 1)
    qc.reset(q - 1)
    return qc


def neqr_classify_complement(
    qc: QuantumCircuit,
    q: int,
    threshold_bit: int,
) -> QuantumCircuit:
    """Binary classification: complement intensity bits below a threshold bit.

    Flips every intensity bit at index < ``threshold_bit``. Useful for
    coarse thresholding workflows.
    """
    if not 0 <= threshold_bit <= q:
        raise ValueError(f"threshold_bit must be in [0, {q}], got {threshold_bit}")
    for i in range(threshold_bit):
        qc.x(i)
    return qc
