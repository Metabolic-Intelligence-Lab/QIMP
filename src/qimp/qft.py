"""QFT utilities.

Thin wrappers around `qiskit.circuit.library.QFT` that work on arbitrary
qubit ranges and integrate cleanly into QIMP pipelines.
"""

from __future__ import annotations

from collections.abc import Sequence

from qiskit import QuantumCircuit

# Qiskit ≥ 2.1: QFTGate is the modern replacement for the deprecated QFT class.
try:
    from qiskit.circuit.library import QFTGate as _QFTGate
except ImportError:  # pragma: no cover - Qiskit < 2.1 fallback
    from qiskit.circuit.library import QFT as _QFTGate  # noqa: N811

__all__ = ["apply_inverse_qft", "apply_qft"]


def _resolve_qubits(qc: QuantumCircuit, qubits: Sequence[int] | int) -> list[int]:
    if isinstance(qubits, int):
        if qubits < 1 or qubits > qc.num_qubits:
            raise ValueError(f"qubits int must be in [1, {qc.num_qubits}], got {qubits}")
        return list(range(qubits))
    resolved = list(qubits)
    if not resolved:
        raise ValueError("qubits sequence is empty")
    if any(q < 0 or q >= qc.num_qubits for q in resolved):
        raise ValueError(f"qubit indices out of range for circuit with {qc.num_qubits} qubits")
    return resolved


def apply_qft(qc: QuantumCircuit, qubits: Sequence[int] | int) -> QuantumCircuit:
    """Append a (forward) QFT to `qc` on the specified qubits.

    `qubits` can be either an `int` n (interpreted as `range(n)`) or any
    sequence of qubit indices. Works for arbitrary n.
    """
    indices = _resolve_qubits(qc, qubits)
    qc.append(_QFTGate(len(indices)), indices)
    return qc


def apply_inverse_qft(qc: QuantumCircuit, qubits: Sequence[int] | int) -> QuantumCircuit:
    """Append an inverse QFT to `qc` on the specified qubits."""
    indices = _resolve_qubits(qc, qubits)
    qc.append(_QFTGate(len(indices)).inverse(), indices)
    return qc
