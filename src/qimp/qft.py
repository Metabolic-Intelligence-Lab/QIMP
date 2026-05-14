"""QFT utilities.

Thin wrappers around qiskit.circuit.library.QFT that work on arbitrary qubit
ranges and can be inserted into any QIMP processing pipeline.
"""

from __future__ import annotations

# TODO(Fase 3): migrate from legacy/scripts/quantum_qft.py
#   - apply_qft(qc, qubits) -> QuantumCircuit
#   - apply_inverse_qft(qc, qubits) -> QuantumCircuit
