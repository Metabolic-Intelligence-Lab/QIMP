"""Quantum arithmetic on NEQR-encoded images: add-1, ADD/SUB, comparator, sort.

These primitives compare or combine two NEQR images. They work on arbitrary-length
qubit registers (no fixed q) and produce auxiliary qubits used only during
measurement; decoders are provided to project back to image intensities.

Reference: docs/tesi.pdf §3.1.2 (last sub-section), §2.6, §2.7.2
"""

from __future__ import annotations

# TODO(Fase 4):
#   - qc_add_1(qc, a_qubits, b_qubits) -> QuantumCircuit          # one-qubit add
#   - q_ADD_SUB(qc, a_qubits, b_qubits, mode) -> QuantumCircuit   # mode in {"add","sub"}
#   - neqr_comparator(qc, a_qubits, b_qubits) -> QuantumCircuit
#   - neqr_sort(qc, img1_qubits, img2_qubits, sort_index) -> QuantumCircuit
# Decoders accept variable image size and intensity resolution.
