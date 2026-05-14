"""QPIE encoding: Quantum Probability Image Encoding.

Pixel intensities are normalized so that their squared values sum to 1, then
loaded as amplitude probabilities into 2n qubits via Qiskit's `initialize`.
Compact (only 2n qubits) but produces an arbitrary state preparation that
transpiles to deep circuits.

Reference: docs/tesi.pdf §3.1.3 / §2.2.3
"""

from __future__ import annotations

# TODO(Fase 4): implement ex novo per docs/tesi.pdf §2.2.3
#   - qpie_circuit(image: np.ndarray) -> QuantumCircuit
#   - qpie_decode(counts: dict, n: int) -> np.ndarray
# Scale freely with n; intensity resolution is shot-limited (no q parameter).
