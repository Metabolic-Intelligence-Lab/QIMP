"""NEQR encoding: Novel Enhanced Quantum Representation.

A 2^n × 2^n grayscale image with intensity resolution 2^q is encoded into
2n + q qubits. Intensity is stored in a separate register (not a rotation
angle), enabling exact retrieval and arithmetic operations.

Reference: docs/tesi.pdf §3.1.2 / §2.2.2
"""

from __future__ import annotations

# TODO(Fase 4): implement ex novo per docs/tesi.pdf §2.2.2
#   - neqr_circuit(image: np.ndarray, q: int) -> QuantumCircuit
#   - neqr_decode(counts: dict, n: int, q: int) -> np.ndarray
# Scale freely with n and q.
