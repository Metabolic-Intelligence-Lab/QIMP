"""Quantum filters for spatial processing.

Currently exposes Quantum Hadamard Edge Detection (QHED) for QPIE-encoded images.
Reference: docs/tesi.pdf §3.1.4, §2.7.3
"""

from __future__ import annotations

# TODO(Fase 4):
#   - qhed_filter(qc) -> QuantumCircuit    # H on auxiliary + decrement gate
#   - qhed_decode(counts: dict, image_size: int, num_shots: int) -> np.ndarray
