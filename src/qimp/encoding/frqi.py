"""FRQI encoding: Flexible Representation of Quantum Images.

A 2^n × 2^n grayscale image is encoded into 2n+1 qubits: 2n position qubits
plus one color qubit. The pixel intensity is mapped to a rotation angle θ
and applied via a controlled RY gate.

Reference: docs/tesi.pdf §3.1.1
"""

from __future__ import annotations

# TODO(Fase 3): migrate from legacy/scripts/FQRI_lib2.py
#   - load_and_encode_images() -> FrqiEncoder.encode(images: np.ndarray) -> QuantumCircuit
#   - setup_quantum_circuit() -> FrqiEncoder._build_circuit(angles, n, m) -> QuantumCircuit
# Scale freely with n (image side = 2^n) and m (number of stacked images).
