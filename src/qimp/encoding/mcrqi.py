"""MCRQI encoding: FRQI extension to RGB images (3 color qubits)."""

from __future__ import annotations

# TODO(Fase 4): docs/tesi.pdf §3.1.1 (color extension)
#   - mcrqi_circuit(image: np.ndarray) -> QuantumCircuit  # image shape (H, W, 3)
#   - mcrqi_decode(counts: dict, n: int) -> np.ndarray
