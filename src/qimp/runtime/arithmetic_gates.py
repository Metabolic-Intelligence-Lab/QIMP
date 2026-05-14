"""Cyclic increment / decrement gates on a little-endian qubit register.

Used by `qimp.processing.geometric.pos_shift` and
`qimp.processing.filters.qhed_filter`. Factored out to keep a single
implementation.
"""

from __future__ import annotations

from qiskit import QuantumCircuit

__all__ = ["cyclic_decrement", "cyclic_increment"]


def cyclic_increment(qc: QuantumCircuit, qubits: list[int]) -> None:
    """Cyclic increment: ``|k⟩ → |k + 1 mod 2^N⟩`` on a little-endian register.

    Standard Toffoli-tower implementation: ``O(N²)`` gates after decomposition.
    Mutates `qc` in place.
    """
    if not qubits:
        raise ValueError("qubits list is empty")
    n = len(qubits)
    for k in range(n - 1, 0, -1):
        qc.mcx(qubits[:k], qubits[k])
    qc.x(qubits[0])


def cyclic_decrement(qc: QuantumCircuit, qubits: list[int]) -> None:
    """Cyclic decrement: ``|k⟩ → |k − 1 mod 2^N⟩``.

    Inverse of `cyclic_increment`. Mutates `qc` in place.
    """
    if not qubits:
        raise ValueError("qubits list is empty")
    n = len(qubits)
    qc.x(qubits[0])
    for k in range(1, n):
        qc.mcx(qubits[:k], qubits[k])
