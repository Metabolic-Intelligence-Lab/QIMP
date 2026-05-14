"""Tests for qimp.qft."""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

from qimp.qft import apply_inverse_qft, apply_qft


def test_apply_qft_int_form(n_qubits: int) -> None:
    qc = QuantumCircuit(n_qubits + 2)  # extras to verify only first n are touched
    apply_qft(qc, n_qubits)
    assert qc.num_qubits == n_qubits + 2


def test_apply_qft_sequence_form() -> None:
    qc = QuantumCircuit(5)
    apply_qft(qc, [1, 3, 4])
    assert qc.depth() >= 1


def test_inverse_qft_label() -> None:
    qc = QuantumCircuit(3)
    apply_inverse_qft(qc, 3)
    names = [instr.operation.name.lower() for instr in qc.data]
    assert any("qft" in n for n in names)


def test_apply_qft_rejects_out_of_range() -> None:
    qc = QuantumCircuit(2)
    with pytest.raises(ValueError):
        apply_qft(qc, 5)
    with pytest.raises(ValueError):
        apply_qft(qc, [0, 5])


def test_apply_qft_rejects_empty() -> None:
    qc = QuantumCircuit(2)
    with pytest.raises(ValueError):
        apply_qft(qc, [])
