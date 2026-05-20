"""Tests for qimp.qft."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from qimp.qft import apply_inverse_qft, apply_qft


def test_apply_qft_inverse_is_identity(n_qubits: int) -> None:
    """``apply_qft`` followed by ``apply_inverse_qft`` on the same qubits must
    produce the identity (up to numerical precision).
    """
    qc = QuantumCircuit(n_qubits)
    apply_qft(qc, n_qubits)
    apply_inverse_qft(qc, n_qubits)
    unitary = Operator(qc).data
    np.testing.assert_allclose(unitary, np.eye(1 << n_qubits), atol=1e-9)


def test_apply_qft_only_touches_specified_qubits(n_qubits: int) -> None:
    """If we QFT only the first n qubits of a wider circuit, the remaining
    qubits must remain in their initial state.
    """
    extra = 2
    qc = QuantumCircuit(n_qubits + extra)
    # Prepare the "extra" qubits in |1⟩ so any accidental mutation is visible.
    for q in range(n_qubits, n_qubits + extra):
        qc.x(q)
    apply_qft(qc, n_qubits)
    sv = Statevector.from_instruction(qc)
    # Marginalise out the QFT'd qubits and check the rest is exactly |11…1⟩.
    probs = sv.probabilities([n_qubits + i for i in range(extra)])
    expected = np.zeros(1 << extra)
    expected[-1] = 1.0  # all ones
    np.testing.assert_allclose(probs, expected, atol=1e-9)


def test_apply_qft_sequence_form_matches_int_form(n_qubits: int) -> None:
    """Passing ``list(range(n))`` must produce the same operator as passing ``n``."""
    qc_int = QuantumCircuit(n_qubits + 1)
    apply_qft(qc_int, n_qubits)
    qc_seq = QuantumCircuit(n_qubits + 1)
    apply_qft(qc_seq, list(range(n_qubits)))
    np.testing.assert_allclose(Operator(qc_int).data, Operator(qc_seq).data, atol=1e-12)


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
