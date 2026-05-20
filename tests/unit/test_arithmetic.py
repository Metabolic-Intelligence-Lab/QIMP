"""Tests for qimp.processing.arithmetic."""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

from qimp.processing.arithmetic import neqr_comparator, q_add, q_sub, qc_add_1


def _set_register_to_int(qc: QuantumCircuit, qubits: list[int], value: int) -> None:
    """Apply X gates so the (little-endian) qubits encode `value`."""
    for i, q in enumerate(qubits):
        if (value >> i) & 1:
            qc.x(q)


def _read_register(sv: Statevector, qubits: list[int]) -> int:
    """The Statevector is over all qubits; find the basis index with amplitude 1."""
    amplitudes = sv.data
    # In an oracle-style circuit with classical inputs, exactly one basis state
    # has amplitude 1.
    idx = int(abs(amplitudes).argmax())
    out = 0
    for i, q in enumerate(qubits):
        if (idx >> q) & 1:
            out |= 1 << i
    return out


@pytest.mark.parametrize("a,b", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_qc_add_1_truth_table(a: int, b: int) -> None:
    a_q, b_q, cin_q, cout_q = 0, 1, 2, 3
    qc = QuantumCircuit(4)
    if a:
        qc.x(a_q)
    if b:
        qc.x(b_q)
    qc_add_1(qc, a_q, b_q, cin_q, cout_q)
    sv = Statevector.from_instruction(qc)
    sum_bit = _read_register(sv, [b_q])
    carry_bit = _read_register(sv, [cout_q])
    expected_sum = (a + b) & 1
    expected_carry = (a + b) >> 1
    assert sum_bit == expected_sum
    assert carry_bit == expected_carry


@pytest.mark.parametrize(
    "n, a_val, b_val",
    [
        (2, 0, 0),
        (2, 1, 2),
        (2, 3, 2),  # carry chain
        (3, 5, 3),
        (3, 7, 7),  # all-ones
        (4, 9, 6),
    ],
)
def test_q_add_arbitrary_width(n: int, a_val: int, b_val: int) -> None:
    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    c = QuantumRegister(n + 1, "c")
    qc = QuantumCircuit(a, b, c)
    _set_register_to_int(qc, list(range(n)), a_val)
    _set_register_to_int(qc, list(range(n, 2 * n)), b_val)
    q_add(qc, list(range(n)), list(range(n, 2 * n)), list(range(2 * n, 3 * n + 1)))
    sv = Statevector.from_instruction(qc)
    # Read b (now holds the sum's low n bits) + the final carry bit.
    sum_low = _read_register(sv, list(range(n, 2 * n)))
    final_carry = _read_register(sv, [3 * n])
    expected = a_val + b_val
    assert sum_low == (expected & ((1 << n) - 1))
    assert final_carry == ((expected >> n) & 1)


@pytest.mark.parametrize("n, a_val, b_val", [(2, 1, 3), (3, 2, 5), (3, 7, 7), (4, 9, 6)])
def test_q_sub_b_minus_a(n: int, a_val: int, b_val: int) -> None:
    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    c = QuantumRegister(n + 1, "c")
    qc = QuantumCircuit(a, b, c)
    _set_register_to_int(qc, list(range(n)), a_val)
    _set_register_to_int(qc, list(range(n, 2 * n)), b_val)
    q_sub(qc, list(range(n)), list(range(n, 2 * n)), list(range(2 * n, 3 * n + 1)))
    sv = Statevector.from_instruction(qc)
    result = _read_register(sv, list(range(n, 2 * n)))
    expected = (b_val - a_val) & ((1 << n) - 1)
    assert result == expected


@pytest.mark.parametrize(
    "n, a_val, b_val, expected_gt",
    [
        # n = 2
        (2, 3, 1, 1),
        (2, 1, 3, 0),
        (2, 2, 2, 0),
        # n = 3
        (3, 5, 4, 1),
        (3, 4, 5, 0),
        (3, 7, 0, 1),
        # n = 4 (carry-chain stress)
        (4, 15, 14, 1),
        (4, 0, 1, 0),
    ],
)
def test_neqr_comparator(n: int, a_val: int, b_val: int, expected_gt: int) -> None:
    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    c = QuantumRegister(n + 1, "c")
    gt = QuantumRegister(1, "gt")
    qc = QuantumCircuit(a, b, c, gt)
    _set_register_to_int(qc, list(range(n)), a_val)
    _set_register_to_int(qc, list(range(n, 2 * n)), b_val)
    neqr_comparator(
        qc,
        list(range(n)),
        list(range(n, 2 * n)),
        list(range(2 * n, 3 * n + 1)),
        gt_qubit=3 * n + 1,
    )
    sv = Statevector.from_instruction(qc)
    assert _read_register(sv, [3 * n + 1]) == expected_gt


def test_arithmetic_rejects_mismatched_widths() -> None:
    qc = QuantumCircuit(10)
    with pytest.raises(ValueError, match="same length"):
        q_add(qc, [0, 1], [2, 3, 4], [5, 6, 7])
    with pytest.raises(ValueError, match="len"):
        q_add(qc, [0, 1], [2, 3], [4, 5])  # c should be n+1
