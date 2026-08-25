"""Tests for qimp.processing.arithmetic."""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

from qimp.processing.arithmetic import (
    neqr_comparator,
    q_add,
    q_add_ctrl,
    q_add_ctrl_inv,
    q_add_inv,
    q_add_sub_ctrl,
    q_add_sub_ctrl_inv,
    q_div_general,
    q_div_nonrestoring,
    q_div_nonrestoring_inv,
    q_div_restoring,
    q_div_restoring_inv,
    q_mul_const,
    q_mul_const_inv,
    q_sub,
    q_sub_ctrl,
    q_sub_ctrl_inv,
    q_sub_inv,
    qc_add_1,
)


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


# ---------------------------------------------------------------------------
# Stage A.1 — q_add_ctrl, q_add_inv, q_sub_inv
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, a_val, b_val, ctrl_val",
    [
        (3, 5, 3, 0),
        (3, 5, 3, 1),
        (3, 7, 7, 1),
        (4, 9, 6, 1),
        (4, 9, 6, 0),
    ],
)
def test_q_add_ctrl_truth_table(n: int, a_val: int, b_val: int, ctrl_val: int) -> None:
    """Controlled adder reproduces q_add when ctrl=1 and identity when ctrl=0."""
    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    c = QuantumRegister(n + 1, "c")
    ctrl = QuantumRegister(1, "ctrl")
    qc = QuantumCircuit(a, b, c, ctrl)
    a_idx = list(range(n))
    b_idx = list(range(n, 2 * n))
    c_idx = list(range(2 * n, 3 * n + 1))
    ctrl_idx = 3 * n + 1
    _set_register_to_int(qc, a_idx, a_val)
    _set_register_to_int(qc, b_idx, b_val)
    if ctrl_val:
        qc.x(ctrl_idx)
    q_add_ctrl(qc, ctrl_idx, a_idx, b_idx, c_idx)
    sv = Statevector.from_instruction(qc)
    sum_low = _read_register(sv, b_idx)
    final_carry = _read_register(sv, [c_idx[-1]])
    if ctrl_val:
        expected = a_val + b_val
        assert sum_low == (expected & ((1 << n) - 1))
        assert final_carry == ((expected >> n) & 1)
    else:
        assert sum_low == b_val
        assert final_carry == 0


@pytest.mark.parametrize("n, a_val, b_val", [(3, 5, 3), (3, 7, 7), (4, 9, 6)])
def test_q_add_inv_round_trip(n: int, a_val: int, b_val: int) -> None:
    """q_add followed by q_add_inv restores both b and the carry register."""
    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    c = QuantumRegister(n + 1, "c")
    qc = QuantumCircuit(a, b, c)
    a_idx = list(range(n))
    b_idx = list(range(n, 2 * n))
    c_idx = list(range(2 * n, 3 * n + 1))
    _set_register_to_int(qc, a_idx, a_val)
    _set_register_to_int(qc, b_idx, b_val)
    q_add(qc, a_idx, b_idx, c_idx)
    q_add_inv(qc, a_idx, b_idx, c_idx)
    sv = Statevector.from_instruction(qc)
    # Round-trip: b should be back to b_val, carry register should be all zero.
    assert _read_register(sv, b_idx) == b_val
    for q in c_idx:
        assert _read_register(sv, [q]) == 0


@pytest.mark.parametrize("n, a_val, b_val", [(3, 5, 3), (3, 7, 7), (4, 9, 6)])
def test_q_sub_inv_round_trip(n: int, a_val: int, b_val: int) -> None:
    """q_sub followed by q_sub_inv restores b, the carry register, and a."""
    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    c = QuantumRegister(n + 1, "c")
    qc = QuantumCircuit(a, b, c)
    a_idx = list(range(n))
    b_idx = list(range(n, 2 * n))
    c_idx = list(range(2 * n, 3 * n + 1))
    _set_register_to_int(qc, a_idx, a_val)
    _set_register_to_int(qc, b_idx, b_val)
    q_sub(qc, a_idx, b_idx, c_idx)
    q_sub_inv(qc, a_idx, b_idx, c_idx)
    sv = Statevector.from_instruction(qc)
    assert _read_register(sv, a_idx) == a_val
    assert _read_register(sv, b_idx) == b_val
    for q in c_idx:
        assert _read_register(sv, [q]) == 0


# ---------------------------------------------------------------------------
# Stage A.2 — q_mul_const (multiply by classical constant via shift-add)
# ---------------------------------------------------------------------------


def _int_to_bits(value: int, width: int) -> list[int]:
    return [(value >> i) & 1 for i in range(width)]


@pytest.mark.parametrize(
    "q, b_val, k_val, m",
    [
        (3, 5, 0, 3),  # k=0: accum unchanged
        (3, 5, 1, 3),  # k=1: trivial
        (3, 5, 2, 3),  # k=2: single shift
        (3, 5, 3, 3),  # k=3: two shifts (1+2)
        (3, 5, 7, 3),  # k=7: three shifts
        (3, 0, 5, 3),  # b=0: accum unchanged
        (3, 7, 5, 3),  # 7 * 5 = 35
        (4, 9, 6, 4),  # 9 * 6 = 54, fits in 4+4=8 bits
        (4, 3, 3, 4),  # popcount(3) = 2; 25-qubit statevector
    ],
)
def test_q_mul_const_against_python(q: int, b_val: int, k_val: int, m: int) -> None:
    """q_mul_const(b, k, accum) sets accum[..] = b * k mod 2^(q+m)."""
    k_bits = _int_to_bits(k_val, m)
    popcount = sum(k_bits)
    n_carry = max(1, popcount) * (q + 2)
    b = QuantumRegister(q, "b")
    accum = QuantumRegister(q + m, "accum")
    c = QuantumRegister(n_carry, "c")
    guard = QuantumRegister(1, "guard")
    qc = QuantumCircuit(b, accum, c, guard)
    b_idx = list(range(q))
    accum_idx = list(range(q, 2 * q + m))
    c_idx = list(range(2 * q + m, 2 * q + m + n_carry))
    guard_idx = 2 * q + m + n_carry
    _set_register_to_int(qc, b_idx, b_val)
    q_mul_const(qc, b_idx, k_bits, accum_idx, c_idx, guard_idx)
    sv = Statevector.from_instruction(qc)
    got = _read_register(sv, accum_idx)
    expected = (b_val * k_val) & ((1 << (q + m)) - 1)
    assert got == expected, f"q={q} b={b_val} k={k_val}: got {got}, expected {expected}"
    # b is preserved; guard restored to 0.
    assert _read_register(sv, b_idx) == b_val
    assert _read_register(sv, [guard_idx]) == 0


@pytest.mark.parametrize(
    "q, b_val, k_val, m",
    [(3, 5, 3, 3), (3, 7, 7, 3), (4, 9, 6, 4)],
)
def test_q_mul_const_inv_round_trip(q: int, b_val: int, k_val: int, m: int) -> None:
    """q_mul_const followed by q_mul_const_inv restores accum, c, and guard."""
    k_bits = _int_to_bits(k_val, m)
    popcount = sum(k_bits)
    n_carry = max(1, popcount) * (q + 2)
    b = QuantumRegister(q, "b")
    accum = QuantumRegister(q + m, "accum")
    c = QuantumRegister(n_carry, "c")
    guard = QuantumRegister(1, "guard")
    qc = QuantumCircuit(b, accum, c, guard)
    b_idx = list(range(q))
    accum_idx = list(range(q, 2 * q + m))
    c_idx = list(range(2 * q + m, 2 * q + m + n_carry))
    guard_idx = 2 * q + m + n_carry
    _set_register_to_int(qc, b_idx, b_val)
    q_mul_const(qc, b_idx, k_bits, accum_idx, c_idx, guard_idx)
    q_mul_const_inv(qc, b_idx, k_bits, accum_idx, c_idx, guard_idx)
    sv = Statevector.from_instruction(qc)
    # All registers restored: b preserved, accum=0, c=0, guard=0.
    assert _read_register(sv, b_idx) == b_val
    assert _read_register(sv, accum_idx) == 0
    for q_idx in c_idx:
        assert _read_register(sv, [q_idx]) == 0
    assert _read_register(sv, [guard_idx]) == 0


# ---------------------------------------------------------------------------
# Stage A.3 — q_sub_ctrl, q_div_restoring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, a_val, b_val, ctrl_val",
    [
        (3, 5, 3, 0),  # ctrl=0: identity
        (3, 5, 3, 1),  # ctrl=1: b ← b - a = -2 mod 8 = 6
        (3, 7, 7, 1),  # 7 - 7 = 0
        (3, 1, 5, 1),  # 5 - 1 = 4
    ],
)
def test_q_sub_ctrl_truth_table(n: int, a_val: int, b_val: int, ctrl_val: int) -> None:
    """q_sub_ctrl reproduces q_sub when ctrl=1; identity when ctrl=0."""
    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    c = QuantumRegister(n + 1, "c")
    ctrl = QuantumRegister(1, "ctrl")
    qc = QuantumCircuit(a, b, c, ctrl)
    a_idx = list(range(n))
    b_idx = list(range(n, 2 * n))
    c_idx = list(range(2 * n, 3 * n + 1))
    ctrl_idx = 3 * n + 1
    _set_register_to_int(qc, a_idx, a_val)
    _set_register_to_int(qc, b_idx, b_val)
    if ctrl_val:
        qc.x(ctrl_idx)
    q_sub_ctrl(qc, ctrl_idx, a_idx, b_idx, c_idx)
    sv = Statevector.from_instruction(qc)
    expected = b_val - a_val & (1 << n) - 1 if ctrl_val else b_val
    assert _read_register(sv, b_idx) == expected
    # a is restored regardless of ctrl
    assert _read_register(sv, a_idx) == a_val


def _make_div_circuit(q: int, dividend_val: int, divisor_val: int):
    """Construct a quantum circuit that runs q_div_restoring with the
    given (q, dividend, divisor) and returns the (qc, register indices)."""
    needed_c = (q + 1) * (q + 2)
    div = QuantumRegister(q, "div")
    ds = QuantumRegister(q, "ds")
    quo = QuantumRegister(q, "quo")
    work = QuantumRegister(q, "work")
    pad = QuantumRegister(1, "pad")
    c = QuantumRegister(needed_c, "c")
    flag = QuantumRegister(1, "flag")
    qc = QuantumCircuit(div, ds, quo, work, pad, c, flag)
    div_idx = list(range(q))
    ds_idx = list(range(q, 2 * q))
    quo_idx = list(range(2 * q, 3 * q))
    work_idx = list(range(3 * q, 4 * q))
    pad_idx = 4 * q
    c_idx = list(range(4 * q + 1, 4 * q + 1 + needed_c))
    flag_idx = 4 * q + 1 + needed_c
    _set_register_to_int(qc, div_idx, dividend_val)
    _set_register_to_int(qc, ds_idx, divisor_val)
    q_div_restoring(qc, div_idx, ds_idx, quo_idx, work_idx, pad_idx, c_idx, flag_idx)
    return qc, dict(
        div=div_idx, ds=ds_idx, quo=quo_idx, work=work_idx, pad=pad_idx, c=c_idx, flag=flag_idx
    )


@pytest.mark.parametrize(
    "dividend, divisor",
    [(a, b) for a in range(4) for b in range(1, 4)],  # q=2 exhaustive, divisor≠0
)
def test_q_div_restoring_exhaustive_q2(dividend: int, divisor: int) -> None:
    """At q=2 (≤22 qubits) verify dividend // divisor and dividend % divisor."""
    q = 2
    qc, idx = _make_div_circuit(q, dividend, divisor)
    sv = Statevector.from_instruction(qc)
    got_quo = _read_register(sv, idx["quo"])
    got_rem = _read_register(sv, idx["div"])
    exp_quo = dividend // divisor
    exp_rem = dividend % divisor
    assert got_quo == exp_quo, (
        f"q={q} {dividend}/{divisor}: quotient got {got_quo}, expected {exp_quo}"
    )
    assert got_rem == exp_rem, (
        f"q={q} {dividend}/{divisor}: remainder got {got_rem}, expected {exp_rem}"
    )
    # Divisor preserved, work restored to 0, pad restored to 0, flag = 0.
    assert _read_register(sv, idx["ds"]) == divisor
    assert _read_register(sv, idx["work"]) == 0
    assert _read_register(sv, [idx["pad"]]) == 0
    assert _read_register(sv, [idx["flag"]]) == 0


@pytest.mark.parametrize("dividend", [0, 1, 2, 3])
def test_q_div_restoring_zero_divisor_flag_q2(dividend: int) -> None:
    """At q=2 with divisor=0: div_zero_flag must be set; quotient/remainder
    are undefined but we just verify the flag mechanic."""
    q = 2
    qc, idx = _make_div_circuit(q, dividend, 0)
    sv = Statevector.from_instruction(qc)
    assert _read_register(sv, [idx["flag"]]) == 1


@pytest.mark.parametrize(
    "n, a_val, b_val, ctrl_val",
    [
        (3, 5, 3, 0),
        (3, 5, 3, 1),
        (3, 7, 7, 1),
        (4, 9, 6, 1),
    ],
)
def test_q_add_ctrl_inv_round_trip(n: int, a_val: int, b_val: int, ctrl_val: int) -> None:
    """q_add_ctrl followed by q_add_ctrl_inv restores b, a, ctrl, and the
    carry register regardless of ctrl_val."""
    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    c = QuantumRegister(n + 1, "c")
    ctrl = QuantumRegister(1, "ctrl")
    qc = QuantumCircuit(a, b, c, ctrl)
    a_idx = list(range(n))
    b_idx = list(range(n, 2 * n))
    c_idx = list(range(2 * n, 3 * n + 1))
    ctrl_idx = 3 * n + 1
    _set_register_to_int(qc, a_idx, a_val)
    _set_register_to_int(qc, b_idx, b_val)
    if ctrl_val:
        qc.x(ctrl_idx)
    q_add_ctrl(qc, ctrl_idx, a_idx, b_idx, c_idx)
    q_add_ctrl_inv(qc, ctrl_idx, a_idx, b_idx, c_idx)
    sv = Statevector.from_instruction(qc)
    assert _read_register(sv, a_idx) == a_val
    assert _read_register(sv, b_idx) == b_val
    for q in c_idx:
        assert _read_register(sv, [q]) == 0
    assert _read_register(sv, [ctrl_idx]) == ctrl_val


@pytest.mark.parametrize(
    "n, a_val, b_val, ctrl_val",
    [
        (3, 5, 3, 0),
        (3, 5, 3, 1),
        (3, 7, 7, 1),
        (3, 1, 5, 1),
    ],
)
def test_q_sub_ctrl_inv_round_trip(n: int, a_val: int, b_val: int, ctrl_val: int) -> None:
    """q_sub_ctrl followed by q_sub_ctrl_inv restores everything."""
    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    c = QuantumRegister(n + 1, "c")
    ctrl = QuantumRegister(1, "ctrl")
    qc = QuantumCircuit(a, b, c, ctrl)
    a_idx = list(range(n))
    b_idx = list(range(n, 2 * n))
    c_idx = list(range(2 * n, 3 * n + 1))
    ctrl_idx = 3 * n + 1
    _set_register_to_int(qc, a_idx, a_val)
    _set_register_to_int(qc, b_idx, b_val)
    if ctrl_val:
        qc.x(ctrl_idx)
    q_sub_ctrl(qc, ctrl_idx, a_idx, b_idx, c_idx)
    q_sub_ctrl_inv(qc, ctrl_idx, a_idx, b_idx, c_idx)
    sv = Statevector.from_instruction(qc)
    assert _read_register(sv, a_idx) == a_val
    assert _read_register(sv, b_idx) == b_val
    for q in c_idx:
        assert _read_register(sv, [q]) == 0
    assert _read_register(sv, [ctrl_idx]) == ctrl_val


def _make_div_general_circuit(n: int, m: int, dividend_val: int, divisor_val: int):
    """Build a circuit that runs q_div_general at (n, m, quotient=n)."""
    needed_c = (n + 1) * (m + 2)
    div = QuantumRegister(n, "div")
    ds = QuantumRegister(m, "ds")
    quo = QuantumRegister(n, "quo")
    work = QuantumRegister(m, "work")
    pad = QuantumRegister(1, "pad")
    c = QuantumRegister(needed_c, "c")
    flag = QuantumRegister(1, "flag")
    qc = QuantumCircuit(div, ds, quo, work, pad, c, flag)
    div_idx = list(range(n))
    ds_idx = list(range(n, n + m))
    quo_idx = list(range(n + m, 2 * n + m))
    work_idx = list(range(2 * n + m, 2 * n + 2 * m))
    pad_idx = 2 * n + 2 * m
    c_idx = list(range(2 * n + 2 * m + 1, 2 * n + 2 * m + 1 + needed_c))
    flag_idx = 2 * n + 2 * m + 1 + needed_c
    _set_register_to_int(qc, div_idx, dividend_val)
    _set_register_to_int(qc, ds_idx, divisor_val)
    q_div_general(qc, div_idx, ds_idx, quo_idx, work_idx, pad_idx, c_idx, flag_idx)
    return qc, dict(
        div=div_idx, ds=ds_idx, quo=quo_idx, work=work_idx, pad=pad_idx, c=c_idx, flag=flag_idx
    )


@pytest.mark.parametrize(
    "n, m, dividend, divisor",
    [
        # Square case n=m=2 (statevector-tractable at 22 qubits):
        # should match q_div_restoring exactly.
        (2, 2, 3, 1),
        (2, 2, 3, 2),
        (2, 2, 2, 3),
        # Smallest non-square (n=3, m=1): 17 qubits, only divisor=1
        # is non-trivial (m=1 means divisor ∈ {0, 1}, the 0 case is
        # the div-by-zero flag path).
        (3, 1, 0, 1),
        (3, 1, 1, 1),
        (3, 1, 5, 1),
        (3, 1, 7, 1),
    ],
)
def test_q_div_general_correctness(n: int, m: int, dividend: int, divisor: int) -> None:
    """Verify q_div_general at (n, m, quotient=n): dividend // divisor in
    the quotient register, dividend mod divisor in the dividend register.

    Note: larger non-square configurations like (n=3, m=2) push the
    statevector past 28 qubits (~10 GB with Qiskit's transpilation
    overhead) and are statevector-infeasible on the development laptop;
    they are exercised via the AerSimulator(method='mps') path in the
    Stage E QAE demo (out of scope here).
    """
    qc, idx = _make_div_general_circuit(n, m, dividend, divisor)
    sv = Statevector.from_instruction(qc)
    got_quo = _read_register(sv, idx["quo"])
    got_rem = _read_register(sv, idx["div"])
    exp_quo = dividend // divisor
    exp_rem = dividend % divisor
    assert got_quo == exp_quo, (
        f"q_div_general({n},{m}) {dividend}/{divisor}: quotient got {got_quo}, expected {exp_quo}"
    )
    assert got_rem == exp_rem, (
        f"q_div_general({n},{m}) {dividend}/{divisor}: remainder got {got_rem}, expected {exp_rem}"
    )
    # Divisor preserved, work restored, pad restored.
    assert _read_register(sv, idx["ds"]) == divisor
    assert _read_register(sv, idx["work"]) == 0
    assert _read_register(sv, [idx["pad"]]) == 0


@pytest.mark.parametrize(
    "dividend, divisor",
    [(a, b) for a in range(4) for b in range(0, 4)],  # q=2, include b=0
)
def test_q_div_restoring_inv_round_trip(dividend: int, divisor: int) -> None:
    """q_div_restoring followed by q_div_restoring_inv restores every
    register to its pre-call state (including the div_zero_flag for the
    divisor=0 case)."""
    q = 2
    qc, idx = _make_div_circuit(q, dividend, divisor)
    q_div_restoring_inv(
        qc,
        dividend_qubits=idx["div"],
        divisor_qubits=idx["ds"],
        quotient_qubits=idx["quo"],
        work_qubits=idx["work"],
        divisor_pad_qubit=idx["pad"],
        c_qubits=idx["c"],
        div_zero_flag=idx["flag"],
    )
    sv = Statevector.from_instruction(qc)
    # Dividend restored
    assert _read_register(sv, idx["div"]) == dividend, (
        f"dividend not restored: got {_read_register(sv, idx['div'])} expected {dividend}"
    )
    # Divisor preserved
    assert _read_register(sv, idx["ds"]) == divisor
    # Quotient back to 0
    assert _read_register(sv, idx["quo"]) == 0
    # Work back to 0
    assert _read_register(sv, idx["work"]) == 0
    # Pad back to 0
    assert _read_register(sv, [idx["pad"]]) == 0
    # Flag back to 0 (self-inverse div-zero detection)
    assert _read_register(sv, [idx["flag"]]) == 0
    # All carry qubits back to 0
    for q_idx in idx["c"]:
        assert _read_register(sv, [q_idx]) == 0, f"c[{q_idx}] not clean"


# ---------------------------------------------------------------------------
# Non-restoring divider tests
# ---------------------------------------------------------------------------


def _make_nonrestoring_div_circuit(q: int, dividend_val: int, divisor_val: int):
    """Same layout as _make_div_circuit but calls q_div_nonrestoring."""
    needed_c = (q + 1) * (q + 2)
    div = QuantumRegister(q, "div")
    ds = QuantumRegister(q, "ds")
    quo = QuantumRegister(q, "quo")
    work = QuantumRegister(q, "work")
    pad = QuantumRegister(1, "pad")
    c = QuantumRegister(needed_c, "c")
    flag = QuantumRegister(1, "flag")
    qc = QuantumCircuit(div, ds, quo, work, pad, c, flag)
    div_idx = list(range(q))
    ds_idx = list(range(q, 2 * q))
    quo_idx = list(range(2 * q, 3 * q))
    work_idx = list(range(3 * q, 4 * q))
    pad_idx = 4 * q
    c_idx = list(range(4 * q + 1, 4 * q + 1 + needed_c))
    flag_idx = 4 * q + 1 + needed_c
    _set_register_to_int(qc, div_idx, dividend_val)
    _set_register_to_int(qc, ds_idx, divisor_val)
    q_div_nonrestoring(qc, div_idx, ds_idx, quo_idx, work_idx, pad_idx, c_idx, flag_idx)
    return qc, dict(
        div=div_idx, ds=ds_idx, quo=quo_idx, work=work_idx, pad=pad_idx, c=c_idx, flag=flag_idx
    )


@pytest.mark.parametrize(
    "dividend, divisor",
    [(a, b) for a in range(4) for b in range(1, 4)],  # q=2, divisor≠0
)
def test_q_div_nonrestoring_exhaustive_q2(dividend: int, divisor: int) -> None:
    """Exhaustive bit-exact verification of q_div_nonrestoring at q=2."""
    q = 2
    qc, idx = _make_nonrestoring_div_circuit(q, dividend, divisor)
    sv = Statevector.from_instruction(qc)
    got_quo = _read_register(sv, idx["quo"])
    got_rem = _read_register(sv, idx["div"])
    exp_quo = dividend // divisor
    exp_rem = dividend % divisor
    assert got_quo == exp_quo, (
        f"q={q} {dividend}/{divisor}: quotient got {got_quo}, expected {exp_quo}"
    )
    assert got_rem == exp_rem, (
        f"q={q} {dividend}/{divisor}: remainder got {got_rem}, expected {exp_rem}"
    )
    # Divisor preserved, work restored to 0, pad restored to 0, flag = 0.
    assert _read_register(sv, idx["ds"]) == divisor
    assert _read_register(sv, idx["work"]) == 0
    assert _read_register(sv, [idx["pad"]]) == 0
    assert _read_register(sv, [idx["flag"]]) == 0


@pytest.mark.parametrize("dividend", [0, 1, 2, 3])
def test_q_div_nonrestoring_zero_divisor_flag_q2(dividend: int) -> None:
    """At q=2 with divisor=0, div_zero_flag must be set."""
    q = 2
    qc, idx = _make_nonrestoring_div_circuit(q, dividend, 0)
    sv = Statevector.from_instruction(qc)
    assert _read_register(sv, [idx["flag"]]) == 1


@pytest.mark.parametrize(
    "dividend, divisor",
    [(a, b) for a in range(4) for b in range(0, 4)],
)
def test_q_div_nonrestoring_inv_round_trip(dividend: int, divisor: int) -> None:
    """q_div_nonrestoring followed by q_div_nonrestoring_inv restores all
    registers (dividend, quotient, work, c, pad, flag) to their pre-call
    state."""
    q = 2
    qc, idx = _make_nonrestoring_div_circuit(q, dividend, divisor)
    q_div_nonrestoring_inv(
        qc,
        dividend_qubits=idx["div"],
        divisor_qubits=idx["ds"],
        quotient_qubits=idx["quo"],
        work_qubits=idx["work"],
        divisor_pad_qubit=idx["pad"],
        c_qubits=idx["c"],
        div_zero_flag=idx["flag"],
    )
    sv = Statevector.from_instruction(qc)
    assert _read_register(sv, idx["div"]) == dividend
    assert _read_register(sv, idx["ds"]) == divisor
    assert _read_register(sv, idx["quo"]) == 0
    assert _read_register(sv, idx["work"]) == 0
    assert _read_register(sv, [idx["pad"]]) == 0
    assert _read_register(sv, [idx["flag"]]) == 0
    for q_idx in idx["c"]:
        assert _read_register(sv, [q_idx]) == 0, f"c[{q_idx}] not clean"


@pytest.mark.parametrize(
    "n, a_val, b_val, ctrl_val",
    [
        (3, 5, 3, 0),  # ctrl=0 → add: b = 3 + 5 = 8 (mod 16) = 8
        (3, 5, 3, 1),  # ctrl=1 → sub: b = 3 - 5 = -2 mod 16 = 14
        (3, 7, 7, 0),  # ctrl=0 → add: b = 7 + 7 = 14
        (3, 7, 7, 1),  # ctrl=1 → sub: b = 0
        (4, 9, 6, 0),  # ctrl=0 → add: b = 6 + 9 = 15
        (4, 9, 6, 1),  # ctrl=1 → sub: b = 6 - 9 = -3 mod 32 = 29
    ],
)
def test_q_add_sub_ctrl_truth_table(n: int, a_val: int, b_val: int, ctrl_val: int) -> None:
    """q_add_sub_ctrl: ctrl=0 → b ← b + a; ctrl=1 → b ← b − a (mod 2^(n+1)).
    `a` is restored regardless of ctrl."""
    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    c = QuantumRegister(n + 1, "c")
    ctrl = QuantumRegister(1, "ctrl")
    qc = QuantumCircuit(a, b, c, ctrl)
    a_idx = list(range(n))
    b_idx = list(range(n, 2 * n))
    c_idx = list(range(2 * n, 3 * n + 1))
    ctrl_idx = 3 * n + 1
    _set_register_to_int(qc, a_idx, a_val)
    _set_register_to_int(qc, b_idx, b_val)
    if ctrl_val:
        qc.x(ctrl_idx)
    q_add_sub_ctrl(qc, ctrl_idx, a_idx, b_idx, c_idx)
    sv = Statevector.from_instruction(qc)
    expected = b_val - a_val & (1 << n + 1) - 1 if ctrl_val else b_val + a_val & (1 << n + 1) - 1
    # The b register only holds the low n bits of the (n+1)-bit result;
    # the carry-out goes to c[-1].
    assert _read_register(sv, b_idx) == (expected & ((1 << n) - 1))
    assert _read_register(sv, a_idx) == a_val


@pytest.mark.parametrize(
    "n, a_val, b_val, ctrl_val",
    [(3, 5, 3, 0), (3, 5, 3, 1), (4, 9, 6, 1), (4, 0, 7, 0)],
)
def test_q_add_sub_ctrl_inv_round_trip(n: int, a_val: int, b_val: int, ctrl_val: int) -> None:
    """q_add_sub_ctrl followed by q_add_sub_ctrl_inv restores b and c."""
    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    c = QuantumRegister(n + 1, "c")
    ctrl = QuantumRegister(1, "ctrl")
    qc = QuantumCircuit(a, b, c, ctrl)
    a_idx = list(range(n))
    b_idx = list(range(n, 2 * n))
    c_idx = list(range(2 * n, 3 * n + 1))
    ctrl_idx = 3 * n + 1
    _set_register_to_int(qc, a_idx, a_val)
    _set_register_to_int(qc, b_idx, b_val)
    if ctrl_val:
        qc.x(ctrl_idx)
    q_add_sub_ctrl(qc, ctrl_idx, a_idx, b_idx, c_idx)
    q_add_sub_ctrl_inv(qc, ctrl_idx, a_idx, b_idx, c_idx)
    sv = Statevector.from_instruction(qc)
    assert _read_register(sv, a_idx) == a_val
    assert _read_register(sv, b_idx) == b_val
    for q_idx in c_idx:
        assert _read_register(sv, [q_idx]) == 0
