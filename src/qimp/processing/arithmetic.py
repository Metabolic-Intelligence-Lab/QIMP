"""Quantum arithmetic primitives for NEQR-encoded images.

Public ops:
- ``qc_add_1(qc, a, b, c_out)`` — one-bit full adder ``b ← (a + b)`` with
  carry-out qubit ``c_out``.
- ``q_add(qc, a_qubits, b_qubits, c_qubits)`` — ripple-carry adder
  ``b ← (a + b)`` for arbitrary-width little-endian registers, using one
  auxiliary qubit per bit (``len(c_qubits) == len(a_qubits) + 1``).
- ``q_add_inv(qc, …)`` — exact inverse of ``q_add``; restores carry register.
- ``q_add_ctrl(qc, ctrl, …)`` — controlled adder ``b ← b + (ctrl ? a : 0)``.
- ``q_sub(qc, a_qubits, b_qubits, c_qubits)`` — wrap of ``q_add`` with the
  two's-complement of ``a``: ``b ← b - a``.
- ``q_sub_inv(qc, …)`` — exact inverse of ``q_sub``.
- ``q_mul_const(qc, b_qubits, k_bits, accum_qubits, c_qubits)`` —
  multiply ``b`` by a classical constant (bit pattern ``k_bits``) into the
  accumulator via shift-add.
- ``q_div_restoring(qc, …)`` — fixed-point unsigned restoring division.
- ``neqr_comparator(qc, a_qubits, b_qubits, c_qubits, gt_qubit)`` — sets
  ``gt_qubit = 1`` iff ``a > b``.

Conventions
-----------
All registers are **little-endian** lists of qubit indices (``[lsb, …, msb]``).
`q_add` follows the standard ripple-carry construction with one carry qubit per
input bit plus one final-out qubit (so ``len(c_qubits) = n + 1``).

Position-register independence
------------------------------
Every primitive in this module acts only on the explicit qubit-index lists
passed in. None of them touches the NEQR position register, so they all
compose uniformly across a position-superposition: if the position register
holds Σ_p |p⟩ and the intensity register holds |I(p)⟩ for the addressed
pixel, the primitives apply the same arithmetic transformation on |I(p)⟩
for every p in the superposition.

Reference: docs/tesi.pdf §3.1.2 (NEQR processing).
"""

from __future__ import annotations

from collections.abc import Sequence

from qiskit import QuantumCircuit

__all__ = [
    "neqr_comparator",
    "q_add",
    "q_add_ctrl",
    "q_add_ctrl_inv",
    "q_add_inv",
    "q_div",
    "q_div_general",
    "q_div_restoring",
    "q_div_restoring_inv",
    "q_mul_const",
    "q_mul_const_inv",
    "q_sub",
    "q_sub_ctrl",
    "q_sub_ctrl_inv",
    "q_sub_inv",
    "qc_add_1",
    "qc_add_1_ctrl",
    "qc_add_1_ctrl_inv",
    "qc_add_1_inv",
]


def qc_add_1(qc: QuantumCircuit, a: int, b: int, c_in: int, c_out: int) -> QuantumCircuit:
    """Single-bit full adder: ``b ← a ⊕ b ⊕ c_in``, ``c_out ← MAJ(a, b, c_in)``.

    `a` and `c_in` are read; `b` is overwritten with the new sum bit; `c_out`
    is set to the new carry.
    """
    # c_out ← MAJ(a, b, c_in); b ← a ⊕ b ⊕ c_in. `a` and `c_in` are preserved.
    qc.ccx(a, b, c_out)
    qc.cx(a, b)
    qc.ccx(b, c_in, c_out)
    qc.cx(c_in, b)
    return qc


def _check_registers(
    a_qubits: Sequence[int], b_qubits: Sequence[int], c_qubits: Sequence[int]
) -> None:
    if len(a_qubits) != len(b_qubits):
        raise ValueError(f"a and b must have same length, got {len(a_qubits)} vs {len(b_qubits)}")
    if len(c_qubits) != len(a_qubits) + 1:
        raise ValueError(f"c must have len(a)+1 = {len(a_qubits) + 1} qubits, got {len(c_qubits)}")


def q_add(
    qc: QuantumCircuit,
    a_qubits: Sequence[int],
    b_qubits: Sequence[int],
    c_qubits: Sequence[int],
) -> QuantumCircuit:
    """Ripple-carry adder: ``b ← (a + b) mod 2^(n+1)``.

    `c_qubits` provides the n+1 carry qubits (initially expected to be |0⟩;
    end state holds the final carry chain in `c_qubits[-1]`).

    Complexity: O(n) full-adder stages.
    """
    _check_registers(a_qubits, b_qubits, c_qubits)
    n = len(a_qubits)
    for i in range(n):
        qc_add_1(qc, a_qubits[i], b_qubits[i], c_qubits[i], c_qubits[i + 1])
    return qc


def _bitwise_not(qc: QuantumCircuit, qubits: Sequence[int]) -> None:
    for q in qubits:
        qc.x(q)


def q_sub(
    qc: QuantumCircuit,
    a_qubits: Sequence[int],
    b_qubits: Sequence[int],
    c_qubits: Sequence[int],
) -> QuantumCircuit:
    """Subtractor via two's-complement: ``b ← (b - a) mod 2^(n+1)``.

    Implemented as ``b ← b + (~a + 1)`` using the same ``q_add`` plumbing.
    On exit:

    - ``a_qubits`` are restored to their input state (bitwise-NOT reversed).
    - ``c_qubits[0]`` is restored to ``|0⟩`` (the +1 preload is uncomputed).
    - ``c_qubits[1..n]`` hold the carry chain produced by the addition;
      ``c_qubits[-1]`` is the final borrow bit (used by ``neqr_comparator``).
    """
    _check_registers(a_qubits, b_qubits, c_qubits)
    # ~a: flip every a bit.
    _bitwise_not(qc, a_qubits)
    # +1: pre-load the carry-in of bit 0. qc_add_1 reads c_in but never
    # writes to it, so after q_add this qubit still holds |1⟩.
    qc.x(c_qubits[0])
    q_add(qc, a_qubits, b_qubits, c_qubits)
    # Restore ``a`` and the +1 preload so the only persistent side effects
    # are the new ``b`` (= b - a) and the carry chain on c_qubits[1..n].
    _bitwise_not(qc, a_qubits)
    qc.x(c_qubits[0])
    return qc


def neqr_comparator(
    qc: QuantumCircuit,
    a_qubits: Sequence[int],
    b_qubits: Sequence[int],
    c_qubits: Sequence[int],
    gt_qubit: int,
) -> QuantumCircuit:
    """Set ``gt_qubit = 1`` iff ``a > b``.

    Implementation: compute ``b - a`` via `q_sub` into the b/c registers; the
    final carry-out (``c_qubits[-1]``) is 1 iff ``b ≥ a``; we use a CNOT to
    write the *negation* into ``gt_qubit``, and an X to flip equality to
    strict-greater. The caller is responsible for uncomputing b / c if they
    need the original `b`.
    """
    _check_registers(a_qubits, b_qubits, c_qubits)
    q_sub(qc, a_qubits, b_qubits, c_qubits)
    # If a > b then b - a wraps to a "negative" number → final carry = 0.
    # If a ≤ b then final carry = 1.
    # So: gt_qubit = NOT(final_carry).
    qc.cx(c_qubits[-1], gt_qubit)
    qc.x(gt_qubit)
    return qc


def qc_add_1_inv(qc: QuantumCircuit, a: int, b: int, c_in: int, c_out: int) -> QuantumCircuit:
    """Inverse of :func:`qc_add_1`: gates in reverse order."""
    qc.cx(c_in, b)
    qc.ccx(b, c_in, c_out)
    qc.cx(a, b)
    qc.ccx(a, b, c_out)
    return qc


def q_add_inv(
    qc: QuantumCircuit,
    a_qubits: Sequence[int],
    b_qubits: Sequence[int],
    c_qubits: Sequence[int],
) -> QuantumCircuit:
    """Inverse of :func:`q_add`: ``b ← (b - a) mod 2^(n+1)``, restoring the
    carry chain in ``c_qubits`` to ``|0⟩`` (assuming it was ``|0⟩`` going
    into the matching :func:`q_add`)."""
    _check_registers(a_qubits, b_qubits, c_qubits)
    n = len(a_qubits)
    for i in range(n - 1, -1, -1):
        qc_add_1_inv(qc, a_qubits[i], b_qubits[i], c_qubits[i], c_qubits[i + 1])
    return qc


def q_sub_inv(
    qc: QuantumCircuit,
    a_qubits: Sequence[int],
    b_qubits: Sequence[int],
    c_qubits: Sequence[int],
) -> QuantumCircuit:
    """Inverse of :func:`q_sub`: ``b ← (b + a)`` plus carry-register cleanup.

    Reverses the gate sequence of `q_sub`: undo the +1 preload uncompute,
    undo the ~a bitflip, undo the addition, undo the +1 preload, undo ~a.
    """
    _check_registers(a_qubits, b_qubits, c_qubits)
    # Reverse the q_sub sequence:
    #   1. ~a flip          (preserved → undo last)
    #   2. X(c[0]) preload  (undone in step 5 of q_sub → re-preload now)
    #   3. q_add            (undo by q_add_inv)
    #   4. ~a flip restore  (undo first by re-flipping)
    qc.x(c_qubits[0])
    _bitwise_not(qc, a_qubits)
    q_add_inv(qc, a_qubits, b_qubits, c_qubits)
    _bitwise_not(qc, a_qubits)
    qc.x(c_qubits[0])
    return qc


def qc_add_1_ctrl(
    qc: QuantumCircuit,
    ctrl: int,
    a: int,
    b: int,
    c_in: int,
    c_out: int,
) -> QuantumCircuit:
    """Controlled single-bit full adder: applies :func:`qc_add_1` on
    ``(a, b, c_in, c_out)`` if ``ctrl == 1``, identity otherwise.

    Each gate in ``qc_add_1`` acquires one extra control on ``ctrl``:

      - ``ccx(a, b, c_out)``     → ``mcx([ctrl, a, b], c_out)``
      - ``cx(a, b)``             → ``ccx(ctrl, a, b)``
      - ``ccx(b, c_in, c_out)``  → ``mcx([ctrl, b, c_in], c_out)``
      - ``cx(c_in, b)``          → ``ccx(ctrl, c_in, b)``
    """
    qc.mcx([ctrl, a, b], c_out)
    qc.ccx(ctrl, a, b)
    qc.mcx([ctrl, b, c_in], c_out)
    qc.ccx(ctrl, c_in, b)
    return qc


def q_add_ctrl(
    qc: QuantumCircuit,
    ctrl: int,
    a_qubits: Sequence[int],
    b_qubits: Sequence[int],
    c_qubits: Sequence[int],
) -> QuantumCircuit:
    """Controlled ripple-carry adder: ``b ← (a + b)`` if ``ctrl == 1``,
    identity otherwise. Same register-width invariants as :func:`q_add`.
    """
    _check_registers(a_qubits, b_qubits, c_qubits)
    n = len(a_qubits)
    for i in range(n):
        qc_add_1_ctrl(qc, ctrl, a_qubits[i], b_qubits[i], c_qubits[i], c_qubits[i + 1])
    return qc


def qc_add_1_ctrl_inv(
    qc: QuantumCircuit,
    ctrl: int,
    a: int,
    b: int,
    c_in: int,
    c_out: int,
) -> QuantumCircuit:
    """Inverse of :func:`qc_add_1_ctrl`: same gates in reverse order
    (every gate is self-inverse)."""
    qc.ccx(ctrl, c_in, b)
    qc.mcx([ctrl, b, c_in], c_out)
    qc.ccx(ctrl, a, b)
    qc.mcx([ctrl, a, b], c_out)
    return qc


def q_add_ctrl_inv(
    qc: QuantumCircuit,
    ctrl: int,
    a_qubits: Sequence[int],
    b_qubits: Sequence[int],
    c_qubits: Sequence[int],
) -> QuantumCircuit:
    """Inverse of :func:`q_add_ctrl`: iterates in reverse order and
    applies :func:`qc_add_1_ctrl_inv` per bit.

    Restores ``b`` to its pre-call value when ``ctrl == 1`` (was
    ``b + a``, now back to ``b``) and the carry register to its
    pre-call state (assumed ``|0⟩`` originally).
    """
    _check_registers(a_qubits, b_qubits, c_qubits)
    n = len(a_qubits)
    for i in range(n - 1, -1, -1):
        qc_add_1_ctrl_inv(qc, ctrl, a_qubits[i], b_qubits[i], c_qubits[i], c_qubits[i + 1])
    return qc


def q_mul_const(
    qc: QuantumCircuit,
    b_qubits: Sequence[int],
    k_bits: Sequence[int],
    accum_qubits: Sequence[int],
    c_qubits: Sequence[int],
    guard_qubit: int,
) -> QuantumCircuit:
    """Multiply ``b`` by a classical constant via shift-add into the
    accumulator: ``accum ← accum + b · k`` (mod ``2**(q + m)``).

    Parameters
    ----------
    b_qubits
        ``q``-bit input register (little-endian); preserved on exit.
    k_bits
        Classical bit pattern of ``k`` as a sequence of ``{0, 1}`` values,
        little-endian (``k = Σ k_bits[i] · 2**i``). ``len(k_bits) = m``.
    accum_qubits
        Accumulator register of width ``q + m`` bits; not assumed to be
        initially zero — the operation is ``accum += b · k`` (modular at
        the accumulator width).
    c_qubits
        Carry-scratch register. Width must be at least
        ``popcount(k) * (q + 2)`` — each shift consumes a fresh
        ``(q + 2)``-bit slice (one more than a plain ``q``-bit add to
        accommodate the zero-extended high bit of ``b``). The slices stay
        dirty (entangled with the result) and must be uncomputed by
        running :func:`q_mul_const_inv` at the pipeline tail.
    guard_qubit
        A single ancilla qubit that must be ``|0⟩`` on entry. It is used
        as the (zero-valued) high bit of ``b`` for each shifted add so
        the carry-out propagates one position up into the accumulator
        rather than being trapped in the carry register. It is restored
        to ``|0⟩`` on exit (the all-XOR structure of ``q_add`` flips it
        back to its input value).

        Because the guard qubit returns to ``|0⟩`` after each call, the
        SAME guard qubit may be reused across multiple :func:`q_mul_const`
        calls in the same circuit — for instance in a Class-A GP pipeline
        where you multiply ``I_b`` by ``G`` and then later multiply by
        ``1/(R_ox - R_red)`` for a Class-C reparametrisation. Allocating
        one guard for the whole pipeline is the recommended pattern.

    Notes
    -----
    Cost: ``popcount(k)`` calls to ``q_add`` of width ``q + 1``. The
    carry qubits and accumulator are entangled with each other; only
    :func:`q_mul_const_inv` cleans the whole construction.
    """
    q = len(b_qubits)
    m = len(k_bits)
    if len(accum_qubits) < q + m:
        raise ValueError(f"accum_qubits must have ≥ q + m = {q + m} bits, got {len(accum_qubits)}")
    popcount = sum(1 for bit in k_bits if bit)
    needed_c = popcount * (q + 2)
    if len(c_qubits) < needed_c:
        raise ValueError(
            f"c_qubits must have ≥ popcount(k) * (q + 2) = {needed_c} bits, got {len(c_qubits)}"
        )
    b_extended = [*list(b_qubits), guard_qubit]
    c_off = 0
    for i, bit in enumerate(k_bits):
        if not bit:
            continue
        # accum[i .. i+q] ← accum[i .. i+q] + b_extended (width q+1).
        target = list(accum_qubits[i : i + q + 1])
        c_slice = list(c_qubits[c_off : c_off + q + 2])
        q_add(qc, b_extended, target, c_slice)
        c_off += q + 2
    return qc


def q_mul_const_inv(
    qc: QuantumCircuit,
    b_qubits: Sequence[int],
    k_bits: Sequence[int],
    accum_qubits: Sequence[int],
    c_qubits: Sequence[int],
    guard_qubit: int,
) -> QuantumCircuit:
    """Exact inverse of :func:`q_mul_const`."""
    q = len(b_qubits)
    popcount = sum(1 for bit in k_bits if bit)
    b_extended = [*list(b_qubits), guard_qubit]
    c_off = popcount * (q + 2)
    for i in range(len(k_bits) - 1, -1, -1):
        if not k_bits[i]:
            continue
        c_off -= q + 2
        target = list(accum_qubits[i : i + q + 1])
        c_slice = list(c_qubits[c_off : c_off + q + 2])
        q_add_inv(qc, b_extended, target, c_slice)
    return qc


def q_sub_ctrl(
    qc: QuantumCircuit,
    ctrl: int,
    a_qubits: Sequence[int],
    b_qubits: Sequence[int],
    c_qubits: Sequence[int],
) -> QuantumCircuit:
    """Controlled subtractor: ``b ← (b - a)`` if ``ctrl == 1``, identity
    otherwise.

    Implementation: same two's-complement trick as :func:`q_sub`, but
    every gate that mutates state is conditioned on ``ctrl``. The
    bit-flip steps on ``a`` are made conditional via CNOTs so the
    inversion is undone whether or not the body fires.
    """
    _check_registers(a_qubits, b_qubits, c_qubits)
    # ~a (conditional): if ctrl=1, flip a.
    for q in a_qubits:
        qc.cx(ctrl, q)
    # +1 preload (conditional)
    qc.cx(ctrl, c_qubits[0])
    q_add_ctrl(qc, ctrl, a_qubits, b_qubits, c_qubits)
    for q in a_qubits:
        qc.cx(ctrl, q)
    qc.cx(ctrl, c_qubits[0])
    return qc


def q_sub_ctrl_inv(
    qc: QuantumCircuit,
    ctrl: int,
    a_qubits: Sequence[int],
    b_qubits: Sequence[int],
    c_qubits: Sequence[int],
) -> QuantumCircuit:
    """Inverse of :func:`q_sub_ctrl`: reverses the gate order. All
    constituent CNOTs are self-inverse; the embedded :func:`q_add_ctrl`
    is undone by :func:`q_add_ctrl_inv`.

    When ``ctrl == 1`` on entry, restores ``b`` from ``b − a`` to ``b``,
    and the carry register to ``|0⟩`` (assumed ``|0⟩`` originally).
    Identity when ``ctrl == 0``.
    """
    _check_registers(a_qubits, b_qubits, c_qubits)
    qc.cx(ctrl, c_qubits[0])
    for q in a_qubits:
        qc.cx(ctrl, q)
    q_add_ctrl_inv(qc, ctrl, a_qubits, b_qubits, c_qubits)
    qc.cx(ctrl, c_qubits[0])
    for q in a_qubits:
        qc.cx(ctrl, q)
    return qc


def q_add_sub_ctrl(
    qc: QuantumCircuit,
    ctrl: int,
    a_qubits: Sequence[int],
    b_qubits: Sequence[int],
    c_qubits: Sequence[int],
) -> QuantumCircuit:
    """Controlled add-or-subtract: ``b ← b + a`` if ``ctrl == 0``, ``b ← b − a``
    if ``ctrl == 1``.

    Implementation: a single :func:`q_add` body, sandwiched between conditional
    two's-complement of ``a`` (CNOTs from ``ctrl`` to each bit of ``a`` plus a
    CNOT into ``c_qubits[0]`` for the +1 preload). When ``ctrl = 0`` both
    sandwich layers are identity and the body computes ``b + a``; when
    ``ctrl = 1`` the body computes ``b + (~a + 1) = b − a``.

    This primitive is the workhorse of :func:`q_div_nonrestoring`: each
    iteration of non-restoring division applies one ``q_add_sub_ctrl`` instead
    of restoring's trial-subtract / undo / conditional-subtract trio.
    """
    _check_registers(a_qubits, b_qubits, c_qubits)
    for q in a_qubits:
        qc.cx(ctrl, q)
    qc.cx(ctrl, c_qubits[0])
    q_add(qc, a_qubits, b_qubits, c_qubits)
    for q in a_qubits:
        qc.cx(ctrl, q)
    qc.cx(ctrl, c_qubits[0])
    return qc


def q_add_sub_ctrl_inv(
    qc: QuantumCircuit,
    ctrl: int,
    a_qubits: Sequence[int],
    b_qubits: Sequence[int],
    c_qubits: Sequence[int],
) -> QuantumCircuit:
    """Inverse of :func:`q_add_sub_ctrl`: gates in reverse, body undone by
    :func:`q_add_inv`. Restores ``b`` to its pre-call value and the carry
    chain to ``|0⟩`` (assumed ``|0⟩`` originally)."""
    _check_registers(a_qubits, b_qubits, c_qubits)
    qc.cx(ctrl, c_qubits[0])
    for q in a_qubits:
        qc.cx(ctrl, q)
    q_add_inv(qc, a_qubits, b_qubits, c_qubits)
    qc.cx(ctrl, c_qubits[0])
    for q in a_qubits:
        qc.cx(ctrl, q)
    return qc


def q_div_restoring(
    qc: QuantumCircuit,
    dividend_qubits: Sequence[int],
    divisor_qubits: Sequence[int],
    quotient_qubits: Sequence[int],
    work_qubits: Sequence[int],
    divisor_pad_qubit: int,
    c_qubits: Sequence[int],
    div_zero_flag: int,
) -> QuantumCircuit:
    """Unsigned fixed-point restoring division.

    Computes ``quotient = dividend // divisor`` and overwrites the
    dividend register with the remainder ``dividend mod divisor``.

    Layout (all little-endian, width ``q`` unless stated):
      - ``dividend_qubits``  (q)  → on exit: remainder.
      - ``divisor_qubits``   (q)  → preserved.
      - ``quotient_qubits``  (q)  → |0⟩ on entry, quotient on exit.
      - ``work_qubits``      (q)  → |0⟩ on entry, restored to |0⟩ on exit
                                    (high half of the conceptual 2q-bit
                                    running register R = work || dividend).
      - ``divisor_pad_qubit`` (1) → ancilla, |0⟩ on entry and exit;
                                    zero-pad MSB of the divisor.
      - ``c_qubits``              → carry-scratch, width
                                    ``≥ (q + 1) * (q + 2)``. Layout:
                                    first ``(q + 2)`` qubits are a shared
                                    *clean* slice reused by every
                                    iteration's GTE comparator; the
                                    remaining ``q * (q + 2)`` are *dirty*
                                    per-iteration slices for the
                                    conditional subtract (cleaned only
                                    by :func:`q_div_restoring_inv`).
      - ``div_zero_flag``    (1)  → set to 1 iff divisor == 0 on entry.

    Algorithm (slim-carry variant):
      For i in q-1 down to 0:
        window = R[i .. i+q]                                       (q+1 bits)
        # GTE comparator (clean): trial-subtract then undo, copying the
        # no-borrow sign bit into quotient[i].
        q_sub(D, window, c_cmp)
        CNOT(c_cmp[-1], quotient[i])
        q_sub_inv(D, window, c_cmp)         # restores window and c_cmp.
        # Actual conditional subtract: only fires when quotient[i] = 1.
        q_sub_ctrl(quotient[i], D, window, c_act_i)

    Cost: ``q`` iterations × (one full q_sub-pair + one q_sub_ctrl) ≈
    O(q²) Toffolis with constant factor ~10. Ancilla: ``(q + 1)(q + 2)``
    carry + 1 divisor-pad.

    Composition note: this primitive is suitable for
    *measurement-and-discard* use (Class B / Class A / Class C autonomous
    circuits where the result is read out and the carry register is
    thrown away). It is **not** yet composable inside a QAE-style oracle
    or a multi-stage pipeline that reuses the carry register, because
    the per-iteration ``c_act`` slices are left entangled with the
    quotient and remainder. A clean ``q_div_restoring_inv`` (and the
    prerequisite ``q_add_ctrl_inv`` / ``q_sub_ctrl_inv`` controlled-op
    inverses) would unblock QAE — see TODO in
    ``tests/unit/test_arithmetic.py`` (``test_q_div_restoring_inv_round_trip``)
    and the Stage E note in the project plan.
    """
    q = len(dividend_qubits)
    if len(divisor_qubits) != q:
        raise ValueError(f"divisor must have q={q} bits, got {len(divisor_qubits)}")
    if len(quotient_qubits) != q:
        raise ValueError(f"quotient must have q={q} bits, got {len(quotient_qubits)}")
    if len(work_qubits) != q:
        raise ValueError(f"work_qubits must have q={q} bits, got {len(work_qubits)}")
    needed_c = (q + 1) * (q + 2)
    if len(c_qubits) < needed_c:
        raise ValueError(f"c_qubits must have ≥ (q+1)(q+2) = {needed_c} bits, got {len(c_qubits)}")

    # Detect div-by-zero: div_zero_flag = AND_i ¬divisor[i].
    for d in divisor_qubits:
        qc.x(d)
    qc.mcx(list(divisor_qubits), div_zero_flag)
    for d in divisor_qubits:
        qc.x(d)

    # R = dividend (low) || work (high), 2q bits.
    R = list(dividend_qubits) + list(work_qubits)
    # D = divisor || zero-pad, (q+1) bits with MSB = 0.
    D = [*list(divisor_qubits), divisor_pad_qubit]

    # Shared clean carry slice for the GTE comparator (reused).
    c_cmp = list(c_qubits[: q + 2])
    # Dirty carry block: q slices of (q+2) for the actual conditional sub.
    c_act_base = q + 2

    for i in range(q - 1, -1, -1):
        window = R[i : i + q + 1]
        # GTE comparator: trial-subtract, copy sign bit, undo.
        q_sub(qc, D, window, c_cmp)
        qc.cx(c_cmp[-1], quotient_qubits[i])
        q_sub_inv(qc, D, window, c_cmp)
        # Conditional actual subtract: only fires when quotient[i] = 1.
        # Indexing: iteration i (from q-1 down to 0) uses slice
        # (q-1-i) of the dirty carry block so the slice indices are
        # 0..q-1 in iteration order.
        slot = (q - 1) - i
        c_act = list(c_qubits[c_act_base + slot * (q + 2) : c_act_base + (slot + 1) * (q + 2)])
        q_sub_ctrl(qc, quotient_qubits[i], D, window, c_act)
    return qc


# Convenience alias: ``q_div`` points to the current default divider
# implementation (``q_div_restoring``). When future variants are added
# (``q_div_nonrestoring``, ``q_div_newton``, …), this alias may be
# repointed; downstream callers that want a specific algorithm should
# import it by its full name instead.
q_div = q_div_restoring


def q_div_general(
    qc: QuantumCircuit,
    dividend_qubits: Sequence[int],
    divisor_qubits: Sequence[int],
    quotient_qubits: Sequence[int],
    work_qubits: Sequence[int],
    divisor_pad_qubit: int,
    c_qubits: Sequence[int],
    div_zero_flag: int,
) -> QuantumCircuit:
    """Generalised restoring divider supporting non-square widths.

    Computes ``quotient = dividend // divisor`` for an ``n``-bit
    unsigned dividend and ``m``-bit unsigned divisor (with ``m ≤ n``),
    producing an ``n``-bit unsigned quotient. The dividend register is
    overwritten with the remainder on exit.

    Layout (all little-endian):
      - ``dividend_qubits`` (n bits)   → on exit: low ``m`` bits hold
        the remainder; high ``n - m`` bits return to ``|0⟩``.
      - ``divisor_qubits``  (m bits)   → preserved.
      - ``quotient_qubits`` (n bits)   → ``|0⟩`` on entry; quotient on exit.
      - ``work_qubits``     (m bits)   → ``|0⟩`` on entry and exit
                                          (high zero-pad of R).
      - ``divisor_pad_qubit`` (1 bit)  → ancilla, ``|0⟩`` on entry and exit.
      - ``c_qubits``                    → carry-scratch, width
                                          ``≥ (n + 1) (m + 2)``: first
                                          ``(m + 2)`` qubits are the shared
                                          clean comparator slice; remaining
                                          ``n × (m + 2)`` are dirty
                                          per-iteration slices for the
                                          conditional subtracts.
      - ``div_zero_flag`` (1 bit)       → set to 1 iff divisor == 0.

    For the square case ``n = m = q`` this reduces exactly to
    :func:`q_div_restoring`; the square version is kept as a convenience
    name and points to its own dedicated implementation rather than
    delegating, to avoid spurious widening of the ``q_div`` alias.

    Cost: ``n`` iterations × (q_sub + q_sub_inv + q_sub_ctrl) each of
    width ``m + 1`` ⇒ O(n m) Toffolis. Ancilla: ``(n + 1)(m + 2)`` carry
    + ``m`` work + 1 divisor-pad + 1 div-zero-flag.

    Use this for Class A (|num| / den at GP fractional precision) and
    Class C (post-affine /(R_ox − R_red)) where the dividend has been
    pre-scaled (bit-shifted left by ``q_frac``) to give fractional output.
    """
    n = len(dividend_qubits)
    m = len(divisor_qubits)
    if m > n:
        raise ValueError(f"divisor width m={m} must not exceed dividend width n={n}")
    if len(quotient_qubits) != n:
        raise ValueError(f"quotient width must equal n={n}, got {len(quotient_qubits)}")
    if len(work_qubits) != m:
        raise ValueError(f"work width must equal m={m}, got {len(work_qubits)}")
    needed_c = (n + 1) * (m + 2)
    if len(c_qubits) < needed_c:
        raise ValueError(f"c_qubits must have ≥ (n+1)(m+2) = {needed_c} bits, got {len(c_qubits)}")

    # Div-by-zero detect (self-inverse).
    for d in divisor_qubits:
        qc.x(d)
    qc.mcx(list(divisor_qubits), div_zero_flag)
    for d in divisor_qubits:
        qc.x(d)

    # R = dividend (low n) || work (high m). Width n + m.
    R = list(dividend_qubits) + list(work_qubits)
    # D = divisor || pad. Width m + 1.
    D = [*list(divisor_qubits), divisor_pad_qubit]

    c_cmp = list(c_qubits[: m + 2])
    c_act_base = m + 2
    for i in range(n - 1, -1, -1):
        window = R[i : i + m + 1]
        slot = (n - 1) - i
        c_act = list(c_qubits[c_act_base + slot * (m + 2) : c_act_base + (slot + 1) * (m + 2)])
        q_sub(qc, D, window, c_cmp)
        qc.cx(c_cmp[-1], quotient_qubits[i])
        q_sub_inv(qc, D, window, c_cmp)
        q_sub_ctrl(qc, quotient_qubits[i], D, window, c_act)
    return qc


def q_div_restoring_inv(
    qc: QuantumCircuit,
    dividend_qubits: Sequence[int],
    divisor_qubits: Sequence[int],
    quotient_qubits: Sequence[int],
    work_qubits: Sequence[int],
    divisor_pad_qubit: int,
    c_qubits: Sequence[int],
    div_zero_flag: int,
) -> QuantumCircuit:
    """Exact inverse of :func:`q_div_restoring`.

    Restores all registers to their state immediately before the
    matching :func:`q_div_restoring` call:
      - ``dividend_qubits`` returns from holding the remainder back to
        the original dividend value.
      - ``quotient_qubits`` returns to ``|0⟩``.
      - ``work_qubits`` was already ``|0⟩`` after the forward call;
        it stays ``|0⟩``.
      - ``c_qubits`` all returns to ``|0⟩`` (both the comparator slice,
        which was clean already, and the q dirty per-iteration slices).
      - ``divisor_pad_qubit`` returns to ``|0⟩``.
      - ``div_zero_flag`` returns to its pre-call value (``|0⟩`` if the
        forward call set it; the div-by-zero detection is self-inverse).
      - ``divisor_qubits`` unchanged throughout.

    This is the prerequisite for QAE-style oracle composition: with the
    forward+inverse pair, the divider acts as a clean unitary on
    (dividend, divisor, quotient) with all ancilla returned to ``|0⟩``,
    making it composable inside controlled-power-of-Q reflections.
    """
    q = len(dividend_qubits)
    if len(divisor_qubits) != q:
        raise ValueError(f"divisor must have q={q} bits, got {len(divisor_qubits)}")
    if len(quotient_qubits) != q:
        raise ValueError(f"quotient must have q={q} bits, got {len(quotient_qubits)}")
    if len(work_qubits) != q:
        raise ValueError(f"work_qubits must have q={q} bits, got {len(work_qubits)}")
    needed_c = (q + 1) * (q + 2)
    if len(c_qubits) < needed_c:
        raise ValueError(f"c_qubits must have ≥ (q+1)(q+2) = {needed_c} bits, got {len(c_qubits)}")

    R = list(dividend_qubits) + list(work_qubits)
    D = [*list(divisor_qubits), divisor_pad_qubit]
    c_cmp = list(c_qubits[: q + 2])
    c_act_base = q + 2

    # Reverse iteration order (forward goes q-1 → 0; inverse goes 0 → q-1).
    for i in range(0, q):
        window = R[i : i + q + 1]
        slot = (q - 1) - i  # matches the forward's c_act slicing
        c_act = list(c_qubits[c_act_base + slot * (q + 2) : c_act_base + (slot + 1) * (q + 2)])
        # Reverse the per-iteration sequence: undo q_sub_ctrl first, then
        # the GTE-comparator triple (q_sub_inv ; CNOT ; q_sub).
        q_sub_ctrl_inv(qc, quotient_qubits[i], D, window, c_act)
        # Forward did: q_sub → CNOT(c_cmp[-1], quotient[i]) → q_sub_inv.
        # Inverse: q_sub (undo of q_sub_inv) → CNOT (self-inverse) → q_sub_inv (undo of q_sub).
        q_sub(qc, D, window, c_cmp)
        qc.cx(c_cmp[-1], quotient_qubits[i])
        q_sub_inv(qc, D, window, c_cmp)

    # Reverse the div-by-zero detection (the X-mcx-X sequence is its own
    # inverse, so we apply the same gates).
    for d in divisor_qubits:
        qc.x(d)
    qc.mcx(list(divisor_qubits), div_zero_flag)
    for d in divisor_qubits:
        qc.x(d)
    return qc


def q_div_nonrestoring(
    qc: QuantumCircuit,
    dividend_qubits: Sequence[int],
    divisor_qubits: Sequence[int],
    quotient_qubits: Sequence[int],
    work_qubits: Sequence[int],
    divisor_pad_qubit: int,
    c_qubits: Sequence[int],
    div_zero_flag: int,
) -> QuantumCircuit:
    """Unsigned fixed-point non-restoring division (Thapliyal-style).

    Drop-in replacement for :func:`q_div_restoring` with identical
    signature, identical semantics on the output registers, and ~2× fewer
    two-qubit gates per iteration after transpilation. Computes
    ``quotient = dividend // divisor`` and overwrites the dividend
    register with the remainder.

    Algorithm:
      - First iteration (i = q − 1): unconditional subtract of the divisor
        from the running window, then copy ``NOT sign`` to ``quotient[q−1]``.
      - Subsequent iterations (i = q − 2 .. 0): one *controlled
        add-or-subtract* of the divisor — subtract iff the previous
        quotient bit (i.e. previous ``NOT sign``) was 1, otherwise add.
        Then copy ``NOT sign`` to ``quotient[i]``.
      - Final correction: if the final sign is negative (i.e.
        ``quotient[0] == 0``), add the divisor back so the remainder is
        non-negative.

    The savings vs restoring come from doing **one** controlled
    add-or-subtract per iteration instead of restoring's "trial-subtract
    → CNOT sign → undo → conditional subtract" four-step sequence; the
    transpiled CX cost per iteration drops from ~58(q+1) to ~16(q+1).

    Layout (identical to :func:`q_div_restoring`):
      - ``dividend_qubits``  (q)  → on exit: remainder.
      - ``divisor_qubits``   (q)  → preserved.
      - ``quotient_qubits``  (q)  → |0⟩ on entry, quotient on exit.
      - ``work_qubits``      (q)  → |0⟩ on entry, restored to |0⟩ on exit
                                    (high half of the conceptual 2q-bit
                                    running register R = work || dividend).
      - ``divisor_pad_qubit`` (1) → ancilla, |0⟩ on entry and exit;
                                    zero-pad MSB of the divisor.
      - ``c_qubits``              → carry-scratch, width
                                    ``≥ (q + 1) * (q + 2)``. Layout: one
                                    fresh ``(q + 2)``-slice per iteration
                                    (q in total) plus a final
                                    ``(q + 2)``-slice for the correction
                                    add.
      - ``div_zero_flag``    (1)  → set to 1 iff divisor == 0 on entry.
    """
    q = len(dividend_qubits)
    if len(divisor_qubits) != q:
        raise ValueError(f"divisor must have q={q} bits, got {len(divisor_qubits)}")
    if len(quotient_qubits) != q:
        raise ValueError(f"quotient must have q={q} bits, got {len(quotient_qubits)}")
    if len(work_qubits) != q:
        raise ValueError(f"work_qubits must have q={q} bits, got {len(work_qubits)}")
    needed_c = (q + 1) * (q + 2)
    if len(c_qubits) < needed_c:
        raise ValueError(f"c_qubits must have ≥ (q+1)(q+2) = {needed_c} bits, got {len(c_qubits)}")

    # Detect div-by-zero: same as restoring.
    for d in divisor_qubits:
        qc.x(d)
    qc.mcx(list(divisor_qubits), div_zero_flag)
    for d in divisor_qubits:
        qc.x(d)

    R = list(dividend_qubits) + list(work_qubits)
    D = [*list(divisor_qubits), divisor_pad_qubit]

    def _carry_slot(slot_idx: int) -> list[int]:
        base = slot_idx * (q + 2)
        return list(c_qubits[base : base + q + 2])

    def _copy_not_sign(window: list[int], dest: int) -> None:
        # dest ← NOT window[-1], leaving window[-1] unchanged.
        qc.x(window[-1])
        qc.cx(window[-1], dest)
        qc.x(window[-1])

    # First iteration: unconditional subtract.
    window_first = R[q - 1 : 2 * q]
    q_sub(qc, D, window_first, _carry_slot(0))
    _copy_not_sign(window_first, quotient_qubits[q - 1])

    # Iterations i = q-2 .. 0: controlled add-or-subtract.
    # Control = quotient[i+1]; ctrl=1 → subtract, ctrl=0 → add.
    for i in range(q - 2, -1, -1):
        slot = (q - 1) - i
        window_i = R[i : i + q + 1]
        q_add_sub_ctrl(qc, quotient_qubits[i + 1], D, window_i, _carry_slot(slot))
        _copy_not_sign(window_i, quotient_qubits[i])

    # Final correction: if final sign was negative (quotient[0] == 0),
    # add divisor back once so the dividend register holds a non-negative
    # remainder. Use the last (q+2)-slice of c_qubits.
    qc.x(quotient_qubits[0])
    window_final = R[0 : q + 1]
    q_add_ctrl(qc, quotient_qubits[0], D, window_final, _carry_slot(q))
    qc.x(quotient_qubits[0])

    # Uncompute the leftover sign bits in the work register.
    # The windowed non-restoring loop leaves R[i+q] = NOT quotient[i] for
    # every iteration i — these are sign bits the (conceptual) shift step
    # never overwrites. R[q] (iter 0's sign) was zeroed by the final
    # correction in both branches (no-correction case had it = 0 already;
    # correction case flips it 1 → 0). The remaining bits R[q+1..2q-1]
    # equal NOT quotient[i] for i = 1..q-1 and must be cleared so the
    # work register returns to |0⟩ for composition / inverse usage.
    for i in range(1, q):
        qc.cx(quotient_qubits[i], R[i + q])
        qc.x(R[i + q])
    return qc


def q_div_nonrestoring_inv(
    qc: QuantumCircuit,
    dividend_qubits: Sequence[int],
    divisor_qubits: Sequence[int],
    quotient_qubits: Sequence[int],
    work_qubits: Sequence[int],
    divisor_pad_qubit: int,
    c_qubits: Sequence[int],
    div_zero_flag: int,
) -> QuantumCircuit:
    """Exact inverse of :func:`q_div_nonrestoring`.

    Restores all registers to their pre-call state, including the carry
    register, work register, quotient (back to |0⟩), and dividend
    (remainder → original). Prerequisite for QAE-style oracle
    composition.
    """
    q = len(dividend_qubits)
    needed_c = (q + 1) * (q + 2)
    if len(c_qubits) < needed_c:
        raise ValueError(f"c_qubits must have ≥ (q+1)(q+2) = {needed_c} bits, got {len(c_qubits)}")

    R = list(dividend_qubits) + list(work_qubits)
    D = [*list(divisor_qubits), divisor_pad_qubit]

    def _carry_slot(slot_idx: int) -> list[int]:
        base = slot_idx * (q + 2)
        return list(c_qubits[base : base + q + 2])

    def _uncopy_not_sign(window: list[int], dest: int) -> None:
        # NOT-sign copy is self-inverse on the dest (assuming dest was
        # |0⟩ before forward); we re-apply the same gates.
        qc.x(window[-1])
        qc.cx(window[-1], dest)
        qc.x(window[-1])

    # Undo the work-bit uncompute step (reverse order).
    for i in range(q - 1, 0, -1):
        qc.x(R[i + q])
        qc.cx(quotient_qubits[i], R[i + q])

    # Undo final correction.
    qc.x(quotient_qubits[0])
    window_final = R[0 : q + 1]
    q_add_ctrl_inv(qc, quotient_qubits[0], D, window_final, _carry_slot(q))
    qc.x(quotient_qubits[0])

    # Undo iterations i = 0 .. q-2 in reverse forward order
    # (forward went q-2 down to 0; inverse goes 0 up to q-2).
    for i in range(0, q - 1):
        slot = (q - 1) - i
        window_i = R[i : i + q + 1]
        _uncopy_not_sign(window_i, quotient_qubits[i])
        q_add_sub_ctrl_inv(qc, quotient_qubits[i + 1], D, window_i, _carry_slot(slot))

    # Undo first iteration.
    window_first = R[q - 1 : 2 * q]
    _uncopy_not_sign(window_first, quotient_qubits[q - 1])
    q_sub_inv(qc, D, window_first, _carry_slot(0))

    # Undo div-by-zero detection (self-inverse).
    for d in divisor_qubits:
        qc.x(d)
    qc.mcx(list(divisor_qubits), div_zero_flag)
    for d in divisor_qubits:
        qc.x(d)
    return qc
