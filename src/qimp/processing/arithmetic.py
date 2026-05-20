"""Quantum arithmetic primitives for NEQR-encoded images.

Public ops:
- ``qc_add_1(qc, a, b, c_out)`` — one-bit full adder ``b ← (a + b)`` with
  carry-out qubit ``c_out``.
- ``q_add(qc, a_qubits, b_qubits, c_qubits)`` — ripple-carry adder
  ``b ← (a + b)`` for arbitrary-width little-endian registers, using one
  auxiliary qubit per bit (``len(c_qubits) == len(a_qubits) + 1``).
- ``q_sub(qc, a_qubits, b_qubits, c_qubits)`` — wrap of ``q_add`` with the
  two's-complement of ``a``: ``b ← b - a``.
- ``neqr_comparator(qc, a_qubits, b_qubits, c_qubits, gt_qubit)`` — sets
  ``gt_qubit = 1`` iff ``a > b``.

Conventions
-----------
All registers are **little-endian** lists of qubit indices (``[lsb, …, msb]``).
`q_add` follows the standard ripple-carry construction with one carry qubit per
input bit plus one final-out qubit (so ``len(c_qubits) = n + 1``).

Reference: docs/tesi.pdf §3.1.2 (NEQR processing).
"""

from __future__ import annotations

from collections.abc import Sequence

from qiskit import QuantumCircuit

__all__ = [
    "neqr_comparator",
    "q_add",
    "q_sub",
    "qc_add_1",
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
