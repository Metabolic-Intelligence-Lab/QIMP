"""Tests for the truth-table-synthesised divider and its Class-B integration.

Parametrised over q (intensity width) and n (spatial qubits per axis), per the
scalability constraint of the library.
"""
from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qimp.processing.arithmetic import q_div_lookup, q_div_lookup_inv
from qimp.processing.ratiometric_circuit import class_b_ratio, class_b_ratio_inv


def _basis_out(qc: QuantumCircuit, x: int) -> int:
    sv = Statevector.from_int(x, 2**qc.num_qubits).evolve(qc)
    k = int(np.argmax(np.abs(sv.data)))
    assert abs(abs(sv.data[k]) - 1.0) < 1e-9
    return k


@pytest.mark.parametrize("q", [1, 2, 3])
def test_q_div_lookup_exhaustive(q: int) -> None:
    full = (1 << q) - 1
    qc = QuantumCircuit(3 * q + 1)
    A, D, Q, F = list(range(q)), list(range(q, 2 * q)), list(range(2 * q, 3 * q)), 3 * q
    q_div_lookup(qc, A, D, Q, F)
    for a in range(1 << q):
        for d in range(1 << q):
            x = a | (d << q)
            quot, flag = ((a // d), 0) if d else (full, 1)
            assert _basis_out(qc, x) == x | (quot << (2 * q)) | (flag << (3 * q)), (a, d)


@pytest.mark.parametrize("q", [1, 2, 3])
def test_q_div_lookup_self_inverse(q: int) -> None:
    qc = QuantumCircuit(3 * q + 1)
    A, D, Q, F = list(range(q)), list(range(q, 2 * q)), list(range(2 * q, 3 * q)), 3 * q
    q_div_lookup(qc, A, D, Q, F)
    q_div_lookup_inv(qc, A, D, Q, F)
    for x in range(1 << (2 * q)):
        assert _basis_out(qc, x) == x


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("q", [1, 2, 3])
@pytest.mark.parametrize("load", ["mcx", "ucry"])
def test_class_b_ratio_lookup_bit_exact(n: int, q: int, load: str) -> None:
    rng = np.random.default_rng(10 * n + q)
    side = 1 << n
    Ia = rng.integers(0, 1 << q, size=(side, side))
    Ib = rng.integers(0, 1 << q, size=(side, side))
    qc, lay = class_b_ratio(Ia, Ib, q=q, divider="lookup", load=load)
    assert qc.num_qubits == 2 * n + 3 * q + 1
    probs = Statevector(qc).probabilities_dict()
    full = (1 << q) - 1
    seen = set()
    for s, p in probs.items():
        if p < 1e-9:
            continue
        bit = lambda i: int(s[qc.num_qubits - 1 - i])  # noqa: E731
        col = sum(bit(lay["position"][i]) << i for i in range(n))
        row = sum(bit(lay["position"][n + i]) << i for i in range(n))
        a, d = int(Ia[row, col]), int(Ib[row, col])
        quot = sum(bit(lay["quotient"][i]) << i for i in range(q))
        assert sum(bit(lay["I_a"][i]) << i for i in range(q)) == a
        assert sum(bit(lay["I_b"][i]) << i for i in range(q)) == d
        assert quot == ((a // d) if d else full)
        assert bit(lay["flag"]) == (1 if d == 0 else 0)
        seen.add((row, col))
    assert len(seen) == side * side


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("q", [1, 2])
def test_class_b_ratio_lookup_round_trip(n: int, q: int) -> None:
    rng = np.random.default_rng(100 + 10 * n + q)
    side = 1 << n
    Ia = rng.integers(0, 1 << q, size=(side, side))
    Ib = rng.integers(0, 1 << q, size=(side, side))
    qc, lay = class_b_ratio(Ia, Ib, q=q, divider="lookup", load="ucry")
    class_b_ratio_inv(qc, Ia, Ib, q, lay, divider="lookup", load="ucry")
    sv = Statevector(qc)
    assert abs(abs(sv.data[0]) - 1.0) < 1e-8
