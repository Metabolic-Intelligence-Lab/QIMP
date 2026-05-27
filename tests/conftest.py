"""Shared pytest fixtures.

Per the design constraint of scalability, tests are parametrized over the
spatial qubit count `n` (1..4) and intensity qubit count `q` (1..3) wherever
applicable.
"""

from __future__ import annotations

import pytest

# Parametrize markers used across the suite.
N_QUBITS_RANGE = [1, 2, 3, 4]
Q_QUBITS_RANGE = [1, 2, 3]

# Narrower ranges for the heavier reversible-arithmetic tests
# (controlled adder, multiplier, divider). q < 3 collapses corner
# cases away; q > 4 is statevector-impractical with the full ancilla
# budget.
N_QUBITS_ARITH_RANGE = [1, 2]
Q_QUBITS_ARITH_RANGE = [3, 4]


@pytest.fixture(params=N_QUBITS_RANGE, ids=lambda n: f"n={n}")
def n_qubits(request: pytest.FixtureRequest) -> int:
    """Number of spatial qubits per axis. Image side = 2^n."""
    return request.param


@pytest.fixture(params=Q_QUBITS_RANGE, ids=lambda q: f"q={q}")
def q_qubits(request: pytest.FixtureRequest) -> int:
    """Intensity qubits (for NEQR / NCQI)."""
    return request.param


@pytest.fixture(params=N_QUBITS_ARITH_RANGE, ids=lambda n: f"n={n}")
def n_qubits_arith(request: pytest.FixtureRequest) -> int:
    """Position-qubit count for the arithmetic-circuit tests."""
    return request.param


@pytest.fixture(params=Q_QUBITS_ARITH_RANGE, ids=lambda q: f"q={q}")
def q_qubits_arith(request: pytest.FixtureRequest) -> int:
    """Intensity-qubit count for the arithmetic-circuit tests."""
    return request.param
