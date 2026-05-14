"""Tests for qimp.processing.gp_ratio."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from qimp.processing.gp_ratio import (
    apply_gp_function,
    classical_gp_image,
    combined_objective,
)


def test_classical_gp_image_zero_when_equal() -> None:
    a = np.ones((4, 4))
    result = classical_gp_image(a, a)
    np.testing.assert_allclose(result, np.zeros((4, 4)), atol=1e-9)


def test_classical_gp_image_endpoints() -> None:
    g = np.array([[2.0]])
    r = np.array([[0.0]])
    assert classical_gp_image(g, r)[0, 0] == pytest.approx(1.0)
    assert classical_gp_image(r, g)[0, 0] == pytest.approx(-1.0)


def test_classical_gp_image_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        classical_gp_image(np.zeros(2), np.zeros(3))


def test_combined_objective_signs() -> None:
    assert combined_objective(1.0, 0.0, 0.0) == 1.0
    assert combined_objective(0.0, 1.0, 0.0) == -1.0
    assert combined_objective(0.0, 0.0, 1.0) == 1.0


def test_combined_objective_weights() -> None:
    out = combined_objective(2.0, 5.0, 3.0, alpha=0.5, beta=2.0, gamma=0.1)
    assert out == pytest.approx(0.5 * 2 - 2.0 * 5 + 0.1 * 3)


@pytest.mark.parametrize("n,m", [(1, 1), (2, 1), (1, 2)])
def test_apply_gp_function_appends_operations(n: int, m: int) -> None:
    total = 2 * n + m + 1
    qc = QuantumCircuit(total, total)
    num_params = 2 * (1 << (2 * n))
    params = [Parameter(f"θ{i}") for i in range(num_params)]
    depth_before = qc.depth()
    apply_gp_function(qc, n=n, m=m, params=params)
    assert qc.depth() > depth_before
    # Sanity: number of CRY gates equals num_params.
    cry_count = sum(1 for instr in qc.data if instr.operation.name == "cry")
    assert cry_count == num_params


def test_apply_gp_function_validates_param_count() -> None:
    qc = QuantumCircuit(4, 4)
    with pytest.raises(ValueError, match="expected"):
        apply_gp_function(qc, n=1, m=1, params=[Parameter("a")])


def test_apply_gp_function_validates_qubit_count() -> None:
    qc = QuantumCircuit(2, 2)
    params = [Parameter(f"p{i}") for i in range(8)]
    with pytest.raises(ValueError, match="qubits"):
        apply_gp_function(qc, n=1, m=1, params=params)
