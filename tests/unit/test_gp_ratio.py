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
    decode_gp_counts,
    evaluate_gp,
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


def test_classical_gp_image_alpha_weights_red() -> None:
    """alpha must scale the red contribution before the numerator/denominator."""
    g = np.array([[1.0]])
    r = np.array([[1.0]])
    # alpha = 0 → (G - 0) / (G + 0) = 1
    assert classical_gp_image(g, r, alpha=0.0)[0, 0] == pytest.approx(1.0)
    # alpha = 2 → (1 - 2) / (1 + 2) = -1/3
    assert classical_gp_image(g, r, alpha=2.0)[0, 0] == pytest.approx(-1 / 3)


def test_classical_gp_image_rejects_negative_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        classical_gp_image(np.zeros((2, 2)), np.zeros((2, 2)), alpha=-1.0)


def test_calculate_gp_image_matches_classical() -> None:
    """The legacy ``calculate_gp_image`` wrapper must match the canonical formula
    when the output format is 'normalized' and alpha == G."""
    from qimp.io.image import calculate_gp_image

    rng = np.random.default_rng(0)
    g = rng.uniform(0, 1, size=(4, 4))
    r = rng.uniform(0, 1, size=(4, 4))
    for a in (0.0, 0.5, 1.5):
        out = calculate_gp_image(g, r, G=a, output_format="normalized")
        ref = np.clip(classical_gp_image(g, r, alpha=a), -1.0, 1.0)
        np.testing.assert_allclose(out, ref, atol=1e-9)


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
    # Sanity: each parameter binds a multi-controlled RY (annotated). Both
    # the per-pixel "diff" and "norm" rotations should appear, so the total
    # number of parameterised RY-family gates equals num_params.
    bound_rotations = sum(
        1
        for instr in qc.data
        if instr.operation.name in ("ry", "cry", "annotated") and instr.operation.params
    )
    assert bound_rotations == num_params


def test_apply_gp_function_validates_param_count() -> None:
    qc = QuantumCircuit(4, 4)
    with pytest.raises(ValueError, match="expected"):
        apply_gp_function(qc, n=1, m=1, params=[Parameter("a")])


def test_apply_gp_function_validates_qubit_count() -> None:
    qc = QuantumCircuit(2, 2)
    params = [Parameter(f"p{i}") for i in range(8)]
    with pytest.raises(ValueError, match="qubits"):
        apply_gp_function(qc, n=1, m=1, params=params)


def test_decode_gp_counts_returns_image_in_range() -> None:
    """For a balanced count histogram, decoded GP must be in [-1, 1] per pixel."""
    # Two pixels (n=1) with color qubit deterministic: pixel 0 → color=0, pixel 1 → color=1.
    # State key layout (high-to-low): color | selection | pos[1] | pos[0]
    counts = {
        # color=0, sel=0, pos=00 → GP[pos=0] should tend to -1
        "0 000": 1000,
        # color=1, sel=0, pos=01 → GP[pos=1] should tend to +1
        "1 001": 1000,
        # color=0, sel=0, pos=10
        "0 010": 1000,
        # color=1, sel=0, pos=11
        "1 011": 1000,
    }
    img = decode_gp_counts(counts, n=1, m=1)
    assert img.shape == (2, 2)
    assert img.min() >= -1.0 and img.max() <= 1.0


def test_evaluate_gp_runs_end_to_end() -> None:
    """The full encode → apply → simulate → decode loop produces a sane image."""
    # Tiny 2x2 case (n=1) so the statevector is cheap.
    green = np.array([[10.0, 5.0], [5.0, 10.0]])
    red = np.array([[5.0, 10.0], [10.0, 5.0]])
    num_params = 2 * (1 << 2)  # 8 params for n=1
    params = np.zeros(num_params)  # All-zero params → only encoding, no GP rotations
    out = evaluate_gp(green, red, params, exact=True)
    assert out.shape == (2, 2)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_evaluate_gp_rejects_non_square() -> None:
    with pytest.raises(ValueError, match="square"):
        evaluate_gp(np.zeros((2, 4)), np.zeros((2, 4)), np.zeros(8))


def test_evaluate_gp_rejects_non_pow2() -> None:
    with pytest.raises(ValueError, match="power of two"):
        evaluate_gp(np.zeros((3, 3)), np.zeros((3, 3)), np.zeros(8))


def test_evaluate_gp_rejects_all_zero_channels() -> None:
    with pytest.raises(ValueError, match="zero"):
        evaluate_gp(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros(8))


@pytest.mark.slow
def test_optimize_gp_reduces_loss() -> None:
    """A short optimisation must produce a lower loss than the random initial guess."""
    from qimp.processing.gp_ratio import optimize_gp

    rng = np.random.default_rng(0)
    green = rng.uniform(20, 50, size=(2, 2))
    red = rng.uniform(20, 50, size=(2, 2))
    result = optimize_gp(green, red, alpha=0.5, max_iter=15, seed=0, exact=True)
    assert len(result.history_combined) >= 2
    # Final combined-objective should be at or below the initial value.
    assert result.history_combined[-1] <= result.history_combined[0] + 1e-9


@pytest.mark.slow
def test_optimize_gp_reaches_classical_target_n1() -> None:
    """With the per-pixel ansatz fix, n=1 must converge to the exact classical GP.

    COBYLA on the 8-parameter circuit reliably reaches MSE ~ 1e-12 within
    a few hundred iterations. We assert a generous ceiling (1e-4) so the
    test stays robust across scipy versions / platforms.
    """
    from qimp.processing.gp_ratio import evaluate_gp, optimize_gp

    green = np.array([[80.0, 30.0], [60.0, 20.0]])
    red = np.array([[20.0, 80.0], [40.0, 60.0]])
    target = classical_gp_image(green, red, alpha=0.5)
    result = optimize_gp(green, red, alpha=0.5, max_iter=300, seed=0, exact=True)
    decoded = evaluate_gp(green, red, result.optimized_params, exact=True)
    mse_final = float(((target - decoded) ** 2).mean())
    assert mse_final < 1e-4, f"optimised MSE {mse_final:.6f} above tolerance"


def test_apply_gp_function_parameters_are_non_degenerate() -> None:
    """Setting different individual parameters to π must produce different decoded
    images. Regression test for the bug where all 2·2^(2n) parameters collapsed
    onto a single effective rotation because the CRY controlled only on the
    selection qubit and both halves of the per-pixel pair sat on the same
    selection branch.
    """
    from qimp.processing.gp_ratio import evaluate_gp

    green = np.array([[80.0, 30.0], [60.0, 20.0]])
    red = np.array([[20.0, 80.0], [40.0, 60.0]])
    outputs = []
    for k in range(8):
        p = np.zeros(8)
        p[k] = np.pi
        outputs.append(evaluate_gp(green, red, p, exact=True))
    # Each output should be distinct from every other (within numerical noise).
    for i in range(8):
        for j in range(i + 1, 8):
            diff = float(np.abs(outputs[i] - outputs[j]).max())
            assert diff > 1e-6, f"params {i} and {j} produce identical output"
