"""Tests for qimp.metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qimp import metrics


def test_mse_identical_is_zero() -> None:
    img = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert metrics.mse(img, img) == 0.0


def test_mse_known_value() -> None:
    a = np.array([[0.0, 0.0]])
    b = np.array([[1.0, 3.0]])
    assert metrics.mse(a, b) == pytest.approx((1 + 9) / 2)


def test_mse_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        metrics.mse(np.zeros(2), np.zeros(3))


def test_psnr_identical_is_inf() -> None:
    img = np.array([[1.0]])
    assert math.isinf(metrics.psnr(img, img))


def test_psnr_uint8_defaults_to_255() -> None:
    a = np.zeros((4, 4), dtype=np.uint8)
    b = np.ones((4, 4), dtype=np.uint8)
    # MSE = 1, MAX = 255 -> 20 log10(255) ≈ 48.13
    assert metrics.psnr(a, b) == pytest.approx(20 * math.log10(255), rel=1e-3)


def test_psnr_uint16_defaults_to_65535() -> None:
    a = np.zeros((2, 2), dtype=np.uint16)
    b = np.full((2, 2), 1, dtype=np.uint16)
    assert metrics.psnr(a, b) == pytest.approx(20 * math.log10(65535), rel=1e-3)


def test_psnr_float_defaults_to_one() -> None:
    a = np.zeros((2, 2))
    b = np.full((2, 2), 0.5)
    assert metrics.psnr(a, b) == pytest.approx(20 * math.log10(1.0 / 0.5))


def test_psnr_explicit_max_overrides_default() -> None:
    a = np.zeros((2, 2), dtype=np.uint8)
    b = np.ones((2, 2), dtype=np.uint8)
    assert metrics.psnr(a, b, max_intensity=1.0) == pytest.approx(0.0)


def test_total_variation_constant_image_is_zero() -> None:
    img = np.full((4, 4), 0.5)
    assert metrics.total_variation(img) == pytest.approx(0.0)


def test_total_variation_positive_on_edges() -> None:
    img = np.zeros((4, 4))
    img[:, 2:] = 1.0
    assert metrics.total_variation(img) > 0


def test_transpile_summary_shape() -> None:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    summary = metrics.transpile_summary(qc)
    assert set(summary.keys()) == {"depth", "ops", "depth_transpiled", "ops_transpiled"}
    assert summary["depth"] >= 1
    assert summary["depth_transpiled"] >= 1
