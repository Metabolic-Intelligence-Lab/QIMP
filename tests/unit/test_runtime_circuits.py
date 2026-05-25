"""Unit tests for qimp.runtime.circuits — the recipe factory."""

from __future__ import annotations

import numpy as np
import pytest


def test_downsample_to_n_returns_2n_2n_array():
    from qimp.runtime.circuits import _downsample_to_n

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(20, 20), dtype=np.uint8)
    out = _downsample_to_n(img, n=2)
    assert out.shape == (4, 4)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_downsample_to_n_rgb():
    from qimp.runtime.circuits import _downsample_to_n

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
    out = _downsample_to_n(img, n=1)
    assert out.shape == (2, 2, 3)
    assert out.dtype == np.uint8


def test_downsample_to_n_rejects_unsupported_shape():
    from qimp.runtime.circuits import _downsample_to_n

    bad = np.zeros((5,), dtype=np.uint8)
    with pytest.raises(ValueError, match="unsupported image shape"):
        _downsample_to_n(bad, n=2)


def _toy_grayscale(side: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(side, side), dtype=np.uint8)


def _toy_rgb(side: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(side, side, 3), dtype=np.uint8)


@pytest.mark.parametrize("n", [1, 2])
def test_recipe_frqi_single_channel(n):
    from qimp.runtime.circuits import build_recipes

    img = _toy_grayscale(20)
    recipes = build_recipes(img, n=n)
    rec = next(r for r in recipes if r.encoder == "frqi")

    assert rec.label == f"frqi_n{n}"
    assert rec.qc.num_qubits == 2 * n + 1  # FRQI single-image
    assert rec.reference.shape == (1 << n, 1 << n)


@pytest.mark.parametrize("n", [1, 2])
def test_recipe_frqi_multi_two_channel(n):
    from qimp.runtime.circuits import build_recipes

    img = _toy_rgb(20)
    recipes = build_recipes(img, n=n)
    rec = next(r for r in recipes if r.encoder == "frqi_multi")

    assert rec.label == f"frqi_multi_n{n}"
    assert rec.qc.num_qubits == 2 * n + 1 + 1  # m=1
    assert rec.reference.shape == (1 << n, 1 << n)


@pytest.mark.parametrize("n", [1, 2])
def test_recipe_neqr(n):
    from qimp.runtime.circuits import build_recipes

    img = _toy_grayscale(20)
    recipes = build_recipes(img, n=n, q=2)
    rec = next(r for r in recipes if r.encoder == "neqr")
    assert rec.label == f"neqr_n{n}"
    assert rec.qc.num_qubits == 2 * n + 2  # q=2
    assert rec.reference.shape == (1 << n, 1 << n)


@pytest.mark.parametrize("n", [1, 2])
def test_recipe_qpie(n):
    from qimp.runtime.circuits import build_recipes

    img = _toy_grayscale(20)
    recipes = build_recipes(img, n=n)
    rec = next(r for r in recipes if r.encoder == "qpie")
    assert rec.label == f"qpie_n{n}"
    assert rec.qc.num_qubits == 2 * n
    assert rec.reference.shape == (1 << n, 1 << n)


@pytest.mark.parametrize("n", [1, 2])
def test_recipe_mcrqi(n):
    from qimp.runtime.circuits import build_recipes

    img = _toy_rgb(20)
    recipes = build_recipes(img, n=n)
    rec = next(r for r in recipes if r.encoder == "mcrqi")
    assert rec.label == f"mcrqi_n{n}"
    assert rec.qc.num_qubits == 2 * n + 3
    assert rec.reference.shape == (1 << n, 1 << n, 3)


@pytest.mark.parametrize("n", [1, 2])
def test_recipe_ncqi(n):
    from qimp.runtime.circuits import build_recipes

    img = _toy_rgb(20)
    recipes = build_recipes(img, n=n, q=2)
    rec = next(r for r in recipes if r.encoder == "ncqi")
    assert rec.label == f"ncqi_n{n}"
    assert rec.qc.num_qubits == 2 * n + 3 * 2  # q=2
    assert rec.reference.shape == (1 << n, 1 << n, 3)
