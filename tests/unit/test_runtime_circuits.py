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
