"""Tests for qimp.io.image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from qimp.io import image as iio


@pytest.fixture
def square_uint8_png(tmp_path: Path) -> Path:
    arr = np.array(
        [
            [10, 200, 30, 40],
            [50, 60, 70, 255],
            [90, 100, 110, 0],
            [130, 140, 150, 160],
        ],
        dtype=np.uint8,
    )
    p = tmp_path / "square.png"
    Image.fromarray(arr).save(p)
    return p


@pytest.fixture
def rgb_uint8_png(tmp_path: Path) -> Path:
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[..., 0] = 200  # red
    arr[..., 1] = 100  # green
    p = tmp_path / "rgb.png"
    Image.fromarray(arr).save(p)
    return p


def test_image_conv_loads_grayscale(square_uint8_png: Path) -> None:
    arr = iio.image_conv(square_uint8_png)
    assert arr.shape == (4, 4)
    assert arr.dtype == np.uint8


def test_image_conv_rejects_non_pow2(tmp_path: Path) -> None:
    arr = np.zeros((3, 3), dtype=np.uint8)
    p = tmp_path / "three.png"
    Image.fromarray(arr).save(p)
    with pytest.raises(ValueError, match="power of two"):
        iio.image_conv(p)


def test_image_conv_rejects_non_square(tmp_path: Path) -> None:
    arr = np.zeros((2, 4), dtype=np.uint8)
    p = tmp_path / "rect.png"
    Image.fromarray(arr).save(p)
    with pytest.raises(ValueError, match="not square"):
        iio.image_conv(p)


def test_color_image_conv_loads_rgb(rgb_uint8_png: Path) -> None:
    arr = iio.color_image_conv(rgb_uint8_png)
    assert arr.shape == (4, 4, 3)
    assert arr[0, 0, 0] == 200
    assert arr[0, 0, 1] == 100


def test_apply_filters_preserves_shape() -> None:
    img = np.random.rand(8, 8).astype(np.float32)
    out = iio.apply_filters(img, sigma=1.0, median_size=3)
    assert out.shape == img.shape


def test_apply_filters_with_output_buffer() -> None:
    img = np.random.rand(8, 8)
    buf = np.zeros_like(img)
    out = iio.apply_filters(img, out=buf)
    assert out is buf


def test_calculate_gp_normalized_in_range() -> None:
    g = np.array([[1.0, 2.0], [3.0, 4.0]])
    r = np.array([[0.5, 1.0], [1.5, 2.0]])
    out = iio.calculate_gp_image(g, r, G=0.5)
    assert out.shape == g.shape
    assert np.all(out >= -1.0)
    assert np.all(out <= 1.0)


def test_calculate_gp_zero_denominator_is_zero() -> None:
    g = np.array([[0.0]])
    r = np.array([[0.0]])
    assert iio.calculate_gp_image(g, r) == 0.0


def test_calculate_gp_unknown_format_raises() -> None:
    with pytest.raises(ValueError, match="Unknown output_format"):
        iio.calculate_gp_image(np.zeros(2), np.zeros(2), output_format="bogus")


def test_calculate_gp_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        iio.calculate_gp_image(np.zeros(2), np.zeros(3))


@pytest.mark.parametrize("fmt,dtype", [("16bit", np.uint16), ("uint8", np.uint8)])
def test_calculate_gp_output_formats(fmt: str, dtype: np.dtype) -> None:
    g = np.ones((4, 4))
    r = np.zeros((4, 4))
    out = iio.calculate_gp_image(g, r, output_format=fmt)
    assert out.dtype == dtype
