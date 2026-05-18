"""Tests for qimp.io.datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from qimp.io.datasets import ImageDataset, batch_process_images


def _write_grayscale(path: Path, side: int = 4, fill: int = 100) -> None:
    arr = np.full((side, side), fill, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _write_rgb(path: Path, side: int = 4) -> None:
    arr = np.zeros((side, side, 3), dtype=np.uint8)
    arr[..., 0] = 200
    Image.fromarray(arr).save(path)


def test_dataset_lists_paths(tmp_path: Path) -> None:
    for i in range(3):
        _write_grayscale(tmp_path / f"img_{i}.tif")
    ds = ImageDataset(tmp_path)
    assert len(ds) == 3
    paths = ds.paths()
    assert all(p.suffix == ".tif" for p in paths)


def test_dataset_iter_grayscale(tmp_path: Path) -> None:
    _write_grayscale(tmp_path / "a.tif")
    _write_grayscale(tmp_path / "b.tif")
    ds = ImageDataset(tmp_path, color="L")
    items = list(ds)
    assert len(items) == 2
    for path, arr in items:
        assert path.exists()
        assert arr.shape == (4, 4)


def test_dataset_iter_rgb(tmp_path: Path) -> None:
    _write_rgb(tmp_path / "rgb.tif")
    ds = ImageDataset(tmp_path, color="RGB")
    items = list(ds)
    assert len(items) == 1
    _, arr = items[0]
    assert arr.shape == (4, 4, 3)


def test_dataset_skips_blank_by_default(tmp_path: Path) -> None:
    _write_grayscale(tmp_path / "blank.tif", fill=0)
    _write_grayscale(tmp_path / "nonblank.tif", fill=50)
    ds = ImageDataset(tmp_path)
    items = list(ds)
    assert len(items) == 1
    assert items[0][0].name == "nonblank.tif"


def test_dataset_limit(tmp_path: Path) -> None:
    for i in range(5):
        _write_grayscale(tmp_path / f"img_{i}.tif")
    ds = ImageDataset(tmp_path, limit=2)
    assert len(ds) == 2


def test_dataset_rejects_bad_color() -> None:
    with pytest.raises(ValueError, match="color"):
        ImageDataset("nonexistent", color="CMYK")


def test_dataset_missing_root_returns_empty(tmp_path: Path) -> None:
    ds = ImageDataset(tmp_path / "does_not_exist")
    assert len(ds) == 0
    assert list(ds) == []


def test_batch_process_images(tmp_path: Path) -> None:
    for i in range(3):
        _write_grayscale(tmp_path / f"img_{i}.tif", fill=i * 50)
    ds = ImageDataset(tmp_path)
    results = batch_process_images(ds, lambda arr: int(arr.mean()))
    assert len(results) == 2  # blank one (i=0) is skipped
    assert results[0][1] == 50  # img_1
    assert results[1][1] == 100  # img_2


def test_batch_process_images_on_error_raise(tmp_path: Path) -> None:
    _write_grayscale(tmp_path / "img.tif")
    ds = ImageDataset(tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        batch_process_images(
            ds,
            lambda arr: (_ for _ in ()).throw(RuntimeError("boom")),
            on_error="raise",
        )


def test_batch_process_images_on_error_skip(tmp_path: Path) -> None:
    _write_grayscale(tmp_path / "img.tif")
    ds = ImageDataset(tmp_path)
    results = batch_process_images(
        ds,
        lambda arr: (_ for _ in ()).throw(RuntimeError("boom")),
        on_error="skip",
    )
    assert results == []


def test_batch_process_invalid_on_error_raises() -> None:
    with pytest.raises(ValueError, match="on_error"):
        batch_process_images([], lambda x: x, on_error="bogus")
