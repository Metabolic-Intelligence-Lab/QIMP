"""Shared fixtures for the integration suite.

If the lab's real microscopy data is checked out under
``data/immagini/trainQML/``, the fixtures here yield those tiles directly.
Otherwise — in CI, or on a fresh dev machine — we fall back to a small
deterministic synthetic image so the tests still actually run rather than
being silently skipped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAIN_DIR = _REPO_ROOT / "data" / "immagini" / "trainQML" / "Train_QML_16"
_RGB_DIR = _REPO_ROOT / "data" / "immagini" / "trainQML"


def _have_samples(directory: Path, pattern: str = "*.tif") -> bool:
    return directory.exists() and any(directory.glob(pattern))


def _load_first(directory: Path, pattern: str = "*.tif") -> np.ndarray:
    from PIL import Image

    path = sorted(directory.glob(pattern))[0]
    return np.asarray(Image.open(path))


def _synthetic_grayscale_16() -> np.ndarray:
    """A 16×16 uint8 fixture that mimics the lab's low-intensity tiles.

    Deterministic gradient + a faint cross so encoder/decoder tests can
    verify non-trivial structure (max value ≈ 14 keeps NEQR q=4 valid).
    """
    rng = np.random.default_rng(seed=0)
    base = np.linspace(0, 12, 16, dtype=np.float64)
    img = np.outer(np.ones(16), base) + rng.integers(0, 2, size=(16, 16))
    img[8, :] += 2
    img[:, 8] += 2
    return np.clip(img, 0, 14).astype(np.uint8)


def _synthetic_rgb_110() -> np.ndarray:
    """A 110×110 uint8 RGB fixture loosely mimicking lab microscopy frames."""
    rng = np.random.default_rng(seed=1)
    img = rng.integers(0, 30, size=(110, 110, 3), dtype=np.uint8)
    img[40:70, 40:70, 1] += 60  # bright green patch
    img[20:50, 60:90, 0] += 40  # red patch
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture(scope="session")
def real_grayscale_16() -> np.ndarray:
    """16×16 uint8 grayscale tile — real if available, synthetic otherwise."""
    if _have_samples(_TRAIN_DIR):
        return _load_first(_TRAIN_DIR)
    return _synthetic_grayscale_16()


@pytest.fixture(scope="session")
def real_rgb_110() -> np.ndarray:
    """110×110 uint8 RGB tile — real if available, synthetic otherwise."""
    if _have_samples(_RGB_DIR):
        return _load_first(_RGB_DIR)
    return _synthetic_rgb_110()
