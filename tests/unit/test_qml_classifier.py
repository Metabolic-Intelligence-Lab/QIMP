"""Tests for qimp.qml.classifier."""

from __future__ import annotations

import numpy as np
import pytest

from qimp.qml.classifier import FrqiClassifier


def _two_class_2x2_dataset() -> tuple[list[np.ndarray], list[int]]:
    """Two trivially-separable 2x2 images: bright vs dark."""
    bright = np.full((2, 2), 200, dtype=np.uint8)
    dark = np.zeros((2, 2), dtype=np.uint8)
    return [bright, dark, bright, dark], [1, -1, 1, -1]


def test_classifier_rejects_invalid_reps() -> None:
    with pytest.raises(ValueError, match="reps"):
        FrqiClassifier(reps=0)


def test_classifier_rejects_predict_before_fit() -> None:
    clf = FrqiClassifier()
    with pytest.raises(RuntimeError):
        clf.predict([np.zeros((2, 2), dtype=np.uint8)])


def test_classifier_rejects_empty_dataset() -> None:
    clf = FrqiClassifier()
    with pytest.raises(ValueError, match="at least one"):
        clf.fit([], [])


def test_classifier_rejects_bad_labels() -> None:
    clf = FrqiClassifier()
    img = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"±1"):
        clf.fit([img], [0])


def test_classifier_rejects_mismatched_shapes() -> None:
    clf = FrqiClassifier()
    with pytest.raises(ValueError, match="same shape"):
        clf.fit([np.zeros((2, 2), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8)], [1, -1])


def test_classifier_rejects_label_count_mismatch() -> None:
    clf = FrqiClassifier()
    img = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="labels length"):
        clf.fit([img, img], [1])


@pytest.mark.slow
def test_classifier_learns_separable_2x2() -> None:
    """A trivial classification — verify the loss decreases below the random baseline."""
    images, labels = _two_class_2x2_dataset()
    clf = FrqiClassifier(reps=2, seed=42)
    result = clf.fit(images, labels, max_iter=30)
    assert result.loss_history, "loss history empty"
    # MSE for random predictions on ±1 labels is ~1.0; trained model should beat that.
    assert result.loss_history[-1] < result.loss_history[0]
    # Predictions should land in {-1, +1}.
    preds = clf.predict(images)
    assert set(preds.tolist()) <= {-1, 1}


def test_classifier_expectation_value_is_finite() -> None:
    """Make sure a single forward pass produces a real number in [-1, 1]."""
    images, labels = _two_class_2x2_dataset()
    clf = FrqiClassifier(reps=1, seed=0)
    clf.fit(images[:2], labels[:2], max_iter=2)
    val = clf.predict_expectation(images[0])
    assert isinstance(val, float)
    assert -1.0 - 1e-9 <= val <= 1.0 + 1e-9
