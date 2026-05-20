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


def test_classifier_expectation_distinguishes_classes() -> None:
    """After fitting on a separable dataset, the expectation must actually
    differ between classes — not just be a finite number.

    This catches degenerate ansatz / parameter-binding bugs where the
    classifier produces the same output regardless of input.
    """
    images, labels = _two_class_2x2_dataset()
    clf = FrqiClassifier(reps=2, seed=7)
    clf.fit(images, labels, max_iter=30)

    bright_exp = clf.predict_expectation(images[0])  # label +1
    dark_exp = clf.predict_expectation(images[1])  # label -1

    assert isinstance(bright_exp, float)
    assert isinstance(dark_exp, float)
    # Expectations live in [-1, 1] by construction (Hermitian observable with
    # unit eigenvalues), but more importantly the two classes must be
    # distinguishable: the gap should exceed pure shot/optim noise.
    assert abs(bright_exp - dark_exp) > 0.05, (
        f"classifier is degenerate: bright={bright_exp:.4f}, dark={dark_exp:.4f}"
    )


def test_classifier_repr_works_on_empty_history() -> None:
    """``FrqiClassifierResult.__repr__`` must not crash when fit produced no
    iterations (this used to fail with a malformed conditional inside the
    f-string format-spec).
    """
    from qimp.qml.classifier import FrqiClassifierResult

    result = FrqiClassifierResult(
        params=np.array([0.0]),
        loss_history=[],
        n_iter=0,
    )
    text = repr(result)
    assert "n_iter=0" in text
    assert "nan" in text
