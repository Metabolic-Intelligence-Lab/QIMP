"""Tests for qimp.encoding.frqi.

These tests are parametrized over n (the spatial qubit count) to ensure the
encoder scales freely without hard-coded sizes.
"""

from __future__ import annotations

import numpy as np
import pytest

from qimp.encoding.frqi import (
    FrqiEncoder,
    angles_to_intensities,
    frqi_circuit,
    frqi_decode,
    intensities_to_angles,
)
from qimp.testing import ideal_simulation


def test_angles_round_trip(n_qubits: int) -> None:
    rng = np.random.default_rng(42)
    intensities = rng.uniform(0.0, 1.0, size=(2**n_qubits, 2**n_qubits))
    angles = intensities_to_angles(intensities, normalization=1.0)
    recovered = angles_to_intensities(angles, normalization=1.0)
    np.testing.assert_allclose(recovered, intensities, atol=1e-10)


def test_intensities_to_angles_endpoints() -> None:
    arr = np.array([0.0, 0.5, 1.0])
    angles = intensities_to_angles(arr, normalization=1.0)
    np.testing.assert_allclose(angles, [0.0, np.pi / 2, np.pi], atol=1e-10)


def test_intensities_to_angles_rejects_bad_normalization() -> None:
    with pytest.raises(ValueError, match="normalization"):
        intensities_to_angles(np.zeros(2), normalization=0.0)


def test_frqi_circuit_qubit_count_scales_with_n(n_qubits: int) -> None:
    image = np.zeros((2**n_qubits, 2**n_qubits), dtype=np.uint8)
    qc = frqi_circuit(image)
    assert qc.num_qubits == 2 * n_qubits + 1


@pytest.mark.parametrize("num_images,expected_m", [(1, 0), (2, 1), (3, 2), (4, 2), (5, 3)])
def test_frqi_circuit_qubit_count_scales_with_m(num_images: int, expected_m: int) -> None:
    images = [np.zeros((4, 4), dtype=np.uint8) for _ in range(num_images)]
    qc = frqi_circuit(images)
    assert qc.num_qubits == 2 * 2 + expected_m + 1


def test_frqi_circuit_rejects_non_square() -> None:
    with pytest.raises(ValueError, match="square"):
        frqi_circuit(np.zeros((4, 8)))


def test_frqi_circuit_rejects_non_pow2() -> None:
    with pytest.raises(ValueError, match="power of two"):
        frqi_circuit(np.zeros((3, 3)))


@pytest.mark.slow
def test_frqi_round_trip_uniform_intensity() -> None:
    """Encode a uniform image, measure, decode — recovered intensity should match.

    Shot-noise budget: with N=40k total shots and 4 pixels, each gets ~10k. With
    p = 100/255, σ_pixel ≈ √(p(1-p)/10000) · 255 ≈ 1.25, so 5σ ≈ 6.
    """
    target_intensity = 100.0
    image = np.full((2, 2), target_intensity, dtype=np.float64)
    qc = frqi_circuit(image, normalization=255.0)
    counts = ideal_simulation(qc, shots=40_000)
    decoded = frqi_decode(counts, n=1, m=0, normalization=255.0)
    assert len(decoded) == 1
    np.testing.assert_allclose(decoded[0], target_intensity, atol=6.0)


@pytest.mark.slow
def test_frqi_round_trip_varying_image() -> None:
    """A 2x2 image with distinct quadrants should be recoverable up to shot noise."""
    image = np.array([[0.0, 64.0], [128.0, 200.0]], dtype=np.float64)
    qc = frqi_circuit(image, normalization=255.0)
    counts = ideal_simulation(qc, shots=50_000)
    decoded = frqi_decode(counts, n=1, m=0, normalization=255.0)[0]
    np.testing.assert_allclose(decoded, image, atol=5.0)


def test_encoder_class_round_trip_dtype() -> None:
    image = np.array([[0, 64], [128, 200]], dtype=np.uint8)
    encoder = FrqiEncoder()
    qc = encoder.encode(image)
    assert encoder.n == 1
    assert encoder.m == 0
    assert encoder.normalization == 255.0
    assert qc.num_qubits == 3


def test_encoder_class_rejects_decode_before_encode() -> None:
    enc = FrqiEncoder()
    with pytest.raises(RuntimeError):
        enc.decode({})


def test_frqi_decode_rejects_invalid_n() -> None:
    with pytest.raises(ValueError):
        frqi_decode({}, n=0)
    with pytest.raises(ValueError):
        frqi_decode({}, n=1, m=-1)


def test_frqi_decode_empty_counts_returns_zeros() -> None:
    result = frqi_decode({}, n=1, m=0)
    assert len(result) == 1
    assert result[0].shape == (2, 2)
    assert np.all(result[0] == 0.0)


def test_frqi_float_image_auto_normalization() -> None:
    """Regression: float images with values > 1 must not be silently clipped.

    Previously the default normalization was 1.0 for any float dtype, which
    saturated all pixels above 1 to π. Now the encoder infers `max(image)` as
    the normalization. Verified at the configuration level here, end-to-end
    in the slow test below.
    """
    image = np.array([[0.0, 64.0], [128.0, 200.0]], dtype=np.float64)
    encoder = FrqiEncoder()
    encoder.encode(image)
    assert encoder.normalization == 200.0


@pytest.mark.slow
def test_frqi_float_image_round_trip_recovers_intensities() -> None:
    from qimp.testing import ideal_simulation

    image = np.array([[0.0, 64.0], [128.0, 200.0]], dtype=np.float64)
    encoder = FrqiEncoder()
    qc = encoder.encode(image)
    counts = ideal_simulation(qc, shots=40_000)
    decoded = encoder.decode(counts)[0]
    np.testing.assert_allclose(decoded, image, atol=8.0)
