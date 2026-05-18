"""Tests for qimp.encoding.mcrqi (FRQI for RGB)."""

from __future__ import annotations

import numpy as np
import pytest

from qimp.encoding.mcrqi import McrqiEncoder, mcrqi_circuit, mcrqi_decode
from qimp.testing import ideal_simulation


def test_mcrqi_circuit_qubit_count(n_qubits: int) -> None:
    image = np.zeros((2**n_qubits, 2**n_qubits, 3), dtype=np.uint8)
    qc = mcrqi_circuit(image)
    assert qc.num_qubits == 2 * n_qubits + 3


def test_mcrqi_circuit_rejects_non_rgb() -> None:
    with pytest.raises(ValueError, match="RGB"):
        mcrqi_circuit(np.zeros((4, 4), dtype=np.uint8))


def test_mcrqi_circuit_rejects_non_square() -> None:
    with pytest.raises(ValueError, match="square"):
        mcrqi_circuit(np.zeros((4, 8, 3), dtype=np.uint8))


def test_mcrqi_circuit_rejects_non_pow2() -> None:
    with pytest.raises(ValueError, match="power of two"):
        mcrqi_circuit(np.zeros((3, 3, 3), dtype=np.uint8))


def test_mcrqi_decode_rejects_invalid_n() -> None:
    with pytest.raises(ValueError):
        mcrqi_decode({}, n=0)


def test_mcrqi_decode_empty_counts() -> None:
    out = mcrqi_decode({}, n=2)
    assert out.shape == (4, 4, 3)
    assert np.all(out == 0.0)


@pytest.mark.slow
def test_mcrqi_round_trip_solid_color() -> None:
    """Encode a tile that is pure red, measure, decode — each channel recovered."""
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[..., 0] = 200  # solid red
    image[..., 1] = 50  # a bit of green
    image[..., 2] = 0  # no blue

    encoder = McrqiEncoder()
    qc = encoder.encode(image)
    counts = ideal_simulation(qc, shots=40_000)
    decoded = encoder.decode(counts)
    assert decoded.shape == (2, 2, 3)
    # Per-channel tolerance ~ shot noise budget.
    np.testing.assert_allclose(decoded[..., 0], 200.0, atol=8.0)
    np.testing.assert_allclose(decoded[..., 1], 50.0, atol=8.0)
    np.testing.assert_allclose(decoded[..., 2], 0.0, atol=8.0)


def test_mcrqi_encoder_class_state() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    encoder = McrqiEncoder()
    encoder.encode(image)
    assert encoder.n == 1
    assert encoder.normalization == 255.0


def test_mcrqi_encoder_rejects_decode_before_encode() -> None:
    enc = McrqiEncoder()
    with pytest.raises(RuntimeError):
        enc.decode({})
