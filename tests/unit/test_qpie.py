"""Tests for qimp.encoding.qpie."""

from __future__ import annotations

import numpy as np
import pytest

from qimp.encoding.qpie import QpieEncoder, qpie_circuit, qpie_decode
from qimp.testing import ideal_simulation


def test_qpie_circuit_qubit_count(n_qubits: int) -> None:
    image = np.ones((2**n_qubits, 2**n_qubits), dtype=np.float64)
    qc = qpie_circuit(image)
    assert qc.num_qubits == 2 * n_qubits


def test_qpie_circuit_rejects_non_square() -> None:
    with pytest.raises(ValueError, match="square"):
        qpie_circuit(np.zeros((4, 8)))


def test_qpie_circuit_rejects_non_pow2() -> None:
    with pytest.raises(ValueError, match="power of two"):
        qpie_circuit(np.zeros((3, 3)))


def test_qpie_circuit_rejects_negative() -> None:
    img = np.array([[-1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="non-negative"):
        qpie_circuit(img)


def test_qpie_decode_rejects_invalid_n() -> None:
    with pytest.raises(ValueError):
        qpie_decode({}, n=0)


def test_qpie_decode_empty_counts() -> None:
    out = qpie_decode({}, n=2)
    assert out.shape == (4, 4)
    assert np.all(out == 0.0)


@pytest.mark.slow
def test_qpie_round_trip_uniform_image() -> None:
    image = np.full((2, 2), 100.0)
    encoder = QpieEncoder()
    qc = encoder.encode(image)
    counts = ideal_simulation(qc, shots=40_000)
    decoded = encoder.decode(counts)
    np.testing.assert_allclose(decoded, image, rtol=0.05)


@pytest.mark.slow
def test_qpie_round_trip_varying_image() -> None:
    image = np.array([[1.0, 2.0], [3.0, 4.0]])
    encoder = QpieEncoder()
    qc = encoder.encode(image)
    counts = ideal_simulation(qc, shots=80_000)
    decoded = encoder.decode(counts)
    # QPIE preserves intensity proportions up to shot noise.
    np.testing.assert_allclose(decoded, image, rtol=0.10)


def test_qpie_encoder_rejects_decode_before_encode() -> None:
    enc = QpieEncoder()
    with pytest.raises(RuntimeError):
        enc.decode({})


def test_qpie_handles_all_zero_image() -> None:
    image = np.zeros((2, 2))
    qc = qpie_circuit(image)  # encodes as uniform superposition
    assert qc.num_qubits == 2
