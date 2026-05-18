"""Tests for qimp.encoding.ncqi (NEQR for RGB)."""

from __future__ import annotations

import numpy as np
import pytest

from qimp.encoding.ncqi import NcqiEncoder, ncqi_circuit, ncqi_decode
from qimp.testing import ideal_simulation


def test_ncqi_circuit_qubit_count(n_qubits: int, q_qubits: int) -> None:
    image = np.zeros((2**n_qubits, 2**n_qubits, 3), dtype=np.int64)
    qc = ncqi_circuit(image, q=q_qubits)
    assert qc.num_qubits == 2 * n_qubits + 3 * q_qubits


def test_ncqi_circuit_rejects_non_rgb() -> None:
    with pytest.raises(ValueError, match="RGB"):
        ncqi_circuit(np.zeros((4, 4), dtype=np.int64), q=4)


def test_ncqi_circuit_rejects_overflow() -> None:
    img = np.zeros((2, 2, 3), dtype=np.int64)
    img[0, 0, 0] = 16  # needs q >= 5
    with pytest.raises(ValueError, match="exceeds"):
        ncqi_circuit(img, q=4)


def test_ncqi_circuit_rejects_bad_q() -> None:
    with pytest.raises(ValueError, match="q"):
        ncqi_circuit(np.zeros((2, 2, 3), dtype=np.int64), q=0)


def test_ncqi_decode_rejects_invalid_params() -> None:
    with pytest.raises(ValueError):
        ncqi_decode({}, n=0, q=2)
    with pytest.raises(ValueError):
        ncqi_decode({}, n=2, q=0)


def test_ncqi_round_trip_exact() -> None:
    """NCQI is exact (basis encoding)."""
    image = np.array(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 0, 1], [2, 3, 4]],
        ],
        dtype=np.int64,
    )
    encoder = NcqiEncoder(q=3)
    qc = encoder.encode(image)
    counts = ideal_simulation(qc, shots=4096)
    decoded = encoder.decode(counts)
    np.testing.assert_array_equal(decoded, image)


def test_ncqi_round_trip_n2_q3() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 8, size=(4, 4, 3), dtype=np.int64)
    encoder = NcqiEncoder(q=3)
    qc = encoder.encode(image)
    counts = ideal_simulation(qc, shots=16_384)
    decoded = encoder.decode(counts)
    np.testing.assert_array_equal(decoded, image)


def test_ncqi_encoder_rejects_decode_before_encode() -> None:
    enc = NcqiEncoder(q=2)
    with pytest.raises(RuntimeError):
        enc.decode({})
