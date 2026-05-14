"""Tests for qimp.encoding.neqr."""

from __future__ import annotations

import numpy as np
import pytest

from qimp.encoding.neqr import NeqrEncoder, neqr_circuit, neqr_decode
from qimp.testing import ideal_simulation


def test_neqr_circuit_qubit_count(n_qubits: int, q_qubits: int) -> None:
    image = np.zeros((2**n_qubits, 2**n_qubits), dtype=np.int64)
    qc = neqr_circuit(image, q=q_qubits)
    assert qc.num_qubits == 2 * n_qubits + q_qubits


def test_neqr_circuit_rejects_negative() -> None:
    img = np.array([[-1, 1], [1, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="non-negative"):
        neqr_circuit(img, q=2)


def test_neqr_circuit_rejects_overflow() -> None:
    img = np.array([[0, 8], [0, 0]], dtype=np.int64)  # 8 needs q >= 4
    with pytest.raises(ValueError, match="exceeds"):
        neqr_circuit(img, q=3)


def test_neqr_circuit_rejects_bad_q() -> None:
    with pytest.raises(ValueError, match="q"):
        neqr_circuit(np.zeros((2, 2), dtype=np.int64), q=0)


def test_neqr_decode_rejects_invalid_params() -> None:
    with pytest.raises(ValueError):
        neqr_decode({}, n=0, q=2)
    with pytest.raises(ValueError):
        neqr_decode({}, n=2, q=0)


def test_neqr_round_trip_n1_q2() -> None:
    """NEQR is exact, so even a small shot count recovers the image perfectly."""
    image = np.array([[0, 1], [2, 3]], dtype=np.int64)
    qc = neqr_circuit(image, q=2)
    counts = ideal_simulation(qc, shots=4096)
    decoded = neqr_decode(counts, n=1, q=2)
    np.testing.assert_array_equal(decoded, image)


def test_neqr_round_trip_n2_q3() -> None:
    image = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [7, 6, 5, 4],
            [3, 2, 1, 0],
        ],
        dtype=np.int64,
    )
    qc = neqr_circuit(image, q=3)
    counts = ideal_simulation(qc, shots=8192)
    decoded = neqr_decode(counts, n=2, q=3)
    np.testing.assert_array_equal(decoded, image)


def test_neqr_encoder_round_trip() -> None:
    image = np.array([[1, 2], [3, 0]], dtype=np.int64)
    encoder = NeqrEncoder(q=2)
    qc = encoder.encode(image)
    counts = ideal_simulation(qc, shots=4096)
    decoded = encoder.decode(counts)
    np.testing.assert_array_equal(decoded, image)


def test_neqr_encoder_rejects_decode_before_encode() -> None:
    enc = NeqrEncoder(q=2)
    with pytest.raises(RuntimeError):
        enc.decode({})
