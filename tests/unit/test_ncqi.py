"""Tests for qimp.encoding.ncqi (NEQR for RGB)."""

from __future__ import annotations

import numpy as np
import pytest

from qimp.encoding.ncqi import NcqiEncoder, ncqi_circuit, ncqi_decode
from qimp.testing import exact_counts


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


@pytest.mark.parametrize("n,q", [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)])
def test_ncqi_round_trip_parametrized(n: int, q: int) -> None:
    """NCQI (NEQR for RGB) is exact for every (n, q) — no shot noise tolerated."""
    rng = np.random.default_rng(seed=n * 17 + q)
    side = 1 << n
    image = rng.integers(0, 1 << q, size=(side, side, 3), dtype=np.int64)
    encoder = NcqiEncoder(q=q)
    qc = encoder.encode(image)
    counts = exact_counts(qc)
    decoded = encoder.decode(counts)
    np.testing.assert_array_equal(decoded, image)


@pytest.mark.slow
@pytest.mark.parametrize("n,q", [(3, 2), (3, 3)])
def test_ncqi_round_trip_large(n: int, q: int) -> None:
    """Slow: larger NCQI cases (statevector dim grows as 2^(2n+3q))."""
    rng = np.random.default_rng(seed=n + q)
    side = 1 << n
    image = rng.integers(0, 1 << q, size=(side, side, 3), dtype=np.int64)
    encoder = NcqiEncoder(q=q)
    qc = encoder.encode(image)
    counts = exact_counts(qc)
    decoded = encoder.decode(counts)
    np.testing.assert_array_equal(decoded, image)


def test_ncqi_encoder_rejects_decode_before_encode() -> None:
    enc = NcqiEncoder(q=2)
    with pytest.raises(RuntimeError):
        enc.decode({})
