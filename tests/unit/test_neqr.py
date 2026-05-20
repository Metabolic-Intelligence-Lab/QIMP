"""Tests for qimp.encoding.neqr."""

from __future__ import annotations

import numpy as np
import pytest

from qimp.encoding.neqr import NeqrEncoder, neqr_circuit, neqr_decode
from qimp.testing import exact_counts


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


@pytest.mark.parametrize(
    "n,q",
    [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 1),
        (3, 2),
        # (n=3, q=3) and beyond are exercised by the slow round-trip suite —
        # the statevector dimension (2^(2n+q)) becomes expensive to compute.
    ],
)
def test_neqr_round_trip_parametrized(n: int, q: int) -> None:
    """NEQR is exact for any (n, q): statevector counts must round-trip bit-perfect.

    Uses ``exact_counts`` so there's no shot noise — any mismatch is a real
    encoder/decoder bug, not a flake.
    """
    rng = np.random.default_rng(seed=n * 31 + q)
    side = 1 << n
    image = rng.integers(0, 1 << q, size=(side, side), dtype=np.int64)
    qc = neqr_circuit(image, q=q)
    counts = exact_counts(qc)
    decoded = neqr_decode(counts, n=n, q=q)
    np.testing.assert_array_equal(decoded, image)


@pytest.mark.slow
@pytest.mark.parametrize("n,q", [(3, 3), (4, 2)])
def test_neqr_round_trip_large(n: int, q: int) -> None:
    rng = np.random.default_rng(seed=n + q)
    side = 1 << n
    image = rng.integers(0, 1 << q, size=(side, side), dtype=np.int64)
    qc = neqr_circuit(image, q=q)
    counts = exact_counts(qc)
    decoded = neqr_decode(counts, n=n, q=q)
    np.testing.assert_array_equal(decoded, image)


def test_neqr_encoder_round_trip() -> None:
    image = np.array([[1, 2], [3, 0]], dtype=np.int64)
    encoder = NeqrEncoder(q=2)
    qc = encoder.encode(image)
    counts = exact_counts(qc)
    decoded = encoder.decode(counts)
    np.testing.assert_array_equal(decoded, image)


def test_neqr_encoder_rejects_decode_before_encode() -> None:
    enc = NeqrEncoder(q=2)
    with pytest.raises(RuntimeError):
        enc.decode({})
