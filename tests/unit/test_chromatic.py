"""Tests for qimp.processing.chromatic."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit

from qimp.encoding.neqr import neqr_circuit, neqr_decode
from qimp.processing import chromatic
from qimp.testing import exact_counts


@pytest.mark.parametrize("n,q", [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 2)])
def test_neqr_color_complement_round_trip(n: int, q: int) -> None:
    rng = np.random.default_rng(seed=n + q * 7)
    side = 1 << n
    image = rng.integers(0, 1 << q, size=(side, side), dtype=np.int64)
    qc = neqr_circuit(image, q=q)
    chromatic.neqr_color_complement(qc, q=q)
    counts = exact_counts(qc)
    decoded = neqr_decode(counts, n=n, q=q)
    expected = (1 << q) - 1 - image
    np.testing.assert_array_equal(decoded, expected)


def test_neqr_color_complement_rejects_bad_q() -> None:
    qc = QuantumCircuit(4)
    with pytest.raises(ValueError):
        chromatic.neqr_color_complement(qc, q=0)


@pytest.mark.parametrize("n,q", [(1, 1), (1, 2), (2, 2), (2, 3)])
def test_neqr_classify_complement_threshold_zero_is_noop(n: int, q: int) -> None:
    rng = np.random.default_rng(seed=n * 5 + q)
    side = 1 << n
    image = rng.integers(0, 1 << q, size=(side, side), dtype=np.int64)
    qc = neqr_circuit(image, q=q)
    chromatic.neqr_classify_complement(qc, q=q, threshold_bit=0)
    counts = exact_counts(qc)
    decoded = neqr_decode(counts, n=n, q=q)
    np.testing.assert_array_equal(decoded, image)


@pytest.mark.parametrize("n,q", [(1, 1), (1, 2), (2, 2), (2, 3)])
def test_neqr_classify_complement_threshold_full_complements(n: int, q: int) -> None:
    rng = np.random.default_rng(seed=n * 11 + q)
    side = 1 << n
    image = rng.integers(0, 1 << q, size=(side, side), dtype=np.int64)
    qc = neqr_circuit(image, q=q)
    chromatic.neqr_classify_complement(qc, q=q, threshold_bit=q)
    counts = exact_counts(qc)
    decoded = neqr_decode(counts, n=n, q=q)
    expected = (1 << q) - 1 - image
    np.testing.assert_array_equal(decoded, expected)


def test_neqr_classify_complement_rejects_out_of_range() -> None:
    qc = QuantumCircuit(4)
    with pytest.raises(ValueError, match="threshold_bit"):
        chromatic.neqr_classify_complement(qc, q=3, threshold_bit=5)


def test_neqr_half_intensity_rejects_q_le_1() -> None:
    qc = QuantumCircuit(4)
    with pytest.raises(ValueError, match="q"):
        chromatic.neqr_half_intensity(qc, q=1, allow_reset=True)


@pytest.mark.parametrize("q", [2, 3])
def test_neqr_half_intensity_round_trip(q: int) -> None:
    """Half-intensity right-shifts each pixel by 1 bit (i → i // 2).

    The implementation uses ``qc.reset(q-1)`` at the end; on a superposition
    image the reset projects bit ``q-1`` onto |0⟩, so the operation only
    behaves as a pure ``i → i // 2`` map for pixels whose LSB (bit 0) is 0
    — i.e., even intensities. The test feeds an even-only image so the
    reset is a deterministic no-op and the semantics hold exactly.
    """
    n = 1
    side = 1 << n
    rng = np.random.default_rng(seed=q)
    # Force even intensities: drop the LSB by multiplying random values by 2
    # and clipping to the q-bit range.
    raw = rng.integers(0, 1 << (q - 1), size=(side, side), dtype=np.int64)
    image = (raw * 2).astype(np.int64)
    assert image.max() < (1 << q)
    qc = neqr_circuit(image, q=q)
    chromatic.neqr_half_intensity(qc, q=q, allow_reset=True)
    counts = exact_counts(qc)
    decoded = neqr_decode(counts, n=n, q=q)
    np.testing.assert_array_equal(decoded, image // 2)


def test_neqr_half_intensity_refuses_reset_by_default() -> None:
    qc = QuantumCircuit(4)
    with pytest.raises(ValueError, match="non-unitary"):
        chromatic.neqr_half_intensity(qc, q=2)


def test_frqi_color_complement_appends_x() -> None:
    qc = QuantumCircuit(3)
    before = qc.depth()
    chromatic.frqi_color_complement(qc)
    assert qc.depth() == before + 1
    last_instr = qc.data[-1]
    assert last_instr.operation.name == "x"


def test_frqi_color_change_appends_unitary() -> None:
    qc = QuantumCircuit(3)
    chromatic.frqi_color_change(qc, theta=0.5)
    last_instr = qc.data[-1]
    assert last_instr.operation.name in ("unitary", "Operator")  # qiskit naming
