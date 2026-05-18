"""Tests for qimp.encoding.compression."""

from __future__ import annotations

import numpy as np
import pytest

from qimp.encoding.compression import (
    FrqiCompressor,
    NeqrCompressor,
    compress_minterms,
    position_strings_to_implicants,
)
from qimp.encoding.neqr import neqr_decode
from qimp.testing import ideal_simulation


def test_compress_minterms_basic_cover() -> None:
    # f = a (i.e. minterms 10, 11) should compress to "1-".
    out = compress_minterms({"10", "11"})
    assert out == ["1-"]


def test_compress_minterms_no_redundancy() -> None:
    # f = a XOR b (minterms 01, 10) cannot be combined further.
    out = compress_minterms({"01", "10"})
    assert set(out) == {"01", "10"}


def test_compress_minterms_full_function() -> None:
    # f = 1 (all 4 minterms) collapses fully.
    out = compress_minterms({"00", "01", "10", "11"})
    assert out == ["--"]


def test_compress_minterms_empty() -> None:
    assert compress_minterms([]) == []


def test_position_strings_to_implicants_width() -> None:
    out = position_strings_to_implicants([0, 1, 2, 3], width=2)
    assert out == ["--"]


def test_position_strings_to_implicants_three_of_four_disjoint() -> None:
    """Disjoint cover for minterms {00, 01, 10}: must cover each exactly once."""
    out = position_strings_to_implicants([0, 1, 2], width=2)
    # Every minterm covered exactly once.
    for m in ("00", "01", "10"):
        matches = sum(
            1 for impl in out if all(ic == "-" or ic == mc for mc, ic in zip(m, impl, strict=True))
        )
        assert matches == 1, f"minterm {m} covered {matches} times by {out}"
    # 11 must NOT be covered (it's not a minterm).
    for impl in out:
        assert not all(ic == "-" or ic == mc for mc, ic in zip("11", impl, strict=True)), (
            f"implicant {impl} accidentally covers non-minterm 11"
        )


def test_neqr_compressor_round_trip_n1_q2() -> None:
    """Compressed NEQR should still recover the original image exactly."""
    image = np.array([[0, 1], [2, 3]], dtype=np.int64)
    compressor = NeqrCompressor(image, q=2)
    qc = compressor.neqr_image()
    counts = ideal_simulation(qc, shots=4096)
    decoded = neqr_decode(counts, n=1, q=2)
    np.testing.assert_array_equal(decoded, image)


def test_neqr_compressor_round_trip_n2_q3() -> None:
    rng = np.random.default_rng(123)
    image = rng.integers(0, 8, size=(4, 4), dtype=np.int64)
    compressor = NeqrCompressor(image, q=3)
    qc = compressor.neqr_image()
    counts = ideal_simulation(qc, shots=16_384)
    decoded = neqr_decode(counts, n=2, q=3)
    np.testing.assert_array_equal(decoded, image)


def test_frqi_compressor_circuit_is_smaller() -> None:
    """For an image with repeated intensities, the compressor must emit fewer
    or equal multi-controlled RY *invocations* than there are non-zero pixels.

    (We don't compare against the brute-force FRQI directly because that
    encoder applies one fully-controlled RY per pixel regardless of intensity,
    while the compressor groups by intensity. The fair check is "fewer RYs
    than non-zero pixels".)
    """
    # 8 non-zero pixels all sharing the same intensity → should collapse to
    # very few implicants.
    image = np.zeros((4, 4), dtype=np.uint8)
    image[:, 2:] = 200

    compressed = FrqiCompressor(image).frqi_image()
    n_nonzero = int((image != 0).sum())
    comp_ry = sum(
        1 for instr in compressed.data if instr.operation.name in ("ry", "cry", "annotated")
    )
    assert comp_ry < n_nonzero, f"compression failed to reduce gate count: {comp_ry} >= {n_nonzero}"


def test_frqi_compressor_groups_only_nonzero() -> None:
    image = np.array([[0, 100], [0, 100]], dtype=np.uint8)
    compressor = FrqiCompressor(image)
    groups = compressor.turn_to_funct()
    # Only intensity=100 should appear (zero pixels are skipped).
    assert set(groups.keys()) == {100}
    assert sorted(groups[100]) == [1, 3]


def test_compression_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="square"):
        FrqiCompressor(np.zeros((4, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="power of two"):
        FrqiCompressor(np.zeros((3, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="q"):
        NeqrCompressor(np.zeros((2, 2), dtype=np.int64), q=0)
    with pytest.raises(ValueError, match="range"):
        NeqrCompressor(np.array([[0, 99], [0, 0]], dtype=np.int64), q=2)
