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
from qimp.encoding.frqi import FrqiEncoder
from qimp.encoding.neqr import neqr_decode
from qimp.testing import exact_counts


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


@pytest.mark.parametrize("n,q", [(1, 1), (1, 2), (2, 2), (2, 3)])
def test_neqr_compressor_round_trip_parametrized(n: int, q: int) -> None:
    """Compressed NEQR must still recover the original image exactly for moderate (n, q)."""
    rng = np.random.default_rng(seed=n * 19 + q)
    side = 1 << n
    image = rng.integers(0, 1 << q, size=(side, side), dtype=np.int64)
    compressor = NeqrCompressor(image, q=q)
    qc = compressor.neqr_image()
    counts = exact_counts(qc)
    decoded = neqr_decode(counts, n=n, q=q)
    np.testing.assert_array_equal(decoded, image)


@pytest.mark.parametrize("n", [1, 2])
def test_frqi_compressor_round_trip(n: int) -> None:
    """Compressed FRQI recovers the same intensities as the brute-force encoder."""
    rng = np.random.default_rng(seed=n * 23)
    side = 1 << n
    # Pick few distinct intensities so compression actually merges minterms.
    image = rng.integers(0, 4, size=(side, side), dtype=np.uint8) * 60
    compressor = FrqiCompressor(image, normalization=255.0)
    qc_compressed = compressor.frqi_image()
    counts_compressed = exact_counts(qc_compressed)

    encoder = FrqiEncoder(normalization=255.0)
    qc_reference = encoder.encode(image)
    counts_reference = exact_counts(qc_reference)

    # Decode both via the FRQI decoder (same n, m=0) and compare.
    from qimp.encoding.frqi import frqi_decode

    decoded_compressed = frqi_decode(counts_compressed, n=n, m=0, normalization=255.0)
    decoded_reference = frqi_decode(counts_reference, n=n, m=0, normalization=255.0)
    np.testing.assert_allclose(decoded_compressed, decoded_reference, atol=1e-6)


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
