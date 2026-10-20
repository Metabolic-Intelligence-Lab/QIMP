"""Tests for qimp.encoding.compression."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from qimp.encoding.compression import (
    ESPRESSO_AUTO_MIN_WIDTH,
    CompressionMethod,
    FrqiCompressor,
    NeqrCompressor,
    compress_minterms,
    espresso_minimize,
    make_disjoint_cover,
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


# ----------------------------------------------------------------- ESPRESSO ----


def _cube_minterms(implicant: str) -> set[int]:
    """All integer minterms covered by an MSB-first implicant string."""
    width = len(implicant)
    out = {0}
    for j, ch in enumerate(implicant):
        bit = 1 << (width - 1 - j)
        if ch == "1":
            out = {m | bit for m in out}
        elif ch == "-":
            out = out | {m | bit for m in out}
    return out


def _coverage_counts(cover: list[str]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for impl in cover:
        for m in _cube_minterms(impl):
            counts[m] = counts.get(m, 0) + 1
    return counts


def _random_on_set(width: int, density: float, seed: int) -> set[int]:
    rng = np.random.default_rng(seed)
    return {i for i in range(1 << width) if rng.random() < density}


def _structured_on_set(width: int) -> set[int]:
    """Rectangle-like region of a 2^(w/2) x 2^(w/2) grid (an image block)."""
    half = width // 2
    side = 1 << half
    return {
        (r << half) | c
        for r in range(side)
        for c in range(side)
        if r < (3 * side) // 4 and c >= side // 4
    }


def _as_strings(on: set[int], width: int) -> set[str]:
    return {format(m, f"0{width}b") for m in on}


@pytest.mark.parametrize("width", [1, 2, 3, 4, 6, 8])
@pytest.mark.parametrize("density", [0.2, 0.5, 0.9])
def test_espresso_minimize_covers_on_set(width: int, density: float) -> None:
    on = _random_on_set(width, density, seed=width * 7 + int(density * 10))
    cover = espresso_minimize(_as_strings(on, width))
    covered = set(_coverage_counts(cover))
    assert covered == on
    assert len(cover) <= len(on)
    assert all(len(impl) == width for impl in cover)


@pytest.mark.parametrize("width", [2, 3, 4, 6, 8])
def test_espresso_minimize_respects_dont_cares(width: int) -> None:
    rng = np.random.default_rng(seed=width)
    labels = rng.integers(0, 3, size=1 << width)  # 0 off, 1 on, 2 don't-care
    on = {i for i, lab in enumerate(labels) if lab == 1}
    dc = {i for i, lab in enumerate(labels) if lab == 2}
    cover = espresso_minimize(_as_strings(on, width), _as_strings(dc, width))
    covered = set(_coverage_counts(cover))
    assert on <= covered <= on | dc


def test_espresso_minimize_small_cases() -> None:
    assert espresso_minimize([]) == []
    assert espresso_minimize({"10", "11"}) == ["1-"]
    assert espresso_minimize({"00", "01", "10", "11"}) == ["--"]
    assert set(espresso_minimize({"01", "10"})) == {"01", "10"}
    # Don't-cares let a single minterm grow to the full cube.
    assert espresso_minimize({"00"}, {"01", "10", "11"}) == ["--"]


def test_espresso_minimize_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="width"):
        espresso_minimize({"01", "1"})
    with pytest.raises(ValueError, match="allowed"):
        espresso_minimize({"0x"})


@pytest.mark.parametrize("width", [2, 4, 6, 8])
def test_make_disjoint_cover_partitions_union(width: int) -> None:
    on = _random_on_set(width, 0.6, seed=100 + width)
    cover = espresso_minimize(_as_strings(on, width))
    disjoint = make_disjoint_cover(cover)
    counts = _coverage_counts(disjoint)
    assert set(counts) == on
    assert all(c == 1 for c in counts.values())
    assert len(disjoint) <= len(on)


@pytest.mark.parametrize("method", ["qm", "espresso", "auto"])
@pytest.mark.parametrize("width", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("kind", ["random", "structured"])
def test_position_strings_to_implicants_exact_disjoint(
    method: CompressionMethod, width: int, kind: str
) -> None:
    if method == "qm" and width > 8:
        pytest.skip("Quine-McCluskey at width 10 is slow and is not the auto path there")
    on = _random_on_set(width, 0.5, seed=width) if kind == "random" else _structured_on_set(width)
    cover = position_strings_to_implicants(sorted(on), width, method=method)
    counts = _coverage_counts(cover)
    assert set(counts) == on
    assert all(c == 1 for c in counts.values())
    assert len(cover) <= len(on)


@pytest.mark.parametrize("width", [2, 4, 6, 8])
def test_auto_matches_qm_below_threshold(width: int) -> None:
    assert width < ESPRESSO_AUTO_MIN_WIDTH
    on = sorted(_random_on_set(width, 0.5, seed=3 * width))
    auto = position_strings_to_implicants(on, width)
    qm = position_strings_to_implicants(on, width, method="qm")
    assert auto == qm


def test_espresso_beats_greedy_qm_on_block() -> None:
    """On an image block the greedy QM partition fragments, ESPRESSO does not."""
    width = 8
    on = sorted(_structured_on_set(width))
    esp = position_strings_to_implicants(on, width, method="espresso")
    qm = position_strings_to_implicants(on, width, method="qm")
    assert len(esp) <= len(qm)


def test_method_validation() -> None:
    image = np.zeros((2, 2), dtype=np.uint8)
    bad: Any = "pyeda"
    with pytest.raises(ValueError, match="method"):
        FrqiCompressor(image, method=bad)
    with pytest.raises(ValueError, match="method"):
        NeqrCompressor(image.astype(np.int64), q=1, method=bad)
    with pytest.raises(ValueError, match="method"):
        position_strings_to_implicants([0], 2, method=bad)


@pytest.mark.parametrize("n,q", [(1, 1), (1, 3), (2, 2), (2, 4), (3, 2)])
def test_neqr_compressor_espresso_matches_uncompressed(n: int, q: int) -> None:
    """ESPRESSO-compressed NEQR has the same statevector as the reference encoder."""
    from qiskit.quantum_info import Statevector

    from qimp.encoding.neqr import neqr_circuit

    rng = np.random.default_rng(seed=n * 31 + q)
    side = 1 << n
    image = rng.integers(0, 1 << q, size=(side, side), dtype=np.int64)
    compressed = NeqrCompressor(image, q=q, method="espresso").neqr_image()
    reference = neqr_circuit(image, q=q)
    sv_c = Statevector.from_instruction(compressed.remove_final_measurements(inplace=False))
    sv_r = Statevector.from_instruction(reference.remove_final_measurements(inplace=False))
    assert sv_c.equiv(sv_r)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_frqi_compressor_espresso_matches_uncompressed(n: int) -> None:
    """ESPRESSO-compressed FRQI has the same statevector as the reference encoder."""
    from qiskit.quantum_info import Statevector

    rng = np.random.default_rng(seed=n * 41)
    side = 1 << n
    image = rng.integers(0, 4, size=(side, side), dtype=np.uint8) * 60
    compressed = FrqiCompressor(image, normalization=255.0, method="espresso").frqi_image()
    reference = FrqiEncoder(normalization=255.0).encode(image)
    sv_c = Statevector.from_instruction(compressed.remove_final_measurements(inplace=False))
    sv_r = Statevector.from_instruction(reference.remove_final_measurements(inplace=False))
    assert sv_c.equiv(sv_r)


@pytest.mark.slow
@pytest.mark.parametrize("width", [10, 12])
@pytest.mark.parametrize("density", [0.1, 0.5, 0.9])
def test_espresso_large_width_exact_disjoint(width: int, density: float) -> None:
    on = _random_on_set(width, density, seed=width + int(density * 100))
    cover = position_strings_to_implicants(sorted(on), width, method="espresso")
    counts = _coverage_counts(cover)
    assert set(counts) == on
    assert all(c == 1 for c in counts.values())
    assert len(cover) <= len(on)


@pytest.mark.slow
@pytest.mark.parametrize("n", [5])
def test_compressors_auto_uses_espresso_at_threshold(n: int) -> None:
    """At 2n >= ESPRESSO_AUTO_MIN_WIDTH, "auto" equals "espresso" and is exact."""
    assert 2 * n >= ESPRESSO_AUTO_MIN_WIDTH
    side = 1 << n
    image = np.zeros((side, side), dtype=np.uint8)
    image[: side // 2, side // 4 :] = 200
    image[side // 2 :, : side // 2] = 100
    auto = FrqiCompressor(image, method="auto").compress()
    esp = FrqiCompressor(image, method="espresso").compress()
    assert auto == esp
    for intensity, cover in auto.items():
        expected = set(np.flatnonzero(image.flatten() == intensity).tolist())
        counts = _coverage_counts(cover)
        assert set(counts) == expected
        assert all(c == 1 for c in counts.values())
