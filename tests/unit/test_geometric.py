"""Tests for qimp.processing.geometric.

End-to-end semantic tests via NEQR (which is exact): apply the geometric op
to the NEQR circuit, simulate, decode, and compare with the classical
numpy operation on the image.
"""

from __future__ import annotations

import numpy as np
import pytest

from qimp.encoding.neqr import neqr_circuit, neqr_decode
from qimp.processing import geometric
from qimp.testing import ideal_simulation


def _make_test_image(n: int) -> np.ndarray:
    """Square image of side 2^n with distinct intensities at each pixel."""
    side = 1 << n
    return np.arange(side * side, dtype=np.int64).reshape(side, side)


def _round_trip(image: np.ndarray, n: int, q: int, op: callable) -> np.ndarray:
    """Encode → apply op (with pos_offset=q for NEQR) → measure → decode."""
    qc = neqr_circuit(image, q=q)
    op(qc, n=n, pos_offset=q)
    counts = ideal_simulation(qc, shots=8192)
    return neqr_decode(counts, n=n, q=q)


@pytest.mark.parametrize("n,q", [(1, 2), (2, 4)])
def test_axis_flip_y(n: int, q: int) -> None:
    img = _make_test_image(n)
    out = _round_trip(img, n, q, lambda qc, **kw: geometric.axis_flip(qc, axis="y", **kw))
    expected = np.fliplr(img)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("n,q", [(1, 2), (2, 4)])
def test_axis_flip_x(n: int, q: int) -> None:
    img = _make_test_image(n)
    out = _round_trip(img, n, q, lambda qc, **kw: geometric.axis_flip(qc, axis="x", **kw))
    expected = np.flipud(img)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("n,q", [(1, 2), (2, 4)])
def test_coord_swap_is_transpose(n: int, q: int) -> None:
    img = _make_test_image(n)
    out = _round_trip(img, n, q, lambda qc, **kw: geometric.coord_swap(qc, **kw))
    np.testing.assert_array_equal(out, img.T)


@pytest.mark.parametrize("n,q", [(1, 2), (2, 4)])
def test_ort_rotation_180(n: int, q: int) -> None:
    img = _make_test_image(n)
    out = _round_trip(img, n, q, lambda qc, **kw: geometric.ort_rotation(qc, angle=180, **kw))
    np.testing.assert_array_equal(out, np.rot90(img, 2))


@pytest.mark.parametrize("n,q", [(1, 2), (2, 4)])
def test_ort_rotation_90(n: int, q: int) -> None:
    img = _make_test_image(n)
    out = _round_trip(img, n, q, lambda qc, **kw: geometric.ort_rotation(qc, angle=90, **kw))
    # The library defines 90° as coord_swap then axis_flip('y'), matching numpy's
    # rot90(image, 1) only after we also account for the (row, col) orientation.
    expected_options = [np.rot90(img, k) for k in (1, 3)]
    assert any(np.array_equal(out, e) for e in expected_options)


def test_ort_rotation_rejects_bad_angle() -> None:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    with pytest.raises(ValueError, match="angle"):
        geometric.ort_rotation(qc, n=1, angle=45)


def test_axis_flip_rejects_bad_axis() -> None:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    with pytest.raises(ValueError, match="axis"):
        geometric.axis_flip(qc, n=1, axis="z")  # type: ignore[arg-type]


@pytest.mark.parametrize("n,q", [(1, 2), (2, 4)])
def test_pos_shift_x_plus_one(n: int, q: int) -> None:
    img = _make_test_image(n)
    out = _round_trip(
        img,
        n,
        q,
        lambda qc, **kw: geometric.pos_shift(qc, axis="x", direction="+", magnitude=1, **kw),
    )
    expected = np.roll(img, shift=1, axis=1)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("n,q", [(1, 2), (2, 4)])
def test_pos_shift_y_minus_one(n: int, q: int) -> None:
    img = _make_test_image(n)
    out = _round_trip(
        img,
        n,
        q,
        lambda qc, **kw: geometric.pos_shift(qc, axis="y", direction="-", magnitude=1, **kw),
    )
    expected = np.roll(img, shift=-1, axis=0)
    np.testing.assert_array_equal(out, expected)


def test_pos_shift_rejects_negative_magnitude() -> None:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    with pytest.raises(ValueError, match="magnitude"):
        geometric.pos_shift(qc, n=1, axis="x", magnitude=-1)


def test_pos_shift_rejects_bad_direction() -> None:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    with pytest.raises(ValueError, match="direction"):
        geometric.pos_shift(qc, n=1, axis="x", direction="*")  # type: ignore[arg-type]


def test_restr_flip_validates_region_bits() -> None:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(4)
    with pytest.raises(ValueError, match="region_bits"):
        geometric.restr_flip(qc, n=2, axis="x", region_bits="012")
    with pytest.raises(ValueError, match="region_bits"):
        geometric.restr_flip(qc, n=2, axis="x", region_bits="000")  # too long


@pytest.mark.parametrize(
    "op_kw",
    [
        {"axis": "y"},
        {"axis": "x"},
    ],
)
def test_axis_flip_rejects_undersized_pos_offset(op_kw: dict) -> None:
    """Validation: pos_offset + 2n must fit in the circuit."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3)
    with pytest.raises(ValueError, match="position register"):
        geometric.axis_flip(qc, n=2, pos_offset=0, **op_kw)


def test_coord_swap_rejects_undersized_pos_offset() -> None:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(4)
    with pytest.raises(ValueError, match="position register"):
        geometric.coord_swap(qc, n=3, pos_offset=0)


@pytest.mark.parametrize("n,q", [(2, 4)])
def test_restr_flip_only_touches_selected_region(n: int, q: int) -> None:
    """``restr_flip(axis='y', region_bits='1')`` should flip columns only on rows
    whose MSB is 1 (the bottom half of a 4-row image with n=2).
    """
    img = _make_test_image(n)
    out = _round_trip(
        img,
        n,
        q,
        lambda qc, **kw: geometric.restr_flip(qc, axis="y", region_bits="1", **kw),
    )
    expected = img.copy()
    expected[2:, :] = np.fliplr(img[2:, :])  # rows 2-3 (MSB=1) get column-flipped
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("n,q", [(2, 4)])
def test_restr_coord_swap_only_touches_selected_region(n: int, q: int) -> None:
    """``restr_coord_swap(region_bits='1')`` swaps coords inside the bottom-right
    quadrant only (where both row MSB and col MSB are 1).
    """
    img = _make_test_image(n)
    out = _round_trip(
        img,
        n,
        q,
        lambda qc, **kw: geometric.restr_coord_swap(qc, region_bits="1", **kw),
    )
    # The quadrant rows=[2:4], cols=[2:4] should be transposed; the rest unchanged.
    expected = img.copy()
    expected[2:4, 2:4] = img[2:4, 2:4].T
    np.testing.assert_array_equal(out, expected)
