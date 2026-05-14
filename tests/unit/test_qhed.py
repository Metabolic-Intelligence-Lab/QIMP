"""Tests for qimp.processing.filters (QHED edge detection)."""

from __future__ import annotations

import numpy as np
import pytest

from qimp.processing.filters import qhed_decode, qhed_filter, qhed_full_edges
from qimp.testing import ideal_simulation


def test_qhed_filter_qubit_count(n_qubits: int) -> None:
    image = np.ones((2**n_qubits, 2**n_qubits), dtype=np.float64)
    qc, n, _rms = qhed_filter(image)
    assert n == n_qubits
    assert qc.num_qubits == 2 * n_qubits + 1


def test_qhed_decode_rejects_invalid_n() -> None:
    with pytest.raises(ValueError):
        qhed_decode({}, n=0)


def test_qhed_decode_empty_counts() -> None:
    out = qhed_decode({}, n=2)
    assert out.shape == (4, 4)
    assert np.all(out == 0.0)


def test_qhed_uniform_image_yields_zero_gradients() -> None:
    """A constant image has no edges: all gradients ≈ 0."""
    image = np.full((4, 4), 100.0)
    qc, n, rms = qhed_filter(image)
    counts = ideal_simulation(qc, shots=8000)
    edges = qhed_decode(counts, n=n, rms=rms)
    # Shot-noise floor: < 5% of the mean intensity.
    assert edges.max() < 0.05 * image.mean()


@pytest.mark.slow
def test_qhed_detects_vertical_edge() -> None:
    """A 4x4 image with a sharp vertical edge has large horizontal gradients.

    QHED is a *cyclic* horizontal filter (the decrement is mod 2^(2n+1)), so two
    kinds of edge appear in the output:
      - column 1: the real bright/dark transition inside each row
      - column 3: the wrap-around between row r col 3 and row r+1 col 0
    Both are genuine edges in the flattened amplitude representation. Columns
    0 and 2 should remain at the noise floor.
    """
    image = np.zeros((4, 4))
    image[:, 2:] = 100.0
    qc, n, rms = qhed_filter(image)
    counts = ideal_simulation(qc, shots=80_000)
    edges = qhed_decode(counts, n=n, rms=rms)
    edge_cols = edges[:, [1, 3]]
    flat_cols = edges[:, [0, 2]]
    assert edge_cols.min() > flat_cols.max()


@pytest.mark.slow
def test_qhed_full_edges_combines_both_axes() -> None:
    image = np.zeros((4, 4))
    image[2:, :] = 100.0  # bright bottom half → horizontal edge
    image[:, 2:] += 50.0  # also a bright right band → vertical edge
    qc_h, _n, _rms = qhed_filter(image)
    qc_v, _, _ = qhed_filter(image.T)
    counts_h = ideal_simulation(qc_h, shots=40_000)
    counts_v = ideal_simulation(qc_v, shots=40_000)
    edges = qhed_full_edges(image, counts_h, counts_v)
    assert edges.shape == image.shape
    # Edges form a cross pattern; corners are dark.
    assert edges[1, 1] + edges[1, 2] > edges[0, 0]
