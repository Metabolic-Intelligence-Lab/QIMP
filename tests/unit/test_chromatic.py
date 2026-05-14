"""Tests for qimp.processing.chromatic."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit

from qimp.encoding.neqr import neqr_circuit, neqr_decode
from qimp.processing import chromatic
from qimp.testing import ideal_simulation


def test_neqr_color_complement_round_trip() -> None:
    image = np.array([[0, 1], [2, 3]], dtype=np.int64)
    q = 2
    qc = neqr_circuit(image, q=q)
    chromatic.neqr_color_complement(qc, q=q)
    counts = ideal_simulation(qc, shots=4096)
    decoded = neqr_decode(counts, n=1, q=q)
    expected = (1 << q) - 1 - image
    np.testing.assert_array_equal(decoded, expected)


def test_neqr_color_complement_rejects_bad_q() -> None:
    qc = QuantumCircuit(4)
    with pytest.raises(ValueError):
        chromatic.neqr_color_complement(qc, q=0)


def test_neqr_classify_complement_threshold_zero_is_noop() -> None:
    image = np.array([[0, 1], [2, 3]], dtype=np.int64)
    q = 2
    qc = neqr_circuit(image, q=q)
    chromatic.neqr_classify_complement(qc, q=q, threshold_bit=0)
    counts = ideal_simulation(qc, shots=2048)
    decoded = neqr_decode(counts, n=1, q=q)
    np.testing.assert_array_equal(decoded, image)


def test_neqr_classify_complement_threshold_full_complements() -> None:
    image = np.array([[0, 1], [2, 3]], dtype=np.int64)
    q = 2
    qc = neqr_circuit(image, q=q)
    chromatic.neqr_classify_complement(qc, q=q, threshold_bit=q)
    counts = ideal_simulation(qc, shots=2048)
    decoded = neqr_decode(counts, n=1, q=q)
    expected = (1 << q) - 1 - image
    np.testing.assert_array_equal(decoded, expected)


def test_neqr_classify_complement_rejects_out_of_range() -> None:
    qc = QuantumCircuit(4)
    with pytest.raises(ValueError, match="threshold_bit"):
        chromatic.neqr_classify_complement(qc, q=3, threshold_bit=5)


def test_neqr_half_intensity_rejects_q_le_1() -> None:
    qc = QuantumCircuit(4)
    with pytest.raises(ValueError, match="q"):
        chromatic.neqr_half_intensity(qc, q=1)


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
