"""Tests for qimp.testing (ideal/noisy simulation harnesses)."""

from __future__ import annotations

from qiskit import QuantumCircuit

from qimp.testing import ideal_simulation, noisy_simulation


def _bell_pair() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def test_ideal_simulation_bell_state() -> None:
    qc = _bell_pair()
    counts = ideal_simulation(qc, shots=4096)
    # Only |00> and |11> outcomes possible (parity preserved).
    for bitstring in counts:
        bits = bitstring.replace(" ", "")
        assert bits in {"00", "11"}, f"unexpected bitstring {bitstring!r}"
    assert sum(counts.values()) == 4096


def test_ideal_simulation_appends_measurements_when_missing() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    counts = ideal_simulation(qc, shots=512)
    assert sum(counts.values()) == 512


def test_noisy_simulation_runs_without_noise_model() -> None:
    counts = noisy_simulation(_bell_pair(), shots=1024)
    assert sum(counts.values()) == 1024
