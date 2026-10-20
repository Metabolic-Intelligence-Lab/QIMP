"""Unit tests for qimp.runtime.zne (gate folding + zero-noise extrapolation).

All offline: statevector checks, synthetic data, and a seeded Aer run with
a depolarising noise model.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Statevector

from qimp.encoding.frqi import frqi_circuit
from qimp.runtime.zne import fold_gates, zne_extrapolate, zne_probabilities

_BASIS = ["cx", "rz", "sx", "x"]


def _two_q(qc: QuantumCircuit) -> int:
    return sum(1 for i in qc.data if i.operation.num_qubits >= 2 and i.operation.name != "barrier")


def _isa_frqi(n: int) -> tuple[QuantumCircuit, QuantumCircuit, GenericBackendV2]:
    """FRQI circuit for a random 2^n x 2^n image, measured and transpiled."""
    rng = np.random.default_rng(n)
    qc = frqi_circuit(rng.uniform(0.0, 1.0, size=(2**n, 2**n)), normalization=1.0)
    backend = GenericBackendV2(qc.num_qubits + 1, basis_gates=_BASIS, seed=11)
    measured = qc.copy()
    measured.measure_all()
    isa = transpile(measured, target=backend.target, optimization_level=1, seed_transpiler=3)
    return qc, isa, backend


@pytest.mark.parametrize("scale", [1, 3, 5])
def test_fold_gates_preserves_state_and_scales_two_qubit_count(n_qubits: int, scale: int) -> None:
    if n_qubits > 3:
        pytest.skip("n=4 FRQI (10 qubits, 5x folded) takes ~20 s; covered by n<=3")
    _, isa, backend = _isa_frqi(n_qubits)
    folded = fold_gates(isa, scale, target=backend.target)

    # Unitary part unchanged (up to global phase), measurements kept once.
    sv_ref = Statevector(isa.remove_final_measurements(inplace=False))
    sv_fold = Statevector(folded.remove_final_measurements(inplace=False))
    assert sv_fold.equiv(sv_ref)
    assert folded.count_ops()["measure"] == isa.count_ops()["measure"]
    # Measurements are still the last operation on every measured qubit.
    assert all(i.operation.name in ("measure", "barrier") for i in folded.data[-isa.num_clbits :])

    assert _two_q(folded) == scale * _two_q(isa)
    assert folded.metadata["zne_fold_scale"] == pytest.approx(scale)
    # Inverse gates are rewritten into the ISA; layout is preserved.
    assert set(folded.count_ops()) <= set(backend.target.operation_names) | {"barrier"}
    assert folded.layout is not None


def test_fold_gates_fractional_scale_partial_fold() -> None:
    qc = QuantumCircuit(2)
    for _ in range(5):
        qc.h(0)
        qc.cx(0, 1)
    folded = fold_gates(qc, 2.0)
    # 10 gates, fractional half-fold -> last 5 gates folded once.
    assert folded.metadata["zne_fold_scale"] == pytest.approx(2.0)
    assert len([i for i in folded.data if i.operation.name != "barrier"]) == 20
    assert Statevector(folded).equiv(Statevector(qc))


def test_fold_gates_rejects_mid_circuit_measurement_and_bad_scale() -> None:
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.x(0)
    with pytest.raises(ValueError, match="mid-circuit"):
        fold_gates(qc, 3)
    with pytest.raises(ValueError, match="scale"):
        fold_gates(QuantumCircuit(1), 0.5)


def test_fold_gates_survives_optimising_transpile() -> None:
    """Barriers between copies keep U^dag U from being cancelled."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    folded = fold_gates(qc, 5)
    again = transpile(folded, basis_gates=_BASIS, optimization_level=3)
    assert _two_q(again) == 5


def test_zne_linear_and_richardson_recover_known_trends() -> None:
    scales = [1.0, 3.0, 5.0]
    # Linear trend, normalised at every scale.
    lin = [{"00": 0.9 - 0.05 * s, "11": 0.1 + 0.05 * s} for s in scales]
    out = zne_probabilities(lin, scales, method="linear")
    assert out == pytest.approx({"00": 0.9, "11": 0.1})

    # Quadratic trend: Richardson through three points is exact.
    def quad(s: float) -> float:
        return 0.8 - 0.06 * s + 0.004 * s**2

    q = [{"0": quad(s), "1": 1.0 - quad(s)} for s in scales]
    out = zne_probabilities(q, scales, method="richardson")
    assert out == pytest.approx({"0": 0.8, "1": 0.2})


def test_zne_exponential_recovers_global_depolarising_decay() -> None:
    scales = [1.0, 3.0, 5.0]
    ideal = {"00": 0.7, "01": 0.0, "10": 0.3, "11": 0.0}
    rate = 0.15

    def noisy(s: float) -> dict[str, float]:
        f = math.exp(-rate * s)
        return {k: f * v + (1 - f) * 0.25 for k, v in ideal.items()}

    out = zne_probabilities([noisy(s) for s in scales], scales, method="exponential")
    assert out == pytest.approx({"00": 0.7, "10": 0.3}, abs=1e-9)


def test_zne_extrapolate_clips_renormalises_and_returns_counts() -> None:
    scales = [1, 3, 5]
    counts = [
        {"0": 900, "1": 100},
        {"0": 700, "1": 300},
        {"0": 500, "1": 500},
    ]
    # Linear: p1(0) = 0.0, p0(0) = 1.0; with a steeper drop it would go < 0.
    out = zne_extrapolate(counts, scales, method="linear")
    assert out == {"0": 1000}
    steep = [{"0": 940, "1": 60}, {"0": 700, "1": 300}, {"0": 500, "1": 500}]
    out = zne_extrapolate(steep, scales, method="richardson", shots=4096)
    assert sum(out.values()) == 4096
    assert all(v > 0 for v in out.values())


def test_zne_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="unknown ZNE method"):
        zne_probabilities([{"0": 1}, {"0": 1}], [1, 3], method="pec")
    with pytest.raises(ValueError, match="distinct"):
        zne_probabilities([{"0": 1}, {"0": 1}], [1, 1])
    with pytest.raises(ValueError, match="scales"):
        zne_probabilities([{"0": 1}], [1, 3])


def test_zne_end_to_end_aer_depolarising_beats_scale_one() -> None:
    """Folded circuits on Aer with gate-only depolarising noise: the
    extrapolated distribution is closer (TVD) to the ideal one than the
    unfolded noisy run."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.ry(0.7, 2)
    qc.cx(1, 2)
    qc.sx(1)
    qc.cx(2, 0)
    ideal = Statevector(qc).probabilities_dict()

    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(0.03, 2), ["cx"])
    noise.add_all_qubit_quantum_error(depolarizing_error(0.003, 1), ["sx", "x"])
    sim = AerSimulator(noise_model=noise)

    measured = qc.copy()
    measured.measure_all()
    isa = transpile(measured, basis_gates=_BASIS, optimization_level=1, seed_transpiler=1)
    scales = [1, 3, 5]
    folded = [fold_gates(isa, s) for s in scales]
    result = sim.run(folded, shots=20_000, seed_simulator=7).result()
    raw = [result.get_counts(i) for i in range(len(scales))]

    def tvd(counts: dict[str, float]) -> float:
        total = sum(counts.values())
        keys = set(counts) | set(ideal)
        return 0.5 * sum(abs(counts.get(k, 0) / total - ideal.get(k, 0.0)) for k in keys)

    base = tvd(raw[0])
    for method in ("richardson", "linear", "exponential"):
        est = tvd(zne_extrapolate(raw, scales, method=method))
        assert est < 0.5 * base, (method, est, base)
