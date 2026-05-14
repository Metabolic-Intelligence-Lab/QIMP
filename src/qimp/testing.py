"""Test harnesses: ideal/noisy simulation and IBM Quantum hardware execution.

Reference: docs/tesi.pdf §3.1.5
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiskit import QuantumCircuit, transpile

from qimp.runtime import get_simulator

if TYPE_CHECKING:
    from qiskit_aer.noise import NoiseModel

__all__ = ["device_test", "ideal_simulation", "noisy_simulation"]

Counts = dict[str, int]


def _ensure_measured(qc: QuantumCircuit) -> QuantumCircuit:
    """If `qc` has no classical bits or no measurement gates, return a copy with
    `measure_all()` appended."""
    has_measure = any(instr.operation.name == "measure" for instr in qc.data)
    if has_measure and qc.num_clbits >= qc.num_qubits:
        return qc
    new_qc = qc.copy()
    new_qc.measure_all()
    return new_qc


def ideal_simulation(qc: QuantumCircuit, shots: int = 8192) -> Counts:
    """Run on a noise-free Aer simulator and return counts."""
    sim = get_simulator()
    measured = _ensure_measured(qc)
    transpiled = transpile(measured, sim)
    result = sim.run(transpiled, shots=shots).result()
    return dict(result.get_counts())


def noisy_simulation(
    qc: QuantumCircuit,
    shots: int = 8192,
    noise_model: NoiseModel | None = None,
) -> Counts:
    """Run on an Aer simulator with a noise model and return counts.

    If `noise_model` is None, runs with the simulator's default noise (none).
    Pass a `qiskit_aer.noise.NoiseModel.from_backend(backend)` for realistic
    device noise.
    """
    sim = get_simulator()
    measured = _ensure_measured(qc)
    transpiled = transpile(measured, sim)
    run_kwargs: dict[str, Any] = {"shots": shots}
    if noise_model is not None:
        run_kwargs["noise_model"] = noise_model
    result = sim.run(transpiled, **run_kwargs).result()
    return dict(result.get_counts())


def device_test(
    qc: QuantumCircuit,
    shots: int = 4096,
    backend_name: str | None = None,
    instance: str | None = None,
) -> Counts:
    """Run on an IBM Quantum device via the runtime service.

    Requires the `[ibm]` optional dependency (`qiskit-ibm-runtime`) and a saved
    IBM Quantum account (`QiskitRuntimeService.save_account(...)`).

    Parameters
    ----------
    backend_name
        Name of the target backend (e.g. "ibm_brisbane"). If None, picks the
        least-busy operational backend.
    instance
        IBM hub/group/project string. Defaults to whatever is saved.
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
    except ImportError as exc:
        raise ImportError(
            "device_test requires `pip install qimp-mi[ibm]` (qiskit-ibm-runtime)"
        ) from exc

    service = QiskitRuntimeService(instance=instance) if instance else QiskitRuntimeService()
    backend = (
        service.backend(backend_name)
        if backend_name
        else service.least_busy(operational=True, simulator=False)
    )

    measured = _ensure_measured(qc)
    transpiled = transpile(measured, backend=backend)
    sampler = Sampler(mode=backend)
    job = sampler.run([transpiled], shots=shots)
    result = job.result()
    quasi = result[0].data.meas.get_counts()
    return dict(quasi)
