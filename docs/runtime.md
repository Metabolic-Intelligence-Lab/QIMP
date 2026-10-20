# Simulation and hardware

## Simulation (`qimp.testing`)

| Function | Use |
|---|---|
| `exact_counts(qc)` | noise-free counts computed from the statevector, without sampling; used by the tests |
| `ideal_simulation(qc, shots, seed=None)` | shot sampling on the noise-free Aer simulator; `seed` makes it reproducible |
| `noisy_simulation(qc, shots, noise_model=None)` | shot sampling with an Aer noise model |
| `device_test(qc, shots, backend_name=None)` | execution on an IBM Quantum backend (needs the `[ibm]` extra) |

Measurements are added when the circuit has none. `qimp.runtime.get_simulator`
returns a shared `AerSimulator` and uses the GPU when `qiskit-aer-gpu` is
installed.

## IBM Quantum (`qimp.runtime.ibm`)

The module needs the `[ibm]` extra and an API token saved with
`QiskitRuntimeService.save_account(...)`.

```python
from qimp.runtime.ibm import get_service, hw_run, pick_backend

service = get_service()
backend = pick_backend(service, min_qubits=qc.num_qubits)   # least busy
counts, transpiled, job_id, summary = hw_run(
    qc, backend=backend, shots=4096, mitigation="trex+dd"
)
```

`hw_run` transpiles the circuit for the backend and submits it with the
SamplerV2 primitive. The `mitigation` argument selects:

| Value | Effect |
|---|---|
| `"trex+dd"` (default) | measurement twirling (TREX) and dynamical decoupling |
| `"trex"` | measurement twirling only |
| `"dd"` | dynamical decoupling only (`dd_sequence`, default `"XY4"`) |
| `"zne"` | zero-noise extrapolation on top of TREX and dynamical decoupling |
| `"none"` | no mitigation |

`summary` holds the depth, gate counts and options of the run, and
`persist_run` saves the circuit (QPY), counts and metadata to disk.
`aer_noisy_run` runs the same circuit on Aer with the noise model of a
backend, without using QPU time.

## Zero-noise extrapolation (`qimp.runtime.zne`)

The sampler returns counts, not expectation values, so zero-noise
extrapolation is done on the counts:

1. `fold_gates(qc, scale)` amplifies the gate noise by global folding,
   `U (U^dag U)^k`, after transpilation, with barriers so that the folds are
   not optimised away. Fractional scales fold a final part of the circuit once
   more. Measurements are kept.
2. The folded circuits for the scales (default 1, 3, 5) are submitted in one
   job.
3. `zne_probabilities` and `zne_extrapolate` extrapolate the probability of
   each bitstring to zero noise with a polynomial through all points
   (`"richardson"`), a least-squares line (`"linear"`) or a global
   depolarising decay (`"exponential"`), then clip to [0, 1] and renormalise.

```python
counts, transpiled, job_id, summary = hw_run(
    qc, backend=backend, mitigation="zne", zne_scales=(1, 3, 5), zne_method="linear"
)
summary["zne_raw_counts"]   # counts at each scale, to redo the fit offline
```

The same extrapolation can be tested without an IBM account:

```python
from qiskit.providers.fake_provider import GenericBackendV2
from qimp.runtime.ibm import aer_noisy_run

counts = aer_noisy_run(qc, backend=GenericBackendV2(5), zne_scales=(1, 3, 5), seed=0)
```

Folding amplifies gate errors but not readout errors, which TREX handles.
Richardson extrapolation amplifies shot noise; on hardware the linear fit is
usually more stable.

## Hardware sweep

`scripts/run_hardware_sweep.py` runs every encoder at several sizes on the
ideal simulator, on Aer with a backend noise model and optionally on hardware,
and writes a summary table, figures and the raw runs to
`data/output/ibm/<UTC-timestamp>/`. See the README for the command-line
options.
