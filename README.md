# QIMP — Quantum Image Processing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Qiskit](https://img.shields.io/badge/qiskit-%E2%89%A51.0-purple.svg)](https://qiskit.org/)

Modular Python library for **Quantum Image Processing** on top of Qiskit:
FRQI, NEQR, QPIE encodings, geometric / chromatic / arithmetic processing,
edge detection (QHED), variational QML on encoded images, and figures of merit.

Specification: Dolciami, Politecnico di Torino,
"A quantum circuit library for image processing", 2022) — Chapter 3 defines the
library's intended structure; this implementation extends it with the
Metabolic-Intelligence Lab's Green-Purple ratio pipeline (`qimp.processing.gp_ratio`).

## Status

The core (FRQI/NEQR/QPIE + processing + testing + metrics) is released; the GP-ratio
application and QML extensions are optional sub-modules. The repository is public and
is the archive of record for the manuscript described under [Paper](#paper).

## Paper

*Autonomous Quantum Ratiometric Image Processing: Reversible NEQR Arithmetic from
Bit-Exact Simulation to Superconducting Hardware* (G. Maulucci, 2026, submitted). The
tag `v2-submission` pins the code, the derived data and the hardware outputs the
manuscript reports. The manuscript itself is not in the repository.

- `src/qimp/processing/arithmetic.py` — the reversible-arithmetic primitives (adders,
  constant multiplier, restoring and non-restoring dividers with exact inverses,
  truth-table synthesis `synth_pixel_map`, `q_div_lookup`).
- `src/qimp/processing/ratiometric_circuit.py` — the Class-A/B/C ratiometric circuits
  (`class_b_ratio(..., load="ucry")`, `class_a_gp_lookup`, `class_c_rogfp_lookup`) and the
  amplitude-estimation oracle.
- `paper/HW_CAMPAIGN_3_PROTOCOL.md` … `_20_` — the pre-registered hardware campaigns, with
  SHA-256 fingerprints and deviation logs.
- `paper/data_autonomous/` — the derived datasets, resource tables and analysis outputs
  the manuscript cites (including the per-job manifest of its Supplementary Table S1).
- `data/output/ibm_hw/<UTC-timestamp>/runs/<label>/` — raw counts, transpiled circuit
  (QPY) and job metadata of every IBM Quantum job; `data/output/noise_sim/` — the runs on
  the calibrated noise model.
- `scripts/` — every analysis is a script that regenerates its table or figure from the
  archived data without a device (`analyse_hw_signal.py`, `decode_bias_corrected.py`,
  `arithmetic_identity.py`, `score_campaign17.py`, `calibrate_noise_model.py`, …); the
  hardware drivers are `run_hardware_class_b_nonrestoring.py` and `run_campaign_*.{py,sh}`.

## Design constraints

- **Scalable over qubit count.** All encoders accept arbitrary `n` (spatial qubits,
  image side = 2^n), `q` (intensity qubits — NEQR), and `m` (number of stacked
  images — multi-image FRQI). No constants are hard-coded. Test suites are
  parametrized over `n` and `q`.
- **Modern stack.** Qiskit ≥ 1.0, Python ≥ 3.10, `src/`-layout, hatchling build,
  ruff + mypy + pytest.

## Install

The package is not on PyPI; install it from the repository (Python ≥ 3.10):

```bash
# Editable install from a clone (recommended for development)
git clone https://github.com/Metabolic-Intelligence-Lab/QIMP.git
cd QIMP
pip install -e ".[dev]"

# Or install a pinned tag directly with pip (no clone needed)
pip install "git+https://github.com/Metabolic-Intelligence-Lab/QIMP.git@v2-submission"
```

### Optional extras

`[ibm]` (IBM Quantum Runtime), `[gpu]` (Aer GPU), `[qml]`
(qiskit-machine-learning), `[notebooks]` (JupyterLab), `[docs]` (mkdocs-material),
`[dev]` (pytest, ruff, mypy, pre-commit).

## Repository layout

```
repo/
├── src/qimp/         # The library
│   ├── encoding/     # frqi, neqr, qpie, mcrqi, ncqi, compression
│   ├── processing/   # geometric, chromatic, arithmetic, filters, gp_ratio
│   ├── qml/          # variational classifier
│   ├── io/           # image & dataset loaders
│   ├── runtime/      # memory pool, simulator manager, caching
│   ├── qft.py        # QFT wrappers
│   ├── testing.py    # ideal / noisy / device simulation harness
│   ├── metrics.py    # PSNR, MSE, TV, transpile summary
│   ├── config.py     # ProcessingConfig dataclass
│   └── cli.py        # `qimp` command-line tool
├── tests/            # pytest, parametrized over n and q
├── scripts/          # analyses, figure generators and hardware drivers of the paper
├── paper/            # pre-registered campaign protocols and the derived data (data_autonomous/)
├── docs/             # mkdocs site
└── data/             # raw images & outputs; gitignored except the paper's hardware
    ├── immagini/     #   outputs (output/ibm_hw/), the noise-model runs (output/noise_sim/)
    └── output/       #   and the single Laurdan frame the manuscript's patches derive from
```

## Quick start

```python
import numpy as np
from qimp.encoding.frqi import FrqiEncoder
from qimp.testing import ideal_simulation
from qimp.metrics import psnr

image = np.random.randint(0, 256, (4, 4), dtype=np.uint8)
encoder = FrqiEncoder()
qc = encoder.encode(image)             # 2n+1 = 5 qubits for n=2
counts = ideal_simulation(qc, shots=8192)
reconstructed = encoder.decode(counts, n=2)
print("PSNR:", psnr(image, reconstructed))
```

## Interactive UI

A Streamlit-based explorer ships with the package. Install the optional extra
and launch:

```bash
pip install -e ".[ui]"
qimp ui    # or: streamlit run apps/qimp_explorer/app.py
```

Four pages — Encoder Explorer, Processing Playground, Benchmark, GP-ratio —
let you exercise every encoder + processing operation interactively, save
outputs to `data/output/run_<timestamp>/`, and compare metrics side by side.
See [`docs/ui.md`](docs/ui.md) for screenshots and tips.

## Hardware execution

A sweep script runs the encoder + GP suite on Aer (ideal + noisy via
`NoiseModel.from_backend`) and, optionally, on IBM Quantum hardware.

```bash
# Local only — Aer ideal statevector for all 7 encoders, all sizes
python scripts/run_hardware_sweep.py \
  --image data/immagini/<file>.tif \
  --sizes 1 2 \
  --skip-hw

# Add Aer + backend noise model (no real QPU time)
python scripts/run_hardware_sweep.py \
  --image data/immagini/<file>.tif \
  --sizes 1 2 \
  --skip-hw \
  --backend ibm_kingston

# Full sweep including real hardware on the whitelisted recipes
# (default: gp@1, gp@2, frqi_multi@1)
python scripts/run_hardware_sweep.py \
  --image data/immagini/<file>.tif \
  --sizes 1 2 \
  --shots 4096

# Diagnostic: list backends visible to your saved IBM Quantum account
python scripts/run_hardware_sweep.py --list-backends
```

Outputs land in `data/output/ibm/<UTC-timestamp>/`:

- `summary.csv` — one row per (encoder, n, pass); shots, depth, PSNR, MSE, job_id (when HW).
- `figures/<label>.png` — classical reference + decoded panels + |diff| per pass.
- `runs/<label>_<pass>/{circuit.qpy, transpiled.qpy?, counts.json, metadata.json}` — full reproducibility from disk.
- `backend_info.json` — name, qubit count, basis gates of the chosen backend.

Designed for the **IBM Quantum Open (free) plan** — hardware execution is restricted to a small whitelist by default to stay well within the monthly QPU budget.

Requires the `[ibm]` extra (`pip install -e ".[ibm]"`) and an IBM Quantum API token saved via `QiskitRuntimeService.save_account(...)`.

## Citation

If you use this library in academic work, please cite the paper and the underlying thesis:

> Maulucci, G. (2026). *Autonomous Quantum Ratiometric Image Processing: Reversible NEQR
> Arithmetic from Bit-Exact Simulation to Superconducting Hardware*. Submitted.
> Companion repository: https://github.com/Metabolic-Intelligence-Lab/QIMP, tag `v2-submission`.

> Dolciami, C. (2022). *A quantum circuit library for image processing*.
> M.Sc. thesis, Politecnico di Torino.

## License

MIT — see [LICENSE](LICENSE).
