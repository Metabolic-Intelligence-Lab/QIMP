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

Under active development. v0.1.0 will ship the core (FRQI/NEQR/QPIE + processing +
testing + metrics) as described in the thesis. The GP-ratio application and QML
extensions are available as optional sub-modules.

## Design constraints

- **Scalable over qubit count.** All encoders accept arbitrary `n` (spatial qubits,
  image side = 2^n), `q` (intensity qubits — NEQR), and `m` (number of stacked
  images — multi-image FRQI). No constants are hard-coded. Test suites are
  parametrized over `n` and `q`.
- **Modern stack.** Qiskit ≥ 1.0, Python ≥ 3.10, `src/`-layout, hatchling build,
  ruff + mypy + pytest.

## Install

The repository is currently **private**. Installation requires a GitHub
Personal Access Token with `repo` scope:

```bash
# Editable install from a clone (recommended for development)
git clone https://<TOKEN>@github.com/Metabolic-Intelligence-Lab/QIMP.git
cd QIMP
pip install -e ".[dev]"

# Or install a pinned tag directly with pip (no clone needed)
pip install "git+https://<TOKEN>@github.com/Metabolic-Intelligence-Lab/QIMP.git@v0.1.0"
```

Substitute `<TOKEN>` with a fresh PAT (https://github.com/settings/tokens →
*Fine-grained* → grant *Contents: Read* on this repo). Don't paste your token
into shell history or chat logs — use a `.netrc` file or `GH_TOKEN`
environment variable instead.

When the project is released publicly (target: v0.2.0), the install will
simplify to:

```bash
pip install qimp-mi
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
├── docs/             # mkdocs site
└── data/             # GITIGNORED: raw images & outputs
    ├── immagini/     # input dataset (microscopy 16-bit TIFFs)
    └── output/       # processing outputs
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

If you use this library in academic work, please cite the underlying thesis:

> Dolciami, C. (2022). *A quantum circuit library for image processing*.
> M.Sc. thesis, Politecnico di Torino.

## License

MIT — see [LICENSE](LICENSE).
