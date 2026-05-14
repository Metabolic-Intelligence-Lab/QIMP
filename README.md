# QIMP — Quantum Image Processing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Qiskit](https://img.shields.io/badge/qiskit-%E2%89%A51.0-purple.svg)](https://qiskit.org/)

Modular Python library for **Quantum Image Processing** on top of Qiskit:
FRQI, NEQR, QPIE encodings, geometric / chromatic / arithmetic processing,
edge detection (QHED), variational QML on encoded images, and figures of merit.

Specification: [`docs/tesi.pdf`](docs/tesi.pdf) (Dolciami, Politecnico di Torino,
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

Once published to PyPI:

```bash
pip install qimp-mi
```

For development from a clone:

```bash
pip install -e ".[dev]"
```

Optional extras: `[ibm]` (IBM Quantum Runtime), `[gpu]` (Aer GPU), `[qml]`
(qiskit-machine-learning), `[notebooks]` (JupyterLab), `[docs]` (mkdocs-material).

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
├── docs/             # tesi.pdf + mkdocs site
├── legacy/           # exploratory scripts (Old/) and pre-package code (scripts/)
└── data/             # GITIGNORED: raw images & outputs
    ├── immagini/     # input dataset (microscopy 16-bit TIFFs)
    └── output/       # processing outputs
```

## Quick start

> **Note:** the package APIs are still being migrated from `legacy/`. The examples
> below describe the target API for v0.1.0.

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

## Citation

If you use this library in academic work, please cite the underlying thesis:

> Dolciami, C. (2022). *A quantum circuit library for image processing*.
> M.Sc. thesis, Politecnico di Torino.

## License

MIT — see [LICENSE](LICENSE).
