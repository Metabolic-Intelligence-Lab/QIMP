# Getting started

## Install

QIMP targets **Python ≥ 3.10** and **Qiskit ≥ 1.0**.

The repository is currently **private**. Installation requires a GitHub PAT with
*Contents: Read* permission. Two options:

```bash
# Editable install from a clone (recommended for development)
git clone https://<TOKEN>@github.com/Metabolic-Intelligence-Lab/QIMP.git
cd QIMP
pip install -e ".[dev]"

# Or install a pinned tag directly with pip
pip install "git+https://<TOKEN>@github.com/Metabolic-Intelligence-Lab/QIMP.git@v0.1.0"
```

A future public release will simplify this to `pip install qimp-mi`.

## Optional extras

| Extra | Adds | When you need it |
|---|---|---|
| `[ibm]` | `qiskit-ibm-runtime` | Running on IBM Quantum hardware via `device_test` |
| `[gpu]` | `qiskit-aer-gpu` | GPU-accelerated Aer simulation for large `n` |
| `[qml]` | `qiskit-machine-learning`, `scikit-learn` | Variational image classifiers (v0.2) |
| `[notebooks]` | `jupyterlab`, `ipywidgets` | Running the example notebooks |
| `[docs]` | `mkdocs-material`, `mkdocstrings[python]` | Building these docs locally |
| `[dev]` | `pytest`, `ruff`, `mypy`, `pre-commit` | Contributors |

Combine extras with commas: `pip install "qimp-mi[ibm,gpu]"`.

## Verifying the install

```python
import qimp
print(qimp.__version__)            # 0.1.0
```

```bash
python examples/01_frqi_fidelity.py
```

You should see an encoded → measured → decoded 4×4 image with a PSNR of ~40 dB.

## Scaling

All public APIs are parametric in qubit count. You can encode arbitrarily large
images (memory permitting) without code changes:

```python
import numpy as np
from qimp.encoding.frqi import FrqiEncoder

# n = 4 → 16x16 image → 2n + 1 = 9 qubits
encoder = FrqiEncoder()
qc = encoder.encode(np.zeros((16, 16), dtype=np.uint8))

# n = 6 → 64x64 image → 2n + 1 = 13 qubits
qc = encoder.encode(np.zeros((64, 64), dtype=np.uint8))
```

The hard wall is your simulator's RAM (or QPU's qubit count), not the library.

## Next steps

- [Encodings](encoding.md) — FRQI vs NEQR vs QPIE: when to use which.
- [Processing](processing.md) — geometric, chromatic, arithmetic, QHED.
- [Examples](examples.md) — runnable scripts under `examples/`.
- [API Reference](api/encoding.md) — auto-generated from docstrings.
