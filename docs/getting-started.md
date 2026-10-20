# Getting started

## Install

`qimp-mi` needs Python 3.10 or later and Qiskit 1.0 or later. It is not on
PyPI; install it from the repository.

```bash
# Editable install from a clone, for development
git clone https://github.com/Metabolic-Intelligence-Lab/QIMP.git
cd QIMP
pip install -e ".[dev]"

# Or install a tagged version directly
pip install "git+https://github.com/Metabolic-Intelligence-Lab/QIMP.git@v0.3.1"
```

## Optional extras

| Extra | Adds | Needed for |
|---|---|---|
| `[ibm]` | `qiskit-ibm-runtime` | Running on IBM Quantum hardware |
| `[gpu]` | `qiskit-aer-gpu` | GPU simulation with Aer |
| `[qml]` | `qiskit-machine-learning`, `scikit-learn` | Interoperability with qiskit-machine-learning |
| `[ui]` | `streamlit`, `pandas` | The interactive explorer (`qimp ui`) |
| `[notebooks]` | `jupyterlab`, `ipywidgets` | Working in notebooks |
| `[docs]` | `mkdocs-material`, `mkdocstrings[python]`, `mkdocs-jupyter` | Building this documentation |
| `[dev]` | `pytest`, `pytest-xdist`, `ruff`, `mypy`, `pre-commit` | Development and tests |

Extras can be combined: `pip install -e ".[dev,ibm]"`.

## Checking the install

```python
import qimp
print(qimp.__version__)
```

```bash
python examples/01_frqi_fidelity.py
pytest                         # default suite; slow and hardware tests are opt-in
```

## Scaling

The public API takes the image size from the input, so the same code works
for any size that fits in the simulator or on the device:

```python
import numpy as np
from qimp.encoding.frqi import FrqiEncoder

encoder = FrqiEncoder()
qc = encoder.encode(np.zeros((16, 16), dtype=np.uint8))   # n = 4, 9 qubits
qc = encoder.encode(np.zeros((64, 64), dtype=np.uint8))   # n = 6, 13 qubits
```

The limit is the memory of the simulator or the qubit count and gate fidelity
of the device.
