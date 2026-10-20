# QIMP: quantum image processing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Qiskit](https://img.shields.io/badge/qiskit-%E2%89%A51.0-purple.svg)](https://qiskit.org/)

Python library for quantum image processing on Qiskit: FRQI, NEQR and QPIE
encodings, geometric, chromatic and arithmetic operations, quantum Hadamard
edge detection (QHED), variational classification of encoded images, and
figures of merit.

Designed and developed by G. Maulucci (Metabolic Intelligence Lab, Università
Cattolica del Sacro Cuore). The standard QIMP circuits it covers are also
surveyed in C. Dolciami, "A quantum circuit library for image processing",
M.Sc. thesis, Politecnico di Torino, 2022.

The core (FRQI, NEQR, QPIE, processing, testing and metrics) is released. The
GP-ratio application and the QML extensions are optional subpackages. The
repository is public and is the archive of record for the manuscript listed
below.

## Paper

*Autonomous Ratiometric Image Processing on a Superconducting Quantum
Processor* (G. Maulucci, 2026, submitted). The tag `v2-submission` pins the
code, the derived data and the hardware outputs reported in the manuscript.
The manuscript itself is not in the repository.

- `src/qimp/processing/arithmetic.py`: reversible arithmetic (adders, constant
  multiplier, restoring and non-restoring dividers with exact inverses,
  truth-table synthesis `synth_pixel_map`, `q_div_lookup`).
- `src/qimp/processing/ratiometric_circuit.py`: the Class A, B and C
  ratiometric circuits (`class_b_ratio(..., load="ucry")`, `class_a_gp_lookup`,
  `class_c_rogfp_lookup`) and the amplitude-estimation oracle.
- `paper/HW_CAMPAIGN_3_PROTOCOL.md` to `_20_`: the pre-registered hardware
  campaigns, with SHA-256 fingerprints and deviation logs.
- `paper/data_autonomous/`: derived datasets, resource tables and analysis
  outputs cited in the manuscript, including the job manifest of
  Supplementary Table S1.
- `data/output/ibm_hw/<UTC-timestamp>/runs/<label>/`: raw counts, transpiled
  circuit (QPY) and metadata of each IBM Quantum job. `data/output/noise_sim/`
  holds the runs on the calibrated noise model.
- `scripts/`: each analysis is a script that regenerates its table or figure
  from the archived data without a device (for example `analyse_hw_signal.py`,
  `decode_bias_corrected.py`, `arithmetic_identity.py`, `score_campaign17.py`,
  `calibrate_noise_model.py`). The hardware drivers are
  `run_hardware_class_b_nonrestoring.py` and `run_campaign_*.{py,sh}`.

## Design constraints

- Circuits scale with the number of qubits. Encoders accept any `n` (spatial
  qubits, image side 2^n), `q` (intensity qubits, NEQR) and `m` (number of
  stacked images, multi-image FRQI). No sizes are hard-coded, and the tests
  are parametrised over `n` and `q`.
- Qiskit 1.0 or later, Python 3.10 or later, `src/` layout, hatchling build,
  ruff, mypy and pytest.

## Install

The package is not on PyPI. Install it from the repository (Python 3.10 or
later):

```bash
# Editable install from a clone, for development
git clone https://github.com/Metabolic-Intelligence-Lab/QIMP.git
cd QIMP
pip install -e ".[dev]"

# Or install a tagged version directly
pip install "git+https://github.com/Metabolic-Intelligence-Lab/QIMP.git@v0.3.1"
```

Optional extras: `[ibm]` (IBM Quantum Runtime), `[gpu]` (Aer GPU), `[qml]`
(qiskit-machine-learning), `[notebooks]` (JupyterLab), `[docs]`
(mkdocs-material), `[ui]` (Streamlit), `[dev]` (pytest, ruff, mypy,
pre-commit).

## Repository layout

```
repo/
├── src/qimp/         # the library
│   ├── encoding/     # frqi, neqr, qpie, mcrqi, ncqi, compression
│   ├── processing/   # geometric, chromatic, arithmetic, filters, gp_ratio
│   ├── qml/          # variational classifier
│   ├── io/           # image and dataset loaders
│   ├── runtime/      # memory pool, simulator manager, caching
│   ├── qft.py        # QFT wrappers
│   ├── testing.py    # ideal, noisy and device simulation
│   ├── metrics.py    # PSNR, MSE, TV, transpile summary
│   ├── config.py     # ProcessingConfig dataclass
│   └── cli.py        # `qimp` command-line tool
├── tests/            # pytest, parametrised over n and q
├── scripts/          # analyses, figure generators and hardware drivers
├── paper/            # campaign protocols and derived data (data_autonomous/)
├── docs/             # mkdocs site
└── data/             # images and outputs; gitignored except the hardware
    ├── immagini/     #   outputs (output/ibm_hw/), the noise-model runs
    └── output/       #   (output/noise_sim/) and one Laurdan frame
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
reconstructed = encoder.decode(counts)
print("PSNR:", psnr(image, reconstructed))
```

## Interactive explorer

A Streamlit explorer is included. Install the `[ui]` extra and launch it:

```bash
pip install -e ".[ui]"
qimp ui    # or: streamlit run apps/qimp_explorer/app.py
```

It has four pages (Encoder Explorer, Processing Playground, Benchmark,
GP-ratio) for running the encoders and processing operations, saving outputs
to `data/output/run_<timestamp>/` and comparing metrics. See
[`docs/ui.md`](docs/ui.md).

## Hardware execution

`scripts/run_hardware_sweep.py` runs the encoders and the GP pipeline on Aer
(ideal, and noisy through `NoiseModel.from_backend`) and optionally on IBM
Quantum hardware.

```bash
# Aer ideal statevector for all 7 encoders and all sizes
python scripts/run_hardware_sweep.py \
  --image data/immagini/<file>.tif \
  --sizes 1 2 \
  --skip-hw

# Aer with the noise model of a backend (no QPU time)
python scripts/run_hardware_sweep.py \
  --image data/immagini/<file>.tif \
  --sizes 1 2 \
  --skip-hw \
  --backend ibm_kingston

# Full sweep, with hardware runs for the default recipes gp@1, gp@2, frqi_multi@1
python scripts/run_hardware_sweep.py \
  --image data/immagini/<file>.tif \
  --sizes 1 2 \
  --shots 4096

# List the backends visible to the saved IBM Quantum account
python scripts/run_hardware_sweep.py --list-backends
```

Outputs are written to `data/output/ibm/<UTC-timestamp>/`:

- `summary.csv`: one row per encoder, size and pass, with shots, depth, PSNR,
  MSE and the job ID for hardware runs.
- `figures/<label>.png`: classical reference, decoded images and absolute
  difference for each pass.
- `runs/<label>_<pass>/`: `circuit.qpy`, `transpiled.qpy` (when present),
  `counts.json` and `metadata.json`.
- `backend_info.json`: name, qubit count and basis gates of the backend.

By default only a few small recipes run on hardware, to stay within the
monthly QPU time of the IBM Quantum Open plan. Hardware runs need the `[ibm]`
extra (`pip install -e ".[ibm]"`) and an API token saved with
`QiskitRuntimeService.save_account(...)`.

## Citation

If you use this library, please cite:

> Maulucci, G. (2026). *Autonomous Ratiometric Image Processing on a
> Superconducting Quantum Processor*. Submitted. Repository:
> https://github.com/Metabolic-Intelligence-Lab/QIMP, tag `v2-submission`.

> Maulucci, G. *qimp-mi: a tested, size-parametric Python library for quantum
> image processing on Qiskit*. Submitted to SoftwareX.

See also [`CITATION.cff`](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).
