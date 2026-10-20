# QIMP

`qimp-mi` is a Python library for quantum image processing on Qiskit.

It provides five image encodings (FRQI, NEQR, QPIE and the RGB extensions
MCRQI and NCQI), geometric and chromatic operations, quantum Hadamard edge
detection (QHED), reversible arithmetic on NEQR intensity registers,
Boolean-function compression of the encoders, a variational image classifier,
ratiometric image operators, simulation and IBM Quantum execution helpers, and
image metrics. The standard circuit catalogue is also surveyed in C. Dolciami,
"A quantum circuit library for image processing", M.Sc. thesis, Politecnico di
Torino, 2022.

## Design

- Every circuit is a function of the image size and bit depth. Encoders accept
  any `n` (image side 2^n), `q` (intensity qubits) and `m` (number of stacked
  images, multi-image FRQI). No sizes are hard-coded, and the tests are
  parametrised over `n` and `q`.
- Qiskit 1.0 or later, Python 3.10 or later, `src/` layout, hatchling build,
  ruff and strict mypy.
- Encoding, processing, QML, I/O, runtime and testing are separate
  subpackages. The user interface and the IBM Quantum runtime are optional
  extras.

## Install

```bash
git clone https://github.com/Metabolic-Intelligence-Lab/QIMP.git
cd QIMP
pip install -e ".[dev]"
```

See [Getting started](getting-started.md) for the optional extras.

## Example

```python
import numpy as np
from qimp.encoding.frqi import FrqiEncoder
from qimp.encoding.neqr import NeqrEncoder
from qimp.metrics import psnr
from qimp.testing import exact_counts, ideal_simulation

image = np.random.default_rng(0).integers(0, 256, (4, 4)).astype(np.uint8)

frqi = FrqiEncoder()
qc = frqi.encode(image)                       # 2n + 1 = 5 qubits
decoded = frqi.decode(ideal_simulation(qc, shots=40_000, seed=0))
print("FRQI PSNR:", psnr(image, decoded), "dB")

neqr = NeqrEncoder(q=8)
qc = neqr.encode(image)                       # 2n + q = 12 qubits
assert (neqr.decode(exact_counts(qc)) == image).all()
```

## Contents

- [Getting started](getting-started.md): installation, extras, scaling.
- [Encodings](encoding.md): FRQI, NEQR, QPIE, MCRQI, NCQI and compression.
- [Processing](processing.md): geometric, chromatic, QHED, arithmetic,
  ratiometric operators and GP ratio.
- [Simulation and hardware](runtime.md): simulators, IBM Quantum execution and
  error mitigation.
- [Examples](examples.md): runnable scripts.
- [Interactive UI](ui.md): the Streamlit explorer.
- API reference, generated from the docstrings.

## Citation

> Maulucci, G. *qimp-mi: a tested, size-parametric Python library for quantum
> image processing on Qiskit*. Submitted to SoftwareX.

See also `CITATION.cff` in the repository.

## License

[MIT](https://github.com/Metabolic-Intelligence-Lab/QIMP/blob/main/LICENSE)
