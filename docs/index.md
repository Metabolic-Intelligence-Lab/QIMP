# QIMP

**Quantum Image Processing** on Qiskit, with a modular API and a scalable design.

QIMP implements the three canonical quantum image representations from the
literature — **FRQI**, **NEQR**, and **QPIE** — plus the geometric / chromatic
operations and the Quantum Hadamard Edge Detection (QHED) algorithm that
operate on them. Specifications follow `docs/tesi.pdf` (Dolciami, Politecnico
di Torino, 2022, Chapter 3).

## Design

- **Scalable by construction.** Every encoder accepts arbitrary `n` (image
  side = 2^n), `q` (NEQR intensity qubits), and `m` (multi-image FRQI). No
  constants are hard-coded; tests are parametrized over qubit counts.
- **Modern stack.** Qiskit ≥ 1.0, Python ≥ 3.10, `src/`-layout, hatchling
  build, ruff + mypy strict.
- **Modular.** Encoding, processing, runtime, I/O, and testing live in
  independent sub-packages.

## Install

```bash
pip install qimp-mi
```

For development:

```bash
git clone https://github.com/Metabolic-Intelligence-Lab/QIMP
cd QIMP
pip install -e ".[dev]"
```

Optional extras: `[ibm]`, `[gpu]`, `[qml]`, `[notebooks]`, `[docs]`.

## At a glance

=== "FRQI (phase encoding)"

    ```python
    import numpy as np
    from qimp.encoding.frqi import FrqiEncoder
    from qimp.testing import ideal_simulation
    from qimp.metrics import psnr

    image = np.random.randint(0, 256, (4, 4), dtype=np.uint8)
    encoder = FrqiEncoder()
    qc = encoder.encode(image)          # 2n + 1 = 5 qubits
    counts = ideal_simulation(qc, 40_000)
    decoded = encoder.decode(counts)[0]
    print("PSNR:", psnr(image, decoded), "dB")
    ```

=== "NEQR (basis encoding)"

    ```python
    from qimp.encoding.neqr import NeqrEncoder

    image = np.array([[10, 20], [30, 40]], dtype=np.int64)
    encoder = NeqrEncoder(q=8)
    qc = encoder.encode(image)          # 2n + q = 10 qubits
    counts = ideal_simulation(qc, 4096)
    print(encoder.decode(counts))       # exact recovery
    ```

=== "QPIE (amplitude encoding)"

    ```python
    from qimp.encoding.qpie import QpieEncoder

    encoder = QpieEncoder()
    qc = encoder.encode(image.astype(float))   # 2n = 4 qubits
    counts = ideal_simulation(qc, 80_000)
    print(encoder.decode(counts))
    ```

=== "QHED (edge detection)"

    ```python
    from qimp.processing.filters import qhed_filter, qhed_full_edges

    qc_h, n, rms = qhed_filter(image)
    qc_v, _, _ = qhed_filter(image.T)
    counts_h = ideal_simulation(qc_h, 40_000)
    counts_v = ideal_simulation(qc_v, 40_000)
    edges = qhed_full_edges(image, counts_h, counts_v)
    ```

## Roadmap

| Phase | Status |
|---|---|
| FRQI / NEQR / QPIE encoding + decoding | ✅ shipped (v0.1.0) |
| Geometric / chromatic / filters / QHED | ✅ shipped (v0.1.0) |
| Compression (Quine–McCluskey for FRQI/NEQR) | 📋 v0.2 |
| Arithmetic (q_ADD/SUB, comparator, sort) | 📋 v0.2 |
| MCRQI / NCQI (RGB extensions) | 📋 v0.2 |
| QML variational classifier | 📋 v0.2 |

## Citation

If you use QIMP in academic work, please cite the underlying thesis:

> Dolciami, C. (2022). *A quantum circuit library for image processing*.
> M.Sc. thesis, Politecnico di Torino.

## License

[MIT](https://github.com/Metabolic-Intelligence-Lab/QIMP/blob/main/LICENSE)
