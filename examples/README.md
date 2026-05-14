# Examples

Self-contained, runnable Python scripts demonstrating QIMP features. Each one
uses only `[dev]` dependencies and runs on a CPU Aer simulator in seconds.

| Script | What it shows |
|---|---|
| `01_frqi_fidelity.py` | FRQI encoding round-trip + PSNR fidelity measure |
| `02_neqr_round_trip.py` | NEQR exact retrieval — recovers the image byte-for-byte |
| `03_qhed_edges.py` | Quantum Hadamard Edge Detection on a synthetic image |
| `04_qpie_round_trip.py` | QPIE amplitude encoding round-trip |

Run any of them with:

```bash
python examples/01_frqi_fidelity.py
```
