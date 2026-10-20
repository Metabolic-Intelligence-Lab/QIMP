# Examples

The scripts in `examples/` run on the CPU Aer simulator with the `[dev]`
install. CI runs all of them on every pull request.

| Script | Content |
|---|---|
| `01_frqi_fidelity.py` | FRQI round trip on a random 4x4 image, with MSE and PSNR |
| `02_neqr_round_trip.py` | NEQR round trip with exact recovery |
| `03_qhed_edges.py` | QHED on a synthetic image, horizontal and vertical passes summed |
| `04_qpie_round_trip.py` | QPIE round trip on four qubits, with PSNR |
| `05_real_image_pipeline.py` | every encoder, a geometric operation and QHED on a 16x16 microscopy tile; inputs, outputs and a comparison figure are saved under `data/output/run_<timestamp>/` |

```bash
python examples/01_frqi_fidelity.py
```

Example 05 uses `data/immagini/trainQML/Train_QML_16/` when it is present and
otherwise a 16x16 crop of the Laurdan frame included in the repository.

`scripts/softwarex_figures.py` regenerates the figures and numbers of the
software paper with a fixed simulator seed.
