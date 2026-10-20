# Examples

Runnable scripts. Each uses only the `[dev]` dependencies and runs on the CPU
Aer simulator in seconds.

| Script | Content |
|---|---|
| `01_frqi_fidelity.py` | FRQI round trip, with PSNR |
| `02_neqr_round_trip.py` | NEQR round trip with exact recovery |
| `03_qhed_edges.py` | QHED edge detection on a synthetic image |
| `04_qpie_round_trip.py` | QPIE round trip |
| `05_real_image_pipeline.py` | every encoder, a geometric operation and QHED on a microscopy tile, with outputs saved to `data/output/` |

Run any of them with:

```bash
python examples/01_frqi_fidelity.py
```
