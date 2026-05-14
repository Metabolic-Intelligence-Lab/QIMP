# Examples

Runnable scripts under `examples/`. Each is self-contained and uses only the
default `[dev]` install.

## 01 — FRQI fidelity round-trip

```bash
python examples/01_frqi_fidelity.py
```

Encodes a random 4×4 image with FRQI, simulates 40 000 shots, decodes, and
reports MSE and PSNR. Typical output: PSNR ≈ 40 dB.

## 02 — NEQR exact round-trip

```bash
python examples/02_neqr_round_trip.py
```

Encodes a deterministic 4×4 image with NEQR (`q=8`), simulates, and verifies
byte-for-byte recovery.

## 03 — QHED edge detection

```bash
python examples/03_qhed_edges.py
```

Detects edges of a synthetic image containing a bright square. Runs QHED on
both the original image (horizontal gradients) and its transpose (vertical
gradients), and sums the two maps.

## 04 — QPIE amplitude encoding

```bash
python examples/04_qpie_round_trip.py
```

Encodes a 4×4 image with QPIE (4 qubits only), simulates, and reports PSNR.

## What's next

Once `[notebooks]` is installed (`pip install "qimp-mi[notebooks]"`), open the
Jupyter versions of these examples under `notebooks/`. They include circuit
diagrams and intermediate visualizations.
