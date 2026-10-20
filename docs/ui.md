# Interactive UI

The QIMP Explorer is a Streamlit application for running the encoders, the
processing operations and the GP-ratio pipeline without writing code.

## Install and launch

```bash
pip install -e ".[ui]"        # adds streamlit and pandas
pip install -e ".[ui,ibm]"    # also needed to submit circuits to IBM Quantum
qimp ui                       # or: streamlit run apps/qimp_explorer/app.py
```

The application opens at <http://localhost:8501>.

## Main page

The main page has five steps, each enabled once the previous one is done:

1. Load: upload a TIFF or PNG file, or pick one from `data/immagini/`.
2. Preprocess: resize to a power-of-two side (2 to 64 pixels) and optionally
   convert RGB to grey scale (BT.601 luma).
3. Encode: choose FRQI, NEQR, QPIE, MCRQI or NCQI and the intensity qubits
   where relevant. The step reports the qubit count and the depth before and
   after transpilation.
4. Process (optional): apply geometric operations (flip, swap, rotation) and
   chromatic operations (colour complement, half intensity). After each
   operation an 8192-shot ideal simulation shows the image before and after.
   "Reset ops" restores the encoded circuit.
5. Execute and export: run the circuit on the ideal or noisy simulator, copy
   it as OpenQASM 3, or submit it to an IBM Quantum backend through the
   Sampler primitive. The API token is entered in the sidebar and kept only
   in the session.

Saved outputs are written as TIFF files to `data/output/run_<timestamp>/`.

## Other pages

| Page | Content |
|---|---|
| Benchmark | FRQI, NEQR and QPIE on the same image, with a table of PSNR, total variation, depth and run time, and bar charts |
| GP-ratio | classical GP image in three output formats (normalised, uint8, 16-bit), optional Gaussian and median filtering, and construction of the parametric circuit |
| System Info | library, Qiskit and Python versions, simulator backend and GPU detection, dataset statistics, previous runs, cache clearing |

## Notes

- FRQI circuits grow quickly above n = 3 because of the multi-controlled
  rotations. Resize to 4x4 or 8x8 in the Preprocess step.
- NEQR recovers the image exactly.
- The GP-ratio page builds the parametric circuit and reports its depth and
  gate counts but does not run the optimisation, which takes more than ten
  minutes per image on a CPU simulator. Use `qimp.processing.gp_ratio.optimize_gp`
  in a script for that.

## Structure

```
apps/qimp_explorer/
├── app.py                 # main page
├── app_io.py              # I/O helpers, without streamlit, unit-tested
├── _viz.py                # matplotlib helpers and nearest-neighbour upscaling
├── _ibm.py                # OpenQASM 3 export and IBM Runtime submission
└── pages/
    ├── 1_Benchmark.py
    ├── 2_GP_Ratio.py
    └── 3_System_Info.py
```

The application calls the public `qimp` API and contains no quantum logic of
its own; the imports at the top of each page show which functions it uses.
