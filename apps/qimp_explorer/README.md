# QIMP Explorer

Streamlit app for interactive use and testing of the `qimp-mi` library.

## Install

```bash
pip install -e ".[ui]"   # adds streamlit + pandas to the base install
```

## Launch

Either via the CLI shortcut bundled with the package:

```bash
qimp ui
```

…or directly with Streamlit:

```bash
streamlit run apps/qimp_explorer/app.py
```

A browser tab opens at <http://localhost:8501>.

## Pages

The main page (`app.py`) is a **5-step wizard** that walks through a full
QIMP run end-to-end:

1. **Load** — pick an image from `data/immagini/trainQML/Train_QML_16/` or
   upload a TIFF/PNG.
2. **Preprocess** — optional resize to a power-of-two side (2..64) and
   optional RGB → grayscale conversion.
3. **Encode** — choose FRQI / NEQR / QPIE / MCRQI / NCQI, see qubits + depth.
4. **Process (optional)** — chain geometric / chromatic operations. After
   every applied op a quick ideal-simulation preview is shown side-by-side
   with the input. A **Reset ops** button rewinds to the freshly-encoded
   circuit.
5. **Execute & Export** — run the final circuit on the local simulator (or
   submit to IBM Quantum if `qiskit-ibm-runtime` is installed), save the
   before/after images and the OpenQASM 3 circuit.

In the sidebar, three add-on pages cover supporting workflows:

- **Benchmark** — runs FRQI / NEQR / QPIE on the same image and shows a
  `pandas` table + bar charts of PSNR and transpiled depth.
- **GP-ratio** — Green-Purple microscopy pipeline. Computes the classical
  GP image and constructs the parametric quantum sub-circuit (full
  variational optimisation is left to a notebook).
- **System Info** — Python / Qiskit / library versions, simulator backend
  (CPU / GPU), dataset stats, past-run inventory, and cache controls.

Every save action writes TIFFs to `data/output/run_<timestamp>/`.

## Files

- `app.py` — wizard (Load → Preprocess → Encode → Process → Execute/Export).
- `pages/1_Benchmark.py`
- `pages/2_GP_Ratio.py`
- `pages/3_System_Info.py`
- `app_io.py` — image loaders, output-dir helpers (Streamlit-free for unit tests).
- `_viz.py` — matplotlib panel-grid / bar-chart / circuit-figure helpers,
  plus `upscale_for_display` for crisp NEAREST-neighbor previews.
- `_ibm.py` — OpenQASM 3 export + optional IBM Quantum Runtime submission.
