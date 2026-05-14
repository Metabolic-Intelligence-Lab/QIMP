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

1. **Home** — pick an image from `data/immagini/trainQML/Train_QML_16/` or
   upload a TIFF/PNG. Sets `session_state["image"]` for the other pages.
2. **Encoder Explorer** — run a single encoder (FRQI / NEQR / QPIE) with
   adjustable shots, qubit counts, and down-sampling. Reports PSNR + circuit
   depth.
3. **Processing Playground** — apply geometric (flip / swap / rotate /
   shift), chromatic, or QHED operations. Shows the quantum result next to a
   numpy reference and a "matches exactly" boolean.
4. **Benchmark** — runs FRQI / NEQR / QPIE on the same image and shows a
   `pandas` table + bar charts of PSNR and transpiled depth.
5. **GP-ratio** — Green-Purple microscopy pipeline. Computes the classical
   GP image and constructs the parametric quantum sub-circuit (full
   variational optimisation is left to a notebook).

Every page has a **Save outputs** button that writes TIFFs + a comparison PNG
to `data/output/run_<timestamp>/`.

## Files

- `app.py` — Home, sidebar image picker.
- `pages/1_Encoder_Explorer.py`
- `pages/2_Processing_Playground.py`
- `pages/3_Benchmark.py`
- `pages/4_GP_Ratio.py`
- `_io.py` — image loaders, output-dir helpers (Streamlit-free for unit tests).
- `_viz.py` — matplotlib panel-grid / bar-chart / circuit-figure helpers.
