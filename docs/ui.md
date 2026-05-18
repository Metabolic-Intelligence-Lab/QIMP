# Interactive UI

The Streamlit-based **QIMP Explorer** lets you exercise every encoder, every
processing operation, and the GP-ratio pipeline without writing code.

## Install

```bash
pip install -e ".[ui]"
```

The `[ui]` extra adds **streamlit** and **pandas** to the base install.

## Launch

```bash
qimp ui
# or, equivalently:
streamlit run apps/qimp_explorer/app.py
```

A browser tab opens at <http://localhost:8501>.

## Pages

| Page | What you can do |
|---|---|
| **Home** | Pick an image (upload or from `data/immagini/`); preview + image info. |
| **Encoder Explorer** | Run FRQI / NEQR / QPIE round-trips with configurable shots and qubit counts; see PSNR, MSE, depth, and (optionally) the circuit diagram. |
| **Processing Playground** | Apply geometric transforms (flip, rotate, swap, shift), chromatic ops (NEQR), or QHED edge detection; compare against the numpy reference. |
| **Benchmark** | Run all three encoders on the same image; sortable `pandas` table (PSNR + TV + depth + runtime) + bar charts. |
| **GP-ratio** | Microscopy pipeline: classical GP image in all 3 output formats (normalized / uint8 / 16-bit), optional Gaussian + median preprocessing, and parametric quantum circuit construction. |
| **System Info** | Library / Qiskit / Python versions, simulator backend (GPU detection), dataset stats, past-run inventory, cache-clearing buttons, and roadmap stubs. |

Every page has a **Save outputs** button that writes TIFFs + a comparison PNG
to `data/output/run_<timestamp>/`.

## Tips

- The **Home** image is shared across all pages via `st.session_state`. Switch
  pages without reloading; the image persists.
- **FRQI** above n = 3 is impractical because of the multi-controlled RY
  gates. The sidebar defaults to a 4× down-sample for that reason.
- **NEQR** is exact: the *Encoder Explorer* shows ∞ dB PSNR (or a green
  "exact recovery" badge) when the round-trip is bit-perfect.
- **QHED** is a *cyclic* horizontal filter on the row-major flattened image —
  see the warning in [Processing](processing.md#filters-qimpprocessingfilters).
- The full **GP-ratio optimisation** loop is intentionally not run in the UI
  (it takes 10+ minutes per image on a CPU simulator). The page builds the
  parametric circuit and reports depth / gate counts; use a notebook for the
  full COBYLA optimisation.
- The **Encoder Explorer** can run an ideal simulation or a noisy one with a
  configurable per-gate depolarizing probability (slider in the sidebar).
  Useful for sensitivity studies before committing to a real backend.
- The **Processing Playground** exposes every public operation:
  `axis_flip`, `coord_swap`, `ort_rotation`, `pos_shift` (bidirectional,
  arbitrary magnitude), `restr_flip`, `restr_coord_swap`, FRQI and NEQR
  chromatic operations, `apply_qft` / `apply_inverse_qft`, and both QHED
  variants. Parameter widgets adapt per operation.

## Architecture

```
apps/qimp_explorer/
├── app.py                          # Home + image picker
├── app_io.py                          # I/O helpers (streamlit-free, unit-tested)
├── _viz.py                         # matplotlib helpers
└── pages/
    ├── 1_Encoder_Explorer.py
    ├── 2_Processing_Playground.py
    ├── 3_Benchmark.py
    └── 4_GP_Ratio.py
```

The app is a thin layer over the public `qimp.*` API — no quantum logic
lives here. If you find a behaviour you want, see the imports in the page
file and call those same functions in your own scripts.
