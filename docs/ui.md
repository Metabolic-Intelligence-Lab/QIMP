# Interactive UI

The Streamlit-based **QIMP Explorer** lets you exercise every encoder and every
processing operation, plus the GP-ratio pipeline, without writing code.

## Install

```bash
pip install -e ".[ui]"
```

The `[ui]` extra adds **streamlit** and **pandas** to the base install. Add
`[ibm]` too (`pip install -e ".[ui,ibm]"`) if you want to submit circuits to
real IBM Quantum hardware from the Execute step.

## Launch

```bash
qimp ui
# or, equivalently:
streamlit run apps/qimp_explorer/app.py
```

A browser tab opens at <http://localhost:8501>.

## The wizard (`app.py`)

The main page is a sequential 5-step wizard. Each step unlocks the next:

1. **Load** — upload a TIFF/PNG or pick a file from `data/immagini/`. Shows the
   raw preview (NEAREST-neighbor upscaled so small images stay crisp).
2. **Preprocess** — optionally resize the image to a power-of-two side
   (2, 4, 8, 16, 32, or 64 px) and / or convert RGB → grayscale (BT.601 luma).
3. **Encode** — pick FRQI / NEQR / QPIE / MCRQI / NCQI, choose intensity-qubit
   counts where applicable, then build the circuit. Reports qubits, pre- and
   post-transpile depth.
4. **Process** *(optional)* — chain geometric (flip, swap, rotate) and
   chromatic (color complement, half intensity) operations. After every
   applied op the wizard runs a quick 8 k-shot ideal simulation, decodes
   it, and shows the input image next to the after-image so you can chain
   ops visually. A **Reset ops** button rebuilds the freshly-encoded
   circuit.
5. **Execute & Export** — run the final circuit on the local simulator (ideal
   or noisy), or copy the **OpenQASM 3** snippet, or submit directly to an
   IBM Quantum backend via the Sampler primitive (paste API token in the
   sidebar — token stays in session memory only, never persisted).

Every save action writes TIFFs to `data/output/run_<timestamp>/`.

## Add-on pages (sidebar)

| Page | What you can do |
|---|---|
| **Benchmark** | Run all three encoders on the same image; sortable `pandas` table (PSNR + TV + depth + runtime) + bar charts. |
| **GP-ratio** | Microscopy pipeline: classical GP image in all 3 output formats (normalized / uint8 / 16-bit), optional Gaussian + median preprocessing, and parametric quantum circuit construction. |
| **System Info** | Library / Qiskit / Python versions, simulator backend (GPU detection), dataset stats, past-run inventory, cache-clearing buttons. |

## Tips

- **FRQI** above n = 3 is impractical because of the multi-controlled RY
  gates. Use the Preprocess step to resize to 4×4 or 8×8 first.
- **NEQR** is exact: with enough shots the round-trip is bit-perfect.
- The full **GP-ratio optimisation** loop is intentionally not run in the UI
  (it takes 10+ minutes per image on a CPU simulator). The page builds the
  parametric circuit and reports depth / gate counts; use a notebook for the
  full COBYLA optimisation.

## Architecture

```
apps/qimp_explorer/
├── app.py                 # 5-step wizard
├── app_io.py              # I/O helpers (streamlit-free, unit-tested)
├── _viz.py                # matplotlib helpers + crisp NEAREST upscaling
├── _ibm.py                # OpenQASM 3 export + IBM Runtime submission
└── pages/
    ├── 1_Benchmark.py
    ├── 2_GP_Ratio.py
    └── 3_System_Info.py
```

The app is a thin layer over the public `qimp.*` API — no quantum logic
lives here. If you find a behaviour you want, see the imports in the page
file and call those same functions in your own scripts.
