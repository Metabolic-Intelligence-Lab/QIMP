"""GP-ratio — Green-Purple microscopy pipeline.

Loads an RGB tile (from `data/immagini/trainQML/` or a custom upload), shows
the three channels, computes the classical GP image, and builds the
parametric quantum sub-circuit used by the lab's optimisation pipeline.

Full COBYLA optimisation is intentionally out of scope here — running it
takes 10+ minutes per image and is best done in a notebook.
"""

from __future__ import annotations

import sys
from pathlib import Path

_QIMP_APP_ROOT = Path(__file__).resolve().parent.parent
if str(_QIMP_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_QIMP_APP_ROOT))


import io
from pathlib import Path

import numpy as np
import streamlit as st
from app_io import (
    DATASET_RGB,
    discover_dataset_images,
    new_output_dir,
    save_tiff,
)
from _viz import panel_grid_figure
from PIL import Image
from qiskit.circuit import Parameter

from qimp.encoding.frqi import frqi_circuit
from qimp.io.image import apply_filters, calculate_gp_image
from qimp.metrics import total_variation, transpile_summary
from qimp.processing.gp_ratio import apply_gp_function, classical_gp_image

st.set_page_config(page_title="GP-ratio", page_icon="🟢🟣", layout="wide")
st.title("🟢🟣 GP-ratio Lab Pipeline")
st.caption(
    "Green-Purple ratio computation: classical reference + parametric quantum "
    "circuit construction (no full optimisation in the UI)."
)


# ----------------------------------------------------------------- Sidebar ----

with st.sidebar:
    st.header("RGB image")
    upload = st.file_uploader("Upload an RGB TIFF/PNG", type=["tif", "tiff", "png"])
    dataset_paths = discover_dataset_images(
        DATASET_RGB, pattern="*.tif", max_items=50, require_nonzero=True
    )
    dataset_choice: Path | None = None
    if dataset_paths:
        labels = [p.name for p in dataset_paths]
        i = st.selectbox(
            f"…or pick from `{DATASET_RGB.relative_to(Path.cwd().parent)}`",
            options=list(range(len(labels))),
            format_func=lambda i: labels[i],
            index=0,
        )
        dataset_choice = dataset_paths[i]

    source = st.radio("Active source", ["Upload", "Dataset"], index=1 if upload is None else 0)
    target_size = st.select_slider(
        "Resize to (side)",
        options=[8, 16, 32],
        value=16,
        help="The lab pipeline uses 16×16 for tractable circuits.",
    )
    G_param = st.slider("GP parameter G (α)", 0.0, 2.0, 0.5, step=0.05)

    st.markdown("**Preprocessing filters (optional)**")
    use_filters = st.checkbox("Apply Gaussian + median before GP", value=False)
    sigma = st.slider("Gaussian σ", 0.1, 3.0, 1.0, step=0.1, disabled=not use_filters)
    median_size = st.slider("Median kernel", 1, 7, 3, step=2, disabled=not use_filters)

    run_classical = st.button("Compute classical GP", type="primary", use_container_width=True)
    run_quantum = st.button("Build quantum GP circuit", use_container_width=True)


# ------------------------------------------------------------- Load image ----

if not (run_classical or run_quantum):
    st.info(
        "Pick an RGB image, then press **Compute classical GP** or **Build quantum GP circuit**."
    )
    st.stop()

try:
    if source == "Upload":
        if upload is None:
            st.error("Upload a file first.")
            st.stop()
        rgb_full = np.asarray(Image.open(io.BytesIO(upload.read())).convert("RGB"))
    else:
        if dataset_choice is None:
            st.error("No dataset image available.")
            st.stop()
        rgb_full = np.asarray(Image.open(dataset_choice).convert("RGB"))
except Exception as exc:
    st.error(f"Failed to load image: {exc}")
    st.stop()

# Resize via PIL (bilinear) to a square 2^n × 2^n.
rgb_pil = Image.fromarray(rgb_full).resize((target_size, target_size), Image.Resampling.LANCZOS)
rgb = np.asarray(rgb_pil)
red = rgb[:, :, 0].astype(np.float64)
green = rgb[:, :, 1].astype(np.float64)
blue = rgb[:, :, 2].astype(np.float64)
n = int(np.log2(target_size))

if use_filters:
    red = apply_filters(red, sigma=sigma, median_size=median_size)
    green = apply_filters(green, sigma=sigma, median_size=median_size)
    blue = apply_filters(blue, sigma=sigma, median_size=median_size)

st.markdown(f"**Working RGB tile:** {target_size}×{target_size} (n = {n})")
st.pyplot(
    panel_grid_figure(
        [("Red", red), ("Green", green), ("Blue", blue)],
        cols=3,
    )
)


# -------------------------------------------------- Classical GP computation ----

if run_classical:
    classical_gp = calculate_gp_image(green, red, G=G_param, output_format="16bit")
    gp_uint8 = calculate_gp_image(green, red, G=G_param, output_format="uint8")
    reference = classical_gp_image(green, red)
    st.markdown("### Classical GP image (every output format)")
    st.pyplot(
        panel_grid_figure(
            [
                ("GP normalized [-1, 1]", reference),
                ("GP uint8 [0, 255]", gp_uint8),
                ("GP 16-bit [0, 4096]", classical_gp),
            ],
            cols=3,
        )
    )
    st.write(
        {
            "GP shape": classical_gp.shape,
            "16-bit range": [int(classical_gp.min()), int(classical_gp.max())],
            "uint8 range": [int(gp_uint8.min()), int(gp_uint8.max())],
            "raw range": [round(float(reference.min()), 3), round(float(reference.max()), 3)],
            "TV (normalized)": round(total_variation(reference), 3),
            "filters applied": (f"σ={sigma}, median={median_size}" if use_filters else "none"),
        }
    )
    if st.button("Save classical outputs", key="save_classical", use_container_width=True):
        out_dir = new_output_dir(prefix="gp_classical")
        save_tiff(red, out_dir / "00_red.tif")
        save_tiff(green, out_dir / "01_green.tif")
        save_tiff(classical_gp, out_dir / "02_gp_16bit.tif")
        save_tiff(gp_uint8, out_dir / "03_gp_uint8.tif")
        save_tiff(reference, out_dir / "04_gp_normalized.tif")
        st.success(f"Saved to {out_dir}")


# ---------------------------------------------- Quantum parametric circuit ----

if run_quantum:
    st.markdown("### Quantum GP circuit (parametric)")
    # Multi-image FRQI on (red, green) → 2n + 1 + 1 = 2n + 2 qubits.
    rg_stack = np.stack([red, green], axis=0)
    max_val = float(rg_stack.max())
    if max_val == 0:
        st.error("Both R and G channels are all-zero in this tile; pick another.")
        st.stop()

    base_qc = frqi_circuit(rg_stack, normalization=max_val)
    num_params = 2 * (1 << (2 * n))
    params = [Parameter(f"θ{i}") for i in range(num_params)]
    apply_gp_function(base_qc, n=n, m=1, params=params)

    summary = transpile_summary(base_qc)
    st.write(
        {
            "qubits": base_qc.num_qubits,
            "params": num_params,
            "depth (pre-transpile)": summary["depth"],
            "depth (post-transpile)": summary["depth_transpiled"],
            "op counts": dict(summary["ops"]),
        }
    )
    st.info(
        "Full COBYLA optimisation is intentionally not run in the UI — it takes "
        "tens of minutes per image. Use a notebook for the full pipeline; the "
        "primitives used here are `qimp.processing.gp_ratio.apply_gp_function`, "
        "`qimp.metrics.{psnr,total_variation}`, and `scipy.optimize.minimize`."
    )

    if st.button("Save circuit summary", key="save_quantum", use_container_width=True):
        out_dir = new_output_dir(prefix="gp_quantum")
        with (out_dir / "circuit_summary.txt").open("w", encoding="utf-8") as fh:
            fh.write(f"qubits = {base_qc.num_qubits}\n")
            fh.write(f"params = {num_params}\n")
            fh.write(f"depth = {summary['depth']}\n")
            fh.write(f"depth_transpiled = {summary['depth_transpiled']}\n")
            fh.write(f"ops = {dict(summary['ops'])}\n")
            fh.write(f"ops_transpiled = {dict(summary['ops_transpiled'])}\n")
        save_tiff(red, out_dir / "00_red.tif")
        save_tiff(green, out_dir / "01_green.tif")
        st.success(f"Saved to {out_dir}")
