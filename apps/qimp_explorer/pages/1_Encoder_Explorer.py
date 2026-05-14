"""Encoder Explorer — FRQI / NEQR / QPIE round-trip on the current image."""

from __future__ import annotations

import time

import numpy as np
import streamlit as st
from apps.qimp_explorer._io import (
    infer_n_from_image,
    is_power_of_two,
    new_output_dir,
    save_named_panels,
)
from apps.qimp_explorer._viz import panel_grid_figure, safe_circuit_figure

from qimp.encoding.frqi import FrqiEncoder
from qimp.encoding.neqr import NeqrEncoder
from qimp.encoding.qpie import QpieEncoder
from qimp.metrics import mse, psnr, transpile_summary
from qimp.testing import ideal_simulation

st.set_page_config(page_title="Encoder Explorer", page_icon="🧬", layout="wide")
st.title("🧬 Encoder Explorer")

image = st.session_state.get("image")
if image is None:
    st.warning("Load an image from the **Home** page first.")
    st.stop()

if image.ndim != 2:
    st.error("Encoder Explorer works on 2D grayscale images. Head to **GP-ratio** for RGB images.")
    st.stop()


# -------------------------------------------------------------- Sidebar UI ----

with st.sidebar:
    st.header("Encoding")
    encoding = st.radio("Method", ["FRQI", "NEQR", "QPIE"], horizontal=True)

    downsample = 1
    q_qubits = 4
    shots = 40_000

    if encoding == "FRQI":
        downsample = st.select_slider(
            "Down-sample factor",
            options=[1, 2, 4, 8],
            value=4,
            help="FRQI's multi-controlled RY gates explode in depth for n > 3. "
            "Down-sampling keeps the demo snappy.",
        )
        shots = st.slider("Shots", 1_000, 200_000, 40_000, step=1_000)
    elif encoding == "NEQR":
        q_qubits = st.slider("Intensity qubits q", 1, 8, 4)
        downsample = st.select_slider(
            "Down-sample factor",
            options=[1, 2, 4],
            value=2,
            help="NEQR uses 2n+q qubits and many multi-CX per pixel.",
        )
        shots = st.slider("Shots", 1_000, 50_000, 8_192, step=1_000)
    else:  # QPIE
        shots = st.slider("Shots", 1_000, 500_000, 100_000, step=1_000)

    show_circuit = st.checkbox(
        "Show circuit diagram (only if small enough)",
        value=False,
    )
    run = st.button("Run", type="primary", use_container_width=True)


# ---------------------------------------------------------- Pre-conditions ----

side = image.shape[0]
n = infer_n_from_image(image)
if n is None or not is_power_of_two(side):
    st.error(f"Image must be square with a power-of-two side; got shape {image.shape}.")
    st.stop()

if downsample > 1:
    if side % downsample != 0:
        st.error(f"Cannot down-sample {side}×{side} by {downsample}× (not divisible).")
        st.stop()
    target = image[::downsample, ::downsample]
else:
    target = image

target_n = int(np.log2(target.shape[0]))
st.markdown(
    f"**Working image:** {target.shape[0]}×{target.shape[0]} "
    f"(n = {target_n}, dtype = {target.dtype}, range [{target.min()}, {target.max()}])"
)


# ------------------------------------------------------------------ Run ----

if not run:
    st.info("Tune the parameters in the sidebar and press **Run**.")
    st.stop()

start = time.perf_counter()
try:
    if encoding == "FRQI":
        input_img = target if target.dtype == np.uint8 else target.astype(np.uint8)
        encoder = FrqiEncoder()
        qc = encoder.encode(input_img)
        counts = ideal_simulation(qc, shots=shots)
        decoded = encoder.decode(counts)[0]
        psnr_max = 255.0
    elif encoding == "NEQR":
        if target.max() >= (1 << q_qubits):
            st.error(
                f"Image max value {target.max()} exceeds 2^q - 1 = {(1 << q_qubits) - 1}. "
                "Increase q or pick a different image."
            )
            st.stop()
        input_img = target.astype(np.int64)
        encoder = NeqrEncoder(q=q_qubits)
        qc = encoder.encode(input_img)
        counts = ideal_simulation(qc, shots=shots)
        decoded = encoder.decode(counts).astype(np.float64)
        psnr_max = float((1 << q_qubits) - 1)
    else:  # QPIE
        input_img = target.astype(np.float64)
        encoder = QpieEncoder()
        qc = encoder.encode(input_img)
        counts = ideal_simulation(qc, shots=shots)
        decoded = encoder.decode(counts)
        psnr_max = float(max(target.max(), 1.0))
except Exception as exc:
    st.error(f"Pipeline error: {exc}")
    st.stop()

elapsed = time.perf_counter() - start


# --------------------------------------------------------------- Results ----

fig = panel_grid_figure(
    [
        ("Input", input_img),
        (f"{encoding} decoded", decoded),
    ],
    cols=2,
)
st.pyplot(fig)

err = float(mse(input_img, decoded))
fidelity = (
    "∞ dB (exact)" if err == 0 else f"{psnr(input_img, decoded, max_intensity=psnr_max):.2f} dB"
)
summary = transpile_summary(qc)
st.markdown("### Metrics")
metrics_dict = {
    "qubits": qc.num_qubits,
    "depth (pre-transpile)": summary["depth"],
    "depth (post-transpile)": summary["depth_transpiled"],
    "shots": shots,
    "runtime (s)": round(elapsed, 3),
    "MSE": round(err, 5),
    "PSNR": fidelity,
}
st.write(metrics_dict)

if encoding == "NEQR":
    st.success(f"NEQR exact recovery: {np.array_equal(decoded.astype(np.int64), input_img)}")

if show_circuit:
    fig_qc = safe_circuit_figure(qc, max_ops=80)
    if fig_qc is None:
        st.info(
            "Circuit has too many operations to render in the browser. "
            "Use `qc.draw('mpl')` in a notebook on a smaller instance."
        )
    else:
        with st.expander("Circuit diagram", expanded=True):
            st.pyplot(fig_qc)


# ------------------------------------------------------------ Save outputs ----

if st.button("Save outputs to data/output/", use_container_width=True):
    out_dir = new_output_dir(prefix=f"encoder_{encoding.lower()}")
    save_named_panels(
        [("input", input_img), (f"{encoding}_decoded", decoded)],
        out_dir,
    )
    fig.savefig(out_dir / "comparison.png", dpi=120, bbox_inches="tight")
    st.success(f"Saved to {out_dir}")
