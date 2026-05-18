"""Encoder Explorer — FRQI / NEQR / QPIE round-trip on the current image."""

from __future__ import annotations

import sys
from pathlib import Path

_QIMP_APP_ROOT = Path(__file__).resolve().parent.parent
if str(_QIMP_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_QIMP_APP_ROOT))


import time

import numpy as np
import streamlit as st
from app_io import (
    infer_n_from_image,
    is_power_of_two,
    new_output_dir,
    save_named_panels,
)
from _viz import panel_grid_figure, safe_circuit_figure

from qimp.encoding.frqi import FrqiEncoder
from qimp.encoding.neqr import NeqrEncoder
from qimp.encoding.qpie import QpieEncoder
from qimp.metrics import mse, psnr, total_variation, transpile_summary
from qimp.testing import ideal_simulation, noisy_simulation

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

image_n_default = infer_n_from_image(image) or 0
image_side_default = 1 << image_n_default if image_n_default else image.shape[0]

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
            value=1,
            help="FRQI uses a fully-controlled RY for every pixel. At n=4 "
            "(16×16) the transpile is heavy (~minutes). Pick 2× / 4× to "
            "trade fidelity for speed.",
        )
        shots = st.slider("Shots", 1_000, 200_000, 40_000, step=1_000)
    elif encoding == "NEQR":
        q_qubits = st.slider("Intensity qubits q", 1, 8, 4)
        downsample = st.select_slider(
            "Down-sample factor",
            options=[1, 2, 4],
            value=1,
            help="NEQR is exact (any pixel value recovered byte-for-byte), "
            "but each pixel triggers a multi-CX gate per intensity bit.",
        )
        shots = st.slider("Shots", 1_000, 50_000, 8_192, step=1_000)
    else:  # QPIE
        shots = st.slider("Shots", 1_000, 500_000, 100_000, step=1_000)

    show_circuit = st.checkbox(
        "Show circuit diagram (only if small enough)",
        value=False,
    )

    st.divider()
    st.subheader("Backend")
    backend_mode = st.radio(
        "Simulation mode",
        ["Ideal", "Noisy (depolarizing)"],
        help="Ideal: noise-free Aer. Noisy: a simple uniform depolarizing channel "
        "applied to every gate, useful for sensitivity studies.",
    )
    noise_p = 0.0
    if backend_mode.startswith("Noisy"):
        noise_p = st.slider(
            "Per-gate depolarizing probability",
            min_value=0.0,
            max_value=0.05,
            value=0.001,
            step=0.0005,
            format="%.4f",
            help="Applied as `depolarizing_error(p, k)` on every k-qubit gate.",
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

# Estimated circuit complexity for the chosen encoding + size.
if encoding == "FRQI":
    qubits_estimate = 2 * target_n + 1
    cost_note = (
        f"Each of the {1 << (2 * target_n)} pixels uses a {2 * target_n}-controlled RY. "
        "Expect minutes of transpile/run time at n=4."
        if target_n >= 4
        else "Fast at this size."
    )
elif encoding == "NEQR":
    qubits_estimate = 2 * target_n + q_qubits
    cost_note = (
        f"{1 << (2 * target_n)} pixels × up to {q_qubits} multi-CX each. Slow at n ≥ 4."
        if target_n >= 4
        else "Fast at this size."
    )
else:  # QPIE
    qubits_estimate = 2 * target_n
    cost_note = "QPIE is a single state-preparation; runtime is dominated by `initialize`."

complexity_msg = (
    f"**Working image:** {target.shape[0]}×{target.shape[0]} "
    f"(n = {target_n}, dtype = {target.dtype}, range [{target.min()}, {target.max()}])  \n"
    f"**Estimated circuit:** {qubits_estimate} qubits. {cost_note}"
)
if downsample > 1:
    st.warning(
        f"⚠️ Image was down-sampled {downsample}× — original was "
        f"{image.shape[0]}×{image.shape[0]} (n = {image_n_default}). "
        "Set the **Down-sample factor** to 1 in the sidebar to encode the full image.\n\n"
        + complexity_msg
    )
else:
    st.info(complexity_msg)

# Show the actual working image so the user *sees* what's being processed.
with st.expander("Preview the image being encoded", expanded=False):
    from _viz import image_figure

    st.pyplot(image_figure(target, title=f"{target.shape[0]}×{target.shape[0]} input"))


# ------------------------------------------------------------------ Run ----

if not run:
    st.info("Tune the parameters in the sidebar and press **Run**.")
    st.stop()


def _build_noise_model(p: float) -> object | None:
    """Uniform depolarizing channel on every {1, 2, 3}-qubit gate."""
    if p <= 0:
        return None
    try:
        from qiskit_aer.noise import NoiseModel, depolarizing_error
    except ImportError:
        return None
    nm = NoiseModel()
    for k, gate_names in (
        (1, ["u1", "u2", "u3", "rx", "ry", "rz", "sx", "h", "x", "y", "z"]),
        (2, ["cx", "cz", "swap"]),
        (3, ["ccx"]),
    ):
        err = depolarizing_error(p, k)
        for name in gate_names:
            nm.add_all_qubit_quantum_error(err, name)
    return nm


def _simulate(qc, shots: int):
    if backend_mode.startswith("Noisy"):
        return noisy_simulation(qc, shots=shots, noise_model=_build_noise_model(noise_p))
    return ideal_simulation(qc, shots=shots)


start = time.perf_counter()
try:
    if encoding == "FRQI":
        input_img = target if target.dtype == np.uint8 else target.astype(np.uint8)
        encoder = FrqiEncoder()
        qc = encoder.encode(input_img)
        counts = _simulate(qc, shots=shots)
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
        counts = _simulate(qc, shots=shots)
        decoded = encoder.decode(counts).astype(np.float64)
        psnr_max = float((1 << q_qubits) - 1)
    else:  # QPIE
        input_img = target.astype(np.float64)
        encoder = QpieEncoder()
        qc = encoder.encode(input_img)
        counts = _simulate(qc, shots=shots)
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
tv_input = total_variation(input_img.astype(np.float64))
tv_decoded = total_variation(decoded.astype(np.float64))

st.markdown("### Metrics")
col_metrics, col_ops = st.columns(2)
with col_metrics:
    metrics_dict = {
        "encoding": encoding,
        "backend": backend_mode + (f" (p={noise_p:.4f})" if noise_p > 0 else ""),
        "qubits": qc.num_qubits,
        "shots": shots,
        "runtime (s)": round(elapsed, 3),
        "depth (pre-transpile)": summary["depth"],
        "depth (post-transpile)": summary["depth_transpiled"],
        "MSE": round(err, 5),
        "PSNR": fidelity,
        "TV (input)": round(tv_input, 3),
        "TV (decoded)": round(tv_decoded, 3),
    }
    st.write(metrics_dict)
with col_ops:
    st.markdown("**Gate counts (after transpile)**")
    st.write(dict(summary["ops_transpiled"]))

if encoding == "NEQR":
    st.success(f"NEQR exact recovery: {np.array_equal(decoded.astype(np.int64), input_img)}")

if encoding == "FRQI":
    with st.expander("FRQI intensity → angle mapping (for the encoded image)", expanded=False):
        from qimp.encoding.frqi import intensities_to_angles

        norm = float(encoder.normalization)
        angles = intensities_to_angles(input_img.astype(np.float64), normalization=norm)
        st.write(
            {
                "normalization": norm,
                "min(angle)": round(float(angles.min()), 4),
                "max(angle)": round(float(angles.max()), 4),
                "angle at intensity=0": 0.0,
                "angle at intensity=norm": round(float(np.pi), 4),
            }
        )
        from _viz import image_figure as _image_figure

        st.pyplot(_image_figure(angles, title="θ map (rad)"))

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
