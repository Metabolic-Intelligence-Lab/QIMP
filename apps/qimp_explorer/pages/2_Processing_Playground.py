"""Processing Playground — geometric / chromatic ops + QHED edge detection.

NEQR is the workhorse for the playground: exact retrieval lets us compare
the quantum result with the numpy ground truth byte-for-byte.
"""

from __future__ import annotations

import numpy as np
import streamlit as st
from apps.qimp_explorer._io import (
    infer_n_from_image,
    new_output_dir,
    save_named_panels,
)
from apps.qimp_explorer._viz import panel_grid_figure

from qimp.encoding.neqr import neqr_circuit, neqr_decode
from qimp.processing import chromatic, geometric
from qimp.processing.filters import qhed_decode, qhed_filter
from qimp.testing import ideal_simulation

st.set_page_config(page_title="Processing Playground", page_icon="🛠️", layout="wide")
st.title("🛠️ Processing Playground")

image = st.session_state.get("image")
if image is None:
    st.warning("Load an image from the **Home** page first.")
    st.stop()

if image.ndim != 2:
    st.error("Use a 2D grayscale image. For RGB, head to the **GP-ratio** page.")
    st.stop()

n = infer_n_from_image(image)
if n is None:
    st.error(f"Image must be square 2^n × 2^n; got {image.shape}.")
    st.stop()


# ----------------------------------------------------------------- Sidebar ----

OPS_GEOMETRIC = [
    "axis_flip (X)",
    "axis_flip (Y)",
    "coord_swap (transpose)",
    "ort_rotation 90°",
    "ort_rotation 180°",
    "ort_rotation 270°",
    "pos_shift +X",
    "pos_shift +Y",
]
OPS_CHROMATIC = ["color_complement", "half_intensity", "classify_complement"]
OPS_FILTER = ["QHED edge detection"]

with st.sidebar:
    st.header("Operation")
    category = st.radio("Category", ["Geometric", "Chromatic (NEQR)", "Filter (QHED)"])
    if category == "Geometric":
        op_name = st.selectbox("Operation", OPS_GEOMETRIC)
    elif category == "Chromatic (NEQR)":
        op_name = st.selectbox("Operation", OPS_CHROMATIC)
    else:
        op_name = st.selectbox("Operation", OPS_FILTER)

    q_qubits = st.slider(
        "NEQR intensity qubits q",
        1,
        8,
        4,
        help="Used by geometric and chromatic ops to encode the image exactly.",
    )
    downsample = st.select_slider(
        "Down-sample factor (NEQR circuits are expensive at n ≥ 3)",
        options=[1, 2, 4, 8],
        value=4,
    )
    shots = st.slider("Shots", 1_000, 50_000, 8_192, step=1_000)
    run = st.button("Run", type="primary", use_container_width=True)


# ---------------------------------------------------- Prepare working image ----

if downsample > 1 and image.shape[0] % downsample != 0:
    st.error(f"Cannot down-sample {image.shape[0]}×{image.shape[0]} by {downsample}×.")
    st.stop()
target = image[::downsample, ::downsample]
target_n = int(np.log2(target.shape[0]))
st.markdown(
    f"**Working image:** {target.shape[0]}×{target.shape[0]} "
    f"(n = {target_n}, dtype = {target.dtype}, range [{target.min()}, {target.max()}])"
)


if not run:
    st.info("Pick an operation and press **Run**.")
    st.stop()


# ------------------------------------------------------------------ Execute ----


def _neqr_round_trip(modify_qc) -> np.ndarray:
    """Encode `target` with NEQR, apply `modify_qc(qc, n, q, pos_offset)`, decode."""
    if target.max() >= (1 << q_qubits):
        st.error(f"Image max {target.max()} exceeds 2^q - 1 = {(1 << q_qubits) - 1}. Increase q.")
        st.stop()
    img_int = target.astype(np.int64)
    qc = neqr_circuit(img_int, q=q_qubits)
    modify_qc(qc, target_n, q_qubits, q_qubits)
    counts = ideal_simulation(qc, shots=shots)
    return neqr_decode(counts, n=target_n, q=q_qubits)


try:
    if category == "Geometric":
        if op_name == "axis_flip (X)":
            processed = _neqr_round_trip(
                lambda qc, n, q, off: geometric.axis_flip(qc, n=n, axis="x", pos_offset=off)
            )
            reference = np.flipud(target)
        elif op_name == "axis_flip (Y)":
            processed = _neqr_round_trip(
                lambda qc, n, q, off: geometric.axis_flip(qc, n=n, axis="y", pos_offset=off)
            )
            reference = np.fliplr(target)
        elif op_name == "coord_swap (transpose)":
            processed = _neqr_round_trip(
                lambda qc, n, q, off: geometric.coord_swap(qc, n=n, pos_offset=off)
            )
            reference = target.T
        elif op_name.startswith("ort_rotation"):
            angle = int(op_name.split()[-1].rstrip("°"))
            processed = _neqr_round_trip(
                lambda qc, n, q, off: geometric.ort_rotation(qc, n=n, angle=angle, pos_offset=off)
            )
            # NumPy rot90: k counter-clockwise quarter-turns.
            k = {90: 1, 180: 2, 270: 3}[angle]
            reference = np.rot90(target, k)
        else:  # pos_shift
            axis = "x" if op_name == "pos_shift +X" else "y"
            processed = _neqr_round_trip(
                lambda qc, n, q, off: geometric.pos_shift(
                    qc, n=n, axis=axis, direction="+", magnitude=1, pos_offset=off
                )
            )
            np_axis = 1 if axis == "x" else 0
            reference = np.roll(target, shift=1, axis=np_axis)

    elif category == "Chromatic (NEQR)":
        max_val = (1 << q_qubits) - 1
        if op_name == "color_complement":
            processed = _neqr_round_trip(
                lambda qc, n, q, off: chromatic.neqr_color_complement(qc, q=q)
            )
            reference = max_val - target.astype(np.int64)
        elif op_name == "half_intensity":
            if q_qubits < 2:
                st.error("half_intensity requires q ≥ 2.")
                st.stop()
            processed = _neqr_round_trip(
                lambda qc, n, q, off: chromatic.neqr_half_intensity(qc, q=q)
            )
            reference = target.astype(np.int64) >> 1
        else:  # classify_complement (mid-bit)
            threshold = q_qubits // 2
            processed = _neqr_round_trip(
                lambda qc, n, q, off: chromatic.neqr_classify_complement(
                    qc, q=q, threshold_bit=threshold
                )
            )
            mask = (1 << threshold) - 1
            reference = target.astype(np.int64) ^ mask

    else:  # QHED
        qc, n_qhed, rms = qhed_filter(target.astype(np.float64))
        counts = ideal_simulation(qc, shots=shots * 5)  # QHED benefits from more shots
        processed = qhed_decode(counts, n=n_qhed, rms=rms)
        # No exact classical reference for the quantum-flattened gradient; show TV.
        reference = None

except Exception as exc:
    st.error(f"Pipeline error: {exc}")
    st.stop()


# ------------------------------------------------------------------ Display ----

panels: list[tuple[str, np.ndarray]] = [
    ("Original", target),
    (f"After {op_name}", processed),
]
if reference is not None:
    panels.append(("NumPy reference", reference))

st.pyplot(panel_grid_figure(panels, cols=len(panels)))

if reference is not None:
    matches = np.array_equal(processed.astype(reference.dtype), reference)
    if matches:
        st.success(f"✓ Quantum output matches numpy reference exactly ({op_name}).")
    else:
        diff = np.abs(processed.astype(np.float64) - reference.astype(np.float64))
        st.warning(
            f"Quantum output differs from numpy reference. "
            f"Max abs difference: {diff.max():.2f}, mean: {diff.mean():.3f}."
        )
else:
    from qimp.metrics import total_variation

    st.info(
        f"QHED gradient sum: {processed.sum():.2f} · "
        f"classical TV: {total_variation(target.astype(np.float64)):.2f} · "
        f"max gradient pixel: {processed.max():.2f}"
    )

if st.button("Save outputs to data/output/", use_container_width=True):
    out_dir = new_output_dir(prefix=f"processing_{op_name.replace(' ', '_')}")
    save_named_panels(panels, out_dir)
    panel_grid_figure(panels, cols=len(panels)).savefig(
        out_dir / "comparison.png", dpi=120, bbox_inches="tight"
    )
    st.success(f"Saved to {out_dir}")
