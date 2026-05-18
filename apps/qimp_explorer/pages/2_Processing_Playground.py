"""Processing Playground — every op `qimp.processing.*` and `qimp.qft` exposes.

Each category covers a coherent slice of the public API; widgets adapt to
the selected operation. NEQR is the workhorse for exact comparison; FRQI
chromatic / QHED / QFT use their natural encodings.
"""

from __future__ import annotations

import sys
from pathlib import Path

_QIMP_APP_ROOT = Path(__file__).resolve().parent.parent
if str(_QIMP_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_QIMP_APP_ROOT))


import numpy as np
import streamlit as st
from _viz import panel_grid_figure
from app_io import (
    infer_n_from_image,
    new_output_dir,
    save_named_panels,
)

from qimp.encoding.frqi import FrqiEncoder
from qimp.encoding.neqr import neqr_circuit, neqr_decode
from qimp.metrics import total_variation
from qimp.processing import chromatic, geometric
from qimp.processing.filters import qhed_decode, qhed_filter, qhed_full_edges
from qimp.qft import apply_inverse_qft, apply_qft
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

n_image = infer_n_from_image(image)
if n_image is None:
    st.error(f"Image must be square 2^n × 2^n; got {image.shape}.")
    st.stop()


# ----------------------------------------------------------------- Sidebar ----

CATEGORIES = ["Geometric", "Chromatic (FRQI)", "Chromatic (NEQR)", "Spectral (QFT)", "Filter"]

OPS_GEOMETRIC = [
    "axis_flip",
    "coord_swap (transpose)",
    "ort_rotation",
    "pos_shift",
    "restr_flip",
    "restr_coord_swap",
]
OPS_CHROMATIC_FRQI = ["color_complement", "color_change (θ)"]
OPS_CHROMATIC_NEQR = ["color_complement", "half_intensity", "classify_complement"]
OPS_SPECTRAL = ["apply_qft", "apply_inverse_qft"]
OPS_FILTER = ["QHED (horizontal only)", "QHED full edges (H + V)"]


with st.sidebar:
    st.header("Operation")
    category = st.radio("Category", CATEGORIES)
    if category == "Geometric":
        op_name = st.selectbox("Operation", OPS_GEOMETRIC)
    elif category == "Chromatic (FRQI)":
        op_name = st.selectbox("Operation", OPS_CHROMATIC_FRQI)
    elif category == "Chromatic (NEQR)":
        op_name = st.selectbox("Operation", OPS_CHROMATIC_NEQR)
    elif category == "Spectral (QFT)":
        op_name = st.selectbox("Operation", OPS_SPECTRAL)
    else:
        op_name = st.selectbox("Operation", OPS_FILTER)

    # Operation-specific parameters
    op_axis: str = "x"
    op_angle: int = 90
    op_direction: str = "+"
    op_magnitude: int = 1
    op_region_bits: str = "1"
    op_theta: float = np.pi / 4
    op_threshold_bit: int = 1
    op_qft_qubits: int = 0  # 0 = all position qubits

    if category == "Geometric":
        if op_name in ("axis_flip", "restr_flip"):
            op_axis = st.radio("Axis", ["x", "y"], horizontal=True)
        if op_name == "ort_rotation":
            op_angle = int(st.radio("Angle (°)", [90, 180, 270], horizontal=True))
        if op_name == "pos_shift":
            op_axis = st.radio("Axis", ["x", "y"], horizontal=True)
            op_direction = st.radio("Direction", ["+", "-"], horizontal=True)
            op_magnitude = st.slider(
                "Magnitude (pixels)",
                min_value=1,
                max_value=max(1, 2**n_image - 1),
                value=1,
            )
        if op_name in ("restr_flip", "restr_coord_swap"):
            max_bits = max(1, n_image)
            op_region_bits = st.text_input(
                "Region prefix (binary, MSB-first)",
                value="1",
                max_chars=max_bits,
                help=f"Up to {max_bits} bits. Selects which half / quadrant / … the op "
                "applies to. Example: '1' = bottom half (rows whose MSB is 1).",
            )

    if category == "Chromatic (FRQI)" and op_name == "color_change (θ)":
        op_theta = st.slider(
            "θ (radians)",
            min_value=-float(np.pi),
            max_value=float(np.pi),
            value=float(np.pi / 4),
            step=0.05,
        )

    if category == "Chromatic (NEQR)" and op_name == "classify_complement":
        op_threshold_bit = st.slider("Threshold bit", 0, 8, 1)

    if category == "Spectral (QFT)":
        op_qft_qubits = st.slider(
            "Qubits (from LSB)",
            min_value=1,
            max_value=2 * n_image,
            value=2 * n_image,
            help="Number of position qubits to apply the QFT on.",
        )

    # Encoding & shots
    q_qubits = 4
    if category in ("Geometric", "Chromatic (NEQR)"):
        q_qubits = st.slider(
            "NEQR intensity qubits q",
            1,
            8,
            4,
            help="Used for the exact-recovery NEQR encoding.",
        )
    downsample = st.select_slider(
        "Down-sample factor",
        options=[1, 2, 4, 8],
        value=1,
        help="Quantum circuits grow with n. Bump this up if a run is too slow.",
    )
    shots = st.slider("Shots", 1_000, 50_000, 8_192, step=1_000)
    run = st.button("Run", type="primary", use_container_width=True)


# ---------------------------------------------------- Prepare working image ----

if downsample > 1 and image.shape[0] % downsample != 0:
    st.error(f"Cannot down-sample {image.shape[0]}×{image.shape[0]} by {downsample}×.")
    st.stop()
target = image[::downsample, ::downsample]
target_n = int(np.log2(target.shape[0]))

info_block = (
    f"**Working image:** {target.shape[0]}×{target.shape[0]} "
    f"(n = {target_n}, dtype = {target.dtype}, range [{target.min()}, {target.max()}])"
)
if downsample > 1:
    st.warning(
        f"⚠️ Image was down-sampled {downsample}× — original was "
        f"{image.shape[0]}×{image.shape[0]}. Set **Down-sample factor** to 1 "
        "in the sidebar to process the full image.\n\n" + info_block
    )
else:
    st.info(info_block)

with st.expander("Preview the image being processed", expanded=False):
    from _viz import to_display_uint8

    st.caption(f"{target.shape[0]}×{target.shape[0]} input")
    st.image(to_display_uint8(target), width=240, clamp=True)


if not run:
    st.info("Pick an operation, set its parameters in the sidebar, and press **Run**.")
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


def _frqi_round_trip(modify_qc) -> np.ndarray:
    """Encode `target` with FRQI, apply `modify_qc(qc, n)`, simulate, decode."""
    encoder = FrqiEncoder()
    img = target if target.dtype == np.uint8 else target.astype(np.uint8)
    qc = encoder.encode(img)
    modify_qc(qc, target_n)
    counts = ideal_simulation(qc, shots=shots * 5)  # FRQI needs more shots for decode
    return encoder.decode(counts)[0]


def _restr_reference(op_name: str, axis: str, region_bits: str, src: np.ndarray) -> np.ndarray:
    """Classical reference for restr_* ops. region_bits prefixes the high-order
    qubits of the orthogonal axis (axis_flip) or of both axes (restr_coord_swap).
    """
    out = src.copy()
    n = int(np.log2(src.shape[0]))
    high_count = len(region_bits)
    # The bits select rows (or cols) whose top |bits| MSBs match `region_bits`.
    # Build a mask over the relevant axis.
    sel_value = int(region_bits, 2)
    block_size = 1 << (n - high_count)  # how many consecutive rows match
    start = sel_value * block_size
    stop = start + block_size

    if op_name == "restr_flip":
        if axis == "y":  # flip columns inside selected rows
            out[start:stop, :] = np.fliplr(src[start:stop, :])
        else:  # axis == "x" → flip rows inside selected columns
            out[:, start:stop] = np.flipud(src[:, start:stop])
    elif op_name == "restr_coord_swap":
        # Transpose the (row, col) block where BOTH row and col are in region.
        out[start:stop, start:stop] = src[start:stop, start:stop].T
    return out


try:
    reference: np.ndarray | None = None

    if category == "Geometric":
        if op_name == "axis_flip":
            processed = _neqr_round_trip(
                lambda qc, n, q, off: geometric.axis_flip(qc, n=n, axis=op_axis, pos_offset=off)
            )
            reference = np.flipud(target) if op_axis == "x" else np.fliplr(target)
        elif op_name == "coord_swap (transpose)":
            processed = _neqr_round_trip(
                lambda qc, n, q, off: geometric.coord_swap(qc, n=n, pos_offset=off)
            )
            reference = target.T
        elif op_name == "ort_rotation":
            processed = _neqr_round_trip(
                lambda qc, n, q, off: geometric.ort_rotation(
                    qc, n=n, angle=op_angle, pos_offset=off
                )
            )
            k = {90: 1, 180: 2, 270: 3}[op_angle]
            reference = np.rot90(target, k)
        elif op_name == "pos_shift":
            processed = _neqr_round_trip(
                lambda qc, n, q, off: geometric.pos_shift(
                    qc,
                    n=n,
                    axis=op_axis,
                    direction=op_direction,
                    magnitude=op_magnitude,
                    pos_offset=off,
                )
            )
            np_axis = 1 if op_axis == "x" else 0
            shift = op_magnitude if op_direction == "+" else -op_magnitude
            reference = np.roll(target, shift=shift, axis=np_axis)
        elif op_name == "restr_flip":
            processed = _neqr_round_trip(
                lambda qc, n, q, off: geometric.restr_flip(
                    qc, n=n, axis=op_axis, region_bits=op_region_bits, pos_offset=off
                )
            )
            reference = _restr_reference(op_name, op_axis, op_region_bits, target)
        else:  # restr_coord_swap
            processed = _neqr_round_trip(
                lambda qc, n, q, off: geometric.restr_coord_swap(
                    qc, n=n, region_bits=op_region_bits, pos_offset=off
                )
            )
            reference = _restr_reference(op_name, op_axis, op_region_bits, target)

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
        else:  # classify_complement
            processed = _neqr_round_trip(
                lambda qc, n, q, off: chromatic.neqr_classify_complement(
                    qc, q=q, threshold_bit=op_threshold_bit
                )
            )
            mask = (1 << op_threshold_bit) - 1
            reference = target.astype(np.int64) ^ mask

    elif category == "Chromatic (FRQI)":
        if op_name == "color_complement":
            processed = _frqi_round_trip(lambda qc, n: chromatic.frqi_color_complement(qc))
            # FRQI color complement: intensity I → 255 - I (approximately, after decode).
            reference = (255 - target.astype(np.int64)).clip(0, 255)
        else:  # color_change (θ)
            processed = _frqi_round_trip(
                lambda qc, n: chromatic.frqi_color_change(qc, theta=op_theta)
            )
            reference = None  # custom unitary, no straightforward classical reference

    elif category == "Spectral (QFT)":
        # QFT-applied on FRQI position qubits: not directly comparable to a classical
        # operation. Show the resulting decoded image (will be very different) plus
        # the circuit depth contribution.
        encoder = FrqiEncoder()
        img = target if target.dtype == np.uint8 else target.astype(np.uint8)
        qc = encoder.encode(img)
        qubits_range = list(range(op_qft_qubits))
        if op_name == "apply_qft":
            apply_qft(qc, qubits_range)
        else:
            apply_inverse_qft(qc, qubits_range)
        counts = ideal_simulation(qc, shots=shots * 5)
        processed = encoder.decode(counts)[0]
        reference = None

    else:  # Filter
        if op_name == "QHED (horizontal only)":
            qc, n_qhed, rms = qhed_filter(target.astype(np.float64))
            counts = ideal_simulation(qc, shots=shots * 5)
            processed = qhed_decode(counts, n=n_qhed, rms=rms)
            reference = None
        else:  # QHED full edges
            qc_h, n_qhed, rms = qhed_filter(target.astype(np.float64))
            qc_v, _, _ = qhed_filter(target.astype(np.float64).T)
            counts_h = ideal_simulation(qc_h, shots=shots * 5)
            counts_v = ideal_simulation(qc_v, shots=shots * 5)
            processed = qhed_full_edges(target.astype(np.float64), counts_h, counts_v)
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
    st.info(
        f"No simple classical reference for **{op_name}**. "
        f"Stats: output sum = {processed.sum():.2f}, max = {processed.max():.2f}, "
        f"TV = {total_variation(processed.astype(np.float64)):.2f} "
        f"(input TV = {total_variation(target.astype(np.float64)):.2f})."
    )

if st.button("Save outputs to data/output/", use_container_width=True):
    out_dir = new_output_dir(prefix=f"processing_{op_name.replace(' ', '_')}")
    save_named_panels(panels, out_dir)
    panel_grid_figure(panels, cols=len(panels)).savefig(
        out_dir / "comparison.png", dpi=120, bbox_inches="tight"
    )
    st.success(f"Saved to {out_dir}")
