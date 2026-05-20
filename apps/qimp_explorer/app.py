"""QIMP Explorer — sequential workflow wizard.

Walk an image through every step of the QIMP pipeline:

  1. Load image          (upload or pick from data/immagini/)
  2. Preprocess          (resize to 2^n × 2^n, optional grayscale)
  3. Encode              (FRQI / NEQR / QPIE / MCRQI / NCQI)
  4. Process (optional)  (geometric / chromatic / QFT / QHED)
  5. Execute & export    (ideal / noisy / IBM hardware → save outputs)

Each step unlocks only when its prerequisites are met. Click "Reset" at the
top to start over. The sidebar exposes three add-on pages (Benchmark,
GP-ratio, System Info) for workflows that don't fit a single-image flow.
"""

from __future__ import annotations

import sys
from pathlib import Path

_QIMP_APP_ROOT = Path(__file__).resolve().parent
if str(_QIMP_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_QIMP_APP_ROOT))


import io
import time

import numpy as np
import streamlit as st
from PIL import Image

from _cached import cached_discover_dataset_images
from _ibm import (
    circuit_to_qasm3,
    have_ibm_runtime,
    list_ibm_backends,
    run_on_ibm,
    sanitize_exception,
)
from _viz import to_display_uint8, upscale_for_display
from app_io import (
    DATASET_GRAYSCALE,
    DATASET_RGB,
    RESIZE_OPTIONS,
    load_image,
    new_output_dir,
    resize_to_square,
    save_named_panels,
    to_grayscale,
)
from qimp.encoding.frqi import FrqiEncoder
from qimp.encoding.mcrqi import McrqiEncoder
from qimp.encoding.ncqi import NcqiEncoder
from qimp.encoding.neqr import NeqrEncoder
from qimp.encoding.qpie import QpieEncoder
from qimp.processing import chromatic, geometric
from qimp.qft import apply_inverse_qft, apply_qft
from qimp.testing import ideal_simulation

# Encoding registry: drives step 3 (encode), the reset helper, and any
# step-4 dispatch that depends on the encoder type. Each entry is
# ``(EncoderClass, dtype_cast, needs_q_param)``.
_DTYPE_FOR_FRQI = lambda img: img.astype(np.uint8) if img.dtype != np.uint8 else img  # noqa: E731
_DTYPE_INT64 = lambda img: img.astype(np.int64)  # noqa: E731
_DTYPE_FLOAT64 = lambda img: img.astype(np.float64)  # noqa: E731

_ENCODER_SPECS = {
    "FRQI": (FrqiEncoder, _DTYPE_FOR_FRQI, False),
    "NEQR": (NeqrEncoder, _DTYPE_INT64, True),
    "QPIE": (QpieEncoder, _DTYPE_FLOAT64, False),
    "MCRQI": (McrqiEncoder, _DTYPE_FOR_FRQI, False),
    "NCQI": (NcqiEncoder, _DTYPE_INT64, True),
}

st.set_page_config(page_title="QIMP Explorer", page_icon="🔬", layout="wide")
st.title("🔬 QIMP Explorer — Workflow")
st.caption(
    "Sequential wizard for **qimp-mi**. Each step activates as soon as the "
    "previous one is complete. The sidebar pages cover three add-ons that "
    "don't fit a single-image flow: a side-by-side **Benchmark** of all encoders, "
    "the **GP-ratio** microscopy pipeline (RGB channels + filters), and "
    "**System Info** (versions, GPU detection, cache controls)."
)


# ---------------------------------------------------------- State helpers ----


def _reset_all() -> None:
    """Clear every wizard-owned session_state entry.

    Uses a ``wiz_`` prefix sweep so newly-added keys (op chain, op preview,
    cached op-param widget values, etc.) are reset automatically without
    having to be enumerated here.
    """
    for key in [k for k in st.session_state if k.startswith("wiz_")]:
        st.session_state.pop(key, None)


def _step_done(name: str) -> bool:
    mapping = {
        "load": "wiz_image_raw",
        "preprocess": "wiz_image",
        "encode": "wiz_circuit",
        "execute": "wiz_counts",
    }
    return mapping[name] in st.session_state


# ------------------------------------------------------------ Header bar ----

bar_left, bar_right = st.columns([4, 1])
with bar_left:
    progress: list[str] = []
    for label, name in [
        ("Load", "load"),
        ("Preprocess", "preprocess"),
        ("Encode", "encode"),
        ("Execute", "execute"),
    ]:
        mark = "✅" if _step_done(name) else "⬜"
        progress.append(f"{mark} {label}")
    st.markdown(" → ".join(progress))
with bar_right:
    if st.button("🔄 Reset workflow", use_container_width=True):
        _reset_all()
        st.rerun()

st.divider()


# =========================================================== STEP 1: LOAD ====

with st.container(border=True):
    st.subheader("1️⃣ Load image")

    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        upload = st.file_uploader(
            "Upload TIFF / PNG",
            type=["tif", "tiff", "png"],
            key="wiz_upload",
        )
    with col_b:
        source_kind = st.radio(
            "Or pick from dataset",
            ["Grayscale (Train_QML_16)", "RGB (trainQML)"],
            horizontal=True,
            key="wiz_source_kind",
        )
        dataset_dir = DATASET_GRAYSCALE if source_kind.startswith("Grayscale") else DATASET_RGB
        dataset_paths = [
            Path(p) for p in cached_discover_dataset_images(str(dataset_dir), max_items=50)
        ]
        dataset_choice: Path | None = None
        if dataset_paths:
            labels = [p.name for p in dataset_paths]
            i = st.selectbox(
                "Image",
                options=list(range(len(labels))),
                format_func=lambda i: labels[i],
                key="wiz_dataset_idx",
            )
            dataset_choice = dataset_paths[i]
        else:
            st.caption(f"No images found in {dataset_dir}.")
    with col_c:
        st.markdown("&nbsp;")
        if st.button("Load image", type="primary", use_container_width=True, key="wiz_load_btn"):
            try:
                if upload is not None:
                    arr = np.asarray(Image.open(io.BytesIO(upload.read())))
                    name = upload.name
                elif dataset_choice is not None:
                    arr = load_image(dataset_choice)
                    name = dataset_choice.name
                else:
                    st.error("Pick a source first.")
                    st.stop()
                # Reset everything downstream when (re)loading.
                _reset_all()
                st.session_state["wiz_image_raw"] = arr
                st.session_state["wiz_image_source"] = name
                st.success(f"Loaded {name}")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to load: {exc}")

    raw = st.session_state.get("wiz_image_raw")
    if raw is not None:
        preview = upscale_for_display(to_display_uint8(raw))
        st.image(preview, width=240, clamp=True)
        st.caption(
            f"`{st.session_state.get('wiz_image_source', '?')}` · shape "
            f"{tuple(raw.shape)} · dtype {raw.dtype} · max {raw.max()}"
        )


# ======================================================= STEP 2: PREPROCESS ==

with st.container(border=True):
    st.subheader("2️⃣ Preprocess (resize + optional grayscale)")
    if not _step_done("load"):
        st.info("Load an image first.")
    else:
        raw = st.session_state["wiz_image_raw"]
        current_side = (
            raw.shape[0] if raw.ndim >= 2 and raw.shape[0] == raw.shape[1] else min(raw.shape[:2])
        )
        default_target = max(
            (s for s in RESIZE_OPTIONS if s <= current_side), default=RESIZE_OPTIONS[0]
        )

        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            target_side = st.select_slider(
                "Target side (pixels)",
                options=list(RESIZE_OPTIONS),
                value=default_target,
                key="wiz_resize_target",
            )
        with col2:
            to_gray = st.checkbox(
                "Convert RGB to grayscale (luma BT.601)",
                value=raw.ndim == 3,
                key="wiz_to_gray",
            )
        with col3:
            st.markdown("&nbsp;")
            if st.button("Apply", type="primary", use_container_width=True, key="wiz_prep_btn"):
                try:
                    img = raw
                    if to_gray and img.ndim == 3:
                        img = to_grayscale(img)
                    img = resize_to_square(img, int(target_side))
                    st.session_state["wiz_image"] = img
                    st.session_state["wiz_n"] = int(np.log2(int(target_side)))
                    # Reset downstream.
                    for k in (
                        "wiz_encoding",
                        "wiz_encoder",
                        "wiz_circuit",
                        "wiz_counts",
                        "wiz_decoded",
                        "wiz_run_id",
                        "wiz_ibm_job",
                    ):
                        st.session_state.pop(k, None)
                    st.success(f"Preprocessed to {target_side}×{target_side}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Preprocess failed: {exc}")

        img = st.session_state.get("wiz_image")
        if img is not None:
            st.image(upscale_for_display(to_display_uint8(img)), width=320, clamp=True)
            n = st.session_state["wiz_n"]
            st.caption(
                f"Working image: {img.shape[0]}×{img.shape[0]} "
                f"({'RGB' if img.ndim == 3 else 'grayscale'}), n = {n}, "
                f"dtype {img.dtype}, range [{img.min()}, {img.max()}]"
            )


# ============================================================ STEP 3: ENCODE ==

with st.container(border=True):
    st.subheader("3️⃣ Encode")
    if not _step_done("preprocess"):
        st.info("Complete preprocessing first.")
    else:
        img = st.session_state["wiz_image"]
        n = st.session_state["wiz_n"]
        is_rgb = img.ndim == 3

        encoding_options = ["MCRQI", "NCQI"] if is_rgb else ["FRQI", "NEQR", "QPIE"]

        col_e1, col_e2, col_e3 = st.columns([2, 2, 1])
        with col_e1:
            encoding = st.radio(
                "Method", encoding_options, horizontal=True, key="wiz_encoding_choice"
            )
        with col_e2:
            q_qubits = st.slider(
                "Intensity qubits q (NEQR / NCQI only)",
                1,
                8,
                4,
                disabled=encoding not in ("NEQR", "NCQI"),
                key="wiz_q",
            )
        with col_e3:
            st.markdown("&nbsp;")
            if st.button("Encode", type="primary", use_container_width=True, key="wiz_encode_btn"):
                try:
                    enc_cls, cast, needs_q = _ENCODER_SPECS[encoding]
                    if needs_q and img.max() >= (1 << q_qubits):
                        st.error(
                            f"Image max {img.max()} exceeds 2^q-1 = {(1 << q_qubits) - 1}. "
                            "Increase q."
                        )
                        st.stop()
                    encoder = enc_cls(q=q_qubits) if needs_q else enc_cls()
                    qc = encoder.encode(cast(img))

                    st.session_state["wiz_encoding"] = encoding
                    st.session_state["wiz_encoder"] = encoder
                    st.session_state["wiz_circuit"] = qc
                    # Reset downstream.
                    for k in ("wiz_counts", "wiz_decoded", "wiz_run_id", "wiz_ibm_job"):
                        st.session_state.pop(k, None)
                    st.success(f"Encoded into {qc.num_qubits} qubits")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Encoding failed: {exc}")

        qc = st.session_state.get("wiz_circuit")
        if qc is not None:
            from qimp.metrics import transpile_summary

            summary = transpile_summary(qc)
            st.write(
                {
                    "encoding": st.session_state["wiz_encoding"],
                    "qubits": qc.num_qubits,
                    "depth (pre-transpile)": summary["depth"],
                    "depth (post-transpile)": summary["depth_transpiled"],
                }
            )


# ============================================== STEP 4: PROCESS (optional) ==
#
# The process step exposes the full library catalog grouped by category:
#   - Geometric  : flip, swap, rotation, shift, restricted variants
#   - Chromatic  : intensity / color manipulations (encoding-specific)
#   - Frequency  : forward / inverse QFT on the position or intensity register
#
# Each op may take extra parameters; they're rendered conditionally based on
# the op's `params` schema. Ops can be chained — every Apply mutates the
# current circuit in-place, then re-runs an ideal simulation to refresh the
# before/after preview.


def _unwrap_decoded(decoded, encoding: str):
    """Normalize an encoder's decode() output to a single 2-D / 3-D image array.

    The library's ``FrqiEncoder.decode`` now returns ``np.ndarray`` for the
    single-image case (``m == 0``) and ``list[np.ndarray]`` for multi-image
    encodings — matching the NEQR / QPIE contract. This helper stays as a
    defensive shim against older return shapes and future encoder additions
    that might still emit a batch axis.
    """
    if isinstance(decoded, (list, tuple)):
        return decoded[0]
    arr = np.asarray(decoded)
    if encoding in ("FRQI", "NEQR", "QPIE") and arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    return decoded


def _pos_offset(encoding: str, encoder) -> int:
    """Return the qubit index where the 2n-qubit position register starts."""
    if encoding == "NEQR":
        return encoder.q
    if encoding == "NCQI":
        return 3 * encoder.q
    return 0


def _build_op_catalog(encoding: str, n: int) -> dict[str, list[dict]]:
    """Return ``{category: [op_spec, ...]}`` for the current encoding.

    Each op_spec is a dict with keys: ``key`` (unique id),
    ``label`` (display name), ``params`` (list of param specs).
    A param spec is ``{name, kind, ...}`` where ``kind`` ∈
    ``int_slider | choice | text | float_slider``.
    """
    # An R/G/B channel picker, included only for the RGB encodings (MCRQI / NCQI).
    channel_param = [
        {"name": "channel", "kind": "choice", "choices": ["R", "G", "B"], "default": "R"}
    ]
    rgb_channel_for = lambda e, kinds: channel_param if e in kinds else []  # noqa: E731

    cat: dict[str, list[dict]] = {"Geometric": [], "Chromatic": [], "Frequency": []}

    # ---- Geometric (apply to any encoding that has a position register) ----
    cat["Geometric"].extend(
        [
            {"key": "axis_flip_x", "label": "axis_flip — X (mirror columns)", "params": []},
            {"key": "axis_flip_y", "label": "axis_flip — Y (mirror rows)", "params": []},
            {"key": "coord_swap", "label": "coord_swap (transpose)", "params": []},
            {"key": "rot_90", "label": "ort_rotation 90°", "params": []},
            {"key": "rot_180", "label": "ort_rotation 180°", "params": []},
            {"key": "rot_270", "label": "ort_rotation 270°", "params": []},
            {
                "key": "pos_shift",
                "label": "pos_shift (cyclic translation)",
                "params": [
                    {"name": "axis", "kind": "choice", "choices": ["x", "y"], "default": "x"},
                    {
                        "name": "direction",
                        "kind": "choice",
                        "choices": ["+", "-"],
                        "default": "+",
                    },
                    {
                        "name": "magnitude",
                        "kind": "int_slider",
                        "min": 1,
                        "max": max(1, (1 << n) - 1),
                        "default": 1,
                    },
                ],
            },
            {
                "key": "restr_flip",
                "label": "restr_flip (flip a region)",
                "params": [
                    {"name": "axis", "kind": "choice", "choices": ["x", "y"], "default": "x"},
                    {
                        "name": "region_bits",
                        "kind": "text",
                        "default": "1",
                        "help": (
                            f"Up to {n} chars of 0/1 selecting the orthogonal-axis prefix "
                            "(MSB-first)."
                        ),
                    },
                ],
            },
            {
                "key": "restr_coord_swap",
                "label": "restr_coord_swap (transpose a region)",
                "params": [
                    {
                        "name": "region_bits",
                        "kind": "text",
                        "default": "1",
                        "help": f"Up to {n} chars of 0/1 (MSB-first).",
                    }
                ],
            },
        ]
    )

    # ---- Chromatic (encoding-specific) -------------------------------------
    if encoding in ("FRQI", "MCRQI"):
        cat["Chromatic"].append(
            {
                "key": "frqi_color_complement",
                "label": "frqi_color_complement (θ → π − θ)",
                "params": rgb_channel_for(encoding, {"MCRQI"}),
            }
        )
        cat["Chromatic"].append(
            {
                "key": "frqi_color_change",
                "label": "frqi_color_change (shift all intensities by θ)",
                "params": [
                    {
                        "name": "theta",
                        "kind": "float_slider",
                        "min": -3.1416,
                        "max": 3.1416,
                        "step": 0.05,
                        "default": 0.5,
                    },
                    *rgb_channel_for(encoding, {"MCRQI"}),
                ],
            }
        )
    if encoding in ("NEQR", "NCQI"):
        ch = rgb_channel_for(encoding, {"NCQI"})
        cat["Chromatic"].append(
            {
                "key": "neqr_color_complement",
                "label": "neqr_color_complement (i → 2^q − 1 − i)",
                "params": ch,
            }
        )
        cat["Chromatic"].append(
            {
                "key": "neqr_half_intensity",
                "label": "neqr_half_intensity (i → i // 2)",
                "params": ch,
            }
        )
        q_default = max(1, encoder_q_for(encoding))
        cat["Chromatic"].append(
            {
                "key": "neqr_classify_complement",
                "label": "neqr_classify_complement (threshold flip)",
                "params": [
                    {
                        "name": "threshold_bit",
                        "kind": "int_slider",
                        "min": 0,
                        "max": q_default,
                        "default": max(0, q_default - 1),
                    },
                    *ch,
                ],
            }
        )

    # ---- Frequency (QFT) ---------------------------------------------------
    cat["Frequency"].extend(
        [
            {"key": "qft_pos", "label": "QFT on position register (2n qubits)", "params": []},
            {
                "key": "iqft_pos",
                "label": "inverse QFT on position register (2n qubits)",
                "params": [],
            },
        ]
    )
    if encoding in ("NEQR", "NCQI"):
        ch = rgb_channel_for(encoding, {"NCQI"})
        cat["Frequency"].extend(
            [
                {
                    "key": "qft_intensity",
                    "label": "QFT on intensity register (q qubits)",
                    "params": ch,
                },
                {
                    "key": "iqft_intensity",
                    "label": "inverse QFT on intensity register (q qubits)",
                    "params": ch,
                },
            ]
        )

    return cat


def encoder_q_for(encoding: str) -> int:
    """Return the q parameter of the current encoder (intensity-qubits), or 0."""
    enc = st.session_state.get("wiz_encoder")
    return getattr(enc, "q", 0) if enc is not None else 0


def _channel_color_qubit(encoder, channel: str) -> int:
    """Return the FRQI/MCRQI color qubit for a given channel (R/G/B)."""
    # MCRQI layout: color qubits at indices 2n, 2n+1, 2n+2 (= R, G, B).
    n = encoder.n
    idx_map = {"R": 2 * n, "G": 2 * n + 1, "B": 2 * n + 2}
    return idx_map[channel]


def _ncqi_intensity_range(encoder, channel: str) -> list[int]:
    """Return the q intensity-qubit indices for one NCQI channel (R/G/B)."""
    # NCQI layout: q_R at [0..q-1], q_G at [q..2q-1], q_B at [2q..3q-1].
    q = encoder.q
    offset = {"R": 0, "G": q, "B": 2 * q}[channel]
    return list(range(offset, offset + q))


def _ncqi_complement(qc, encoder, channel: str) -> None:
    for qb in _ncqi_intensity_range(encoder, channel):
        qc.x(qb)


def _ncqi_half_intensity(qc, encoder, channel: str) -> None:
    qs = _ncqi_intensity_range(encoder, channel)
    for i in range(len(qs) - 1):
        qc.swap(qs[i], qs[i + 1])
    qc.reset(qs[-1])


def _ncqi_classify(qc, encoder, channel: str, threshold_bit: int) -> None:
    for qb in _ncqi_intensity_range(encoder, channel)[: int(threshold_bit)]:
        qc.x(qb)


def _intensity_qubits(encoder, encoding: str, args: dict) -> list[int]:
    if encoding == "NCQI":
        return _ncqi_intensity_range(encoder, args["channel"])
    return list(range(encoder.q))


def _apply_op_to_qc(qc, spec: dict, args: dict, n: int, encoding: str, encoder) -> None:
    """Dispatch a single op application (in-place) via the ``_OP_DISPATCH`` table."""
    pos_off = _pos_offset(encoding, encoder)
    pos_qubits = list(range(pos_off, pos_off + 2 * n))
    handler = _OP_DISPATCH.get(spec["key"])
    if handler is None:  # pragma: no cover — guarded by the UI
        raise ValueError(f"unknown op key: {spec['key']}")
    handler(
        qc,
        args=args,
        n=n,
        pos_off=pos_off,
        pos_qubits=pos_qubits,
        encoding=encoding,
        encoder=encoder,
    )


# Each handler takes ``(qc, *, args, n, pos_off, pos_qubits, encoding, encoder)``.
# Using keyword-only so the table stays self-documenting and adding a new op
# means adding one row, not a new elif branch.
_OP_DISPATCH = {
    "axis_flip_x": lambda qc, **kw: geometric.axis_flip(
        qc, n=kw["n"], axis="x", pos_offset=kw["pos_off"]
    ),
    "axis_flip_y": lambda qc, **kw: geometric.axis_flip(
        qc, n=kw["n"], axis="y", pos_offset=kw["pos_off"]
    ),
    "coord_swap": lambda qc, **kw: geometric.coord_swap(qc, n=kw["n"], pos_offset=kw["pos_off"]),
    "rot_90": lambda qc, **kw: geometric.ort_rotation(
        qc, n=kw["n"], angle=90, pos_offset=kw["pos_off"]
    ),
    "rot_180": lambda qc, **kw: geometric.ort_rotation(
        qc, n=kw["n"], angle=180, pos_offset=kw["pos_off"]
    ),
    "rot_270": lambda qc, **kw: geometric.ort_rotation(
        qc, n=kw["n"], angle=270, pos_offset=kw["pos_off"]
    ),
    "pos_shift": lambda qc, **kw: geometric.pos_shift(
        qc,
        n=kw["n"],
        axis=kw["args"]["axis"],
        direction=kw["args"]["direction"],
        magnitude=int(kw["args"]["magnitude"]),
        pos_offset=kw["pos_off"],
    ),
    "restr_flip": lambda qc, **kw: geometric.restr_flip(
        qc,
        n=kw["n"],
        axis=kw["args"]["axis"],
        region_bits=kw["args"]["region_bits"],
        pos_offset=kw["pos_off"],
    ),
    "restr_coord_swap": lambda qc, **kw: geometric.restr_coord_swap(
        qc, n=kw["n"], region_bits=kw["args"]["region_bits"], pos_offset=kw["pos_off"]
    ),
    "frqi_color_complement": lambda qc, **kw: chromatic.frqi_color_complement(
        qc,
        color_qubit=(
            _channel_color_qubit(kw["encoder"], kw["args"]["channel"])
            if kw["encoding"] == "MCRQI"
            else None
        ),
    ),
    "frqi_color_change": lambda qc, **kw: chromatic.frqi_color_change(
        qc,
        theta=float(kw["args"]["theta"]),
        color_qubit=(
            _channel_color_qubit(kw["encoder"], kw["args"]["channel"])
            if kw["encoding"] == "MCRQI"
            else None
        ),
    ),
    "neqr_color_complement": lambda qc, **kw: (
        _ncqi_complement(qc, kw["encoder"], kw["args"]["channel"])
        if kw["encoding"] == "NCQI"
        else chromatic.neqr_color_complement(qc, q=kw["encoder"].q)
    ),
    "neqr_half_intensity": lambda qc, **kw: (
        _ncqi_half_intensity(qc, kw["encoder"], kw["args"]["channel"])
        if kw["encoding"] == "NCQI"
        # UI runs on a simulator: the textbook reset(q-1) is fine here.
        else chromatic.neqr_half_intensity(qc, q=kw["encoder"].q, allow_reset=True)
    ),
    "neqr_classify_complement": lambda qc, **kw: (
        _ncqi_classify(qc, kw["encoder"], kw["args"]["channel"], kw["args"]["threshold_bit"])
        if kw["encoding"] == "NCQI"
        else chromatic.neqr_classify_complement(
            qc, q=kw["encoder"].q, threshold_bit=int(kw["args"]["threshold_bit"])
        )
    ),
    "qft_pos": lambda qc, **kw: apply_qft(qc, kw["pos_qubits"]),
    "iqft_pos": lambda qc, **kw: apply_inverse_qft(qc, kw["pos_qubits"]),
    "qft_intensity": lambda qc, **kw: apply_qft(
        qc, _intensity_qubits(kw["encoder"], kw["encoding"], kw["args"])
    ),
    "iqft_intensity": lambda qc, **kw: apply_inverse_qft(
        qc, _intensity_qubits(kw["encoder"], kw["encoding"], kw["args"])
    ),
}


def _render_param_inputs(spec: dict) -> dict:
    """Render the Streamlit widgets for an op's parameters and return their values."""
    args: dict = {}
    if not spec["params"]:
        return args
    cols = st.columns(min(3, len(spec["params"])))
    for i, p in enumerate(spec["params"]):
        col = cols[i % len(cols)]
        with col:
            wid_key = f"wiz_op_param_{spec['key']}_{p['name']}"
            if p["kind"] == "int_slider":
                args[p["name"]] = st.slider(
                    p["name"], p["min"], p["max"], p["default"], key=wid_key
                )
            elif p["kind"] == "float_slider":
                args[p["name"]] = st.slider(
                    p["name"],
                    float(p["min"]),
                    float(p["max"]),
                    float(p["default"]),
                    step=float(p.get("step", 0.01)),
                    key=wid_key,
                )
            elif p["kind"] == "choice":
                args[p["name"]] = st.selectbox(
                    p["name"], p["choices"], index=p["choices"].index(p["default"]), key=wid_key
                )
            elif p["kind"] == "text":
                args[p["name"]] = st.text_input(
                    p["name"], value=p["default"], help=p.get("help"), key=wid_key
                )
    return args


def _reset_processed_circuit() -> None:
    """Rebuild the encoded circuit from the preprocessed image, clearing the chain."""
    encoding = st.session_state["wiz_encoding"]
    encoder = st.session_state["wiz_encoder"]
    _, cast, _ = _ENCODER_SPECS[encoding]
    st.session_state["wiz_circuit"] = encoder.encode(cast(st.session_state["wiz_image"]))
    st.session_state["wiz_op_chain"] = []
    for key in ("wiz_proc_preview", "wiz_counts", "wiz_decoded"):
        st.session_state.pop(key, None)


with st.container(border=True):
    st.subheader("4️⃣ Process (optional)")
    if not _step_done("encode"):
        st.info("Encode first.")
    else:
        encoding = st.session_state["wiz_encoding"]
        encoder = st.session_state["wiz_encoder"]
        n = st.session_state["wiz_n"]

        catalog = _build_op_catalog(encoding, n)

        # ---- Category + operation pickers ---------------------------------
        non_empty_cats = [c for c, ops in catalog.items() if ops]
        col_cat, col_op = st.columns([1, 3])
        with col_cat:
            category = st.selectbox("Category", non_empty_cats, key="wiz_op_category")
        ops_in_cat = catalog[category]
        labels = [op["label"] for op in ops_in_cat]
        with col_op:
            op_label = st.selectbox("Operation", labels, key=f"wiz_op_{category}")
        spec = ops_in_cat[labels.index(op_label)]

        # ---- Conditional parameter widgets --------------------------------
        args = _render_param_inputs(spec)

        # ---- Action buttons -----------------------------------------------
        chain: list[dict] = st.session_state.setdefault("wiz_op_chain", [])
        col_add, col_apply, col_reset, _ = st.columns([1, 1, 1, 2])
        with col_add:
            add_only_clicked = st.button(
                "Add (no preview)",
                use_container_width=True,
                key="wiz_add_op",
                help=(
                    "Append the op to the chain without running a simulation. "
                    "Recommended for many-op chains or large n where each "
                    "preview is slow."
                ),
            )
        with col_apply:
            apply_clicked = st.button(
                "Add & preview",
                type="primary",
                use_container_width=True,
                key="wiz_apply_op",
                help="Append the op, then run an 8 k-shot ideal simulation to refresh the preview.",
            )
        with col_reset:
            reset_clicked = st.button(
                "Reset ops",
                use_container_width=True,
                key="wiz_reset_ops",
                disabled=not chain,
                help="Discard every applied operation and start from the encoded circuit.",
            )

        if reset_clicked:
            try:
                _reset_processed_circuit()
                st.success("Reset — circuit is back to the freshly-encoded image.")
                st.rerun()
            except Exception as exc:
                st.error(f"Reset failed: {exc}")

        if add_only_clicked or apply_clicked:
            qc = st.session_state["wiz_circuit"]
            try:
                _apply_op_to_qc(qc, spec, args, n=n, encoding=encoding, encoder=encoder)
                st.session_state["wiz_circuit"] = qc
                st.session_state["wiz_op_chain"] = [
                    *chain,
                    {"label": op_label, "args": args},
                ]
                # Invalidate the final-run results since the circuit changed.
                st.session_state.pop("wiz_counts", None)
                st.session_state.pop("wiz_decoded", None)
                if apply_clicked:
                    # Quick ideal-simulation preview so the user sees the effect.
                    try:
                        counts = ideal_simulation(qc, shots=8_192)
                        st.session_state["wiz_proc_preview"] = _unwrap_decoded(
                            encoder.decode(counts), encoding
                        )
                    except Exception as exc:
                        st.warning(f"Couldn't render preview ({exc}). Circuit was applied anyway.")
                else:
                    # No-preview path: discard the stale preview to avoid showing
                    # an "After N ops" panel that no longer matches the chain.
                    st.session_state.pop("wiz_proc_preview", None)
                st.success(f"Applied `{op_label}`")
                st.rerun()
            except Exception as exc:
                st.error(f"Processing failed: {exc}")

        # ----------------- Chain history + before/after preview ----------------
        if chain:
            chain_str = " → ".join(
                f"`{entry['label']}`" + (f" {entry['args']}" if entry["args"] else "")
                for entry in chain
            )
            st.markdown(f"**Chain ({len(chain)} ops):** {chain_str}")
        else:
            st.caption("No operations applied yet. The circuit will pass through to step 5 as-is.")

        preview = st.session_state.get("wiz_proc_preview")
        if preview is not None:
            input_img = st.session_state["wiz_image"]
            # Share a [vmin, vmax] window between both panels so that an
            # identical image after a no-op doesn't appear darker / brighter
            # just because the two encoders use different intensity ranges
            # (e.g. NEQR decodes into [0, 2^q-1] while uint8 input is [0,255]).
            input_arr = np.asarray(input_img)
            preview_arr = np.asarray(preview)
            shared_vmin = float(min(input_arr.min(), preview_arr.min()))
            shared_vmax = float(max(input_arr.max(), preview_arr.max(), shared_vmin + 1.0))

            prev_col_a, prev_col_b = st.columns(2)
            with prev_col_a:
                st.caption("Before (input image)")
                st.image(
                    upscale_for_display(
                        to_display_uint8(input_img, vmin=shared_vmin, vmax=shared_vmax)
                    ),
                    width=320,
                    clamp=True,
                )
            with prev_col_b:
                st.caption(f"After {len(chain)} op(s) — ideal preview, 8 192 shots")
                st.image(
                    upscale_for_display(
                        to_display_uint8(preview, vmin=shared_vmin, vmax=shared_vmax)
                    ),
                    width=320,
                    clamp=True,
                )


# ================================================== STEP 5: EXECUTE & EXPORT ==

with st.container(border=True):
    st.subheader("5️⃣ Execute & export")
    if not _step_done("encode"):
        st.info("Encode first.")
    else:
        qc = st.session_state["wiz_circuit"]
        encoding = st.session_state["wiz_encoding"]
        encoder = st.session_state["wiz_encoder"]
        n = st.session_state["wiz_n"]

        backend_choice = st.radio(
            "Backend",
            ["Local — ideal", "Local — noisy (depolarizing)", "IBM Quantum hardware"],
            key="wiz_backend",
        )

        # ---- Local execution ----
        if backend_choice.startswith("Local"):
            shots = st.slider("Shots", 1_000, 200_000, 40_000, step=1_000, key="wiz_local_shots")
            noise_p = 0.0
            if "noisy" in backend_choice:
                noise_p = st.slider(
                    "Per-gate depolarizing probability",
                    0.0,
                    0.05,
                    0.001,
                    step=0.0005,
                    format="%.4f",
                    key="wiz_noise_p",
                )

            if st.button("Run", type="primary", use_container_width=True, key="wiz_run_local"):
                from qimp.testing import noisy_simulation

                start = time.perf_counter()
                try:
                    if noise_p > 0:
                        from qiskit_aer.noise import NoiseModel, depolarizing_error

                        nm = NoiseModel()
                        for k, names in (
                            (1, ["u1", "u2", "u3", "rx", "ry", "rz", "sx", "h", "x", "y", "z"]),
                            (2, ["cx", "cz", "swap"]),
                            (3, ["ccx"]),
                        ):
                            err = depolarizing_error(noise_p, k)
                            for name in names:
                                nm.add_all_qubit_quantum_error(err, name)
                        counts = noisy_simulation(qc, shots=shots, noise_model=nm)
                    else:
                        counts = ideal_simulation(qc, shots=shots)
                    elapsed = time.perf_counter() - start
                    st.session_state["wiz_counts"] = counts
                    st.session_state["wiz_elapsed"] = elapsed
                    st.session_state["wiz_decoded"] = _unwrap_decoded(
                        encoder.decode(counts), encoding
                    )
                    st.success(f"Ran in {elapsed:.2f}s — {len(counts)} unique outcomes")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Run failed: {exc}")

        # ---- IBM Quantum ----
        else:
            if not have_ibm_runtime():
                st.warning(
                    "`qiskit-ibm-runtime` is not installed. Install with "
                    "`pip install qimp-mi[ibm]`. You can still **download the QASM 3** "
                    "and paste it into the [IBM Quantum Composer](https://quantum.ibm.com/composer)."
                )
            else:
                st.markdown(
                    "Get your API token from <https://quantum.ibm.com> (Account → API token). "
                    "The token is held in browser session memory only — closing this tab forgets it."
                )
                col_t, col_b = st.columns([3, 2])
                with col_t:
                    token = st.text_input(
                        "API token (leave blank to use saved account)",
                        type="password",
                        key="wiz_ibm_token",
                    )
                with col_b:
                    channel = st.selectbox(
                        "Channel",
                        ["ibm_quantum_platform", "ibm_quantum"],
                        index=0,
                        key="wiz_ibm_channel",
                    )

                if st.button("List backends", key="wiz_list_backends"):
                    try:
                        backends = list_ibm_backends(
                            token=token or None,
                            channel=channel,
                            operational_only=True,
                            simulators=True,
                        )
                        st.session_state["wiz_ibm_backends"] = backends
                        st.success(f"Found {len(backends)} backends")
                    except Exception as exc:
                        st.error(f"Couldn't list backends: {sanitize_exception(exc, token)}")

                backends = st.session_state.get("wiz_ibm_backends") or []
                col_bk, col_sh = st.columns([3, 2])
                with col_bk:
                    backend_name = st.selectbox(
                        "Backend",
                        options=backends or ["(none — click List backends)"],
                        key="wiz_ibm_backend_name",
                    )
                with col_sh:
                    ibm_shots = st.number_input(
                        "Shots", 1, 100_000, 4_096, step=1_000, key="wiz_ibm_shots"
                    )

                if st.button("Submit to IBM", type="primary", key="wiz_submit_ibm"):
                    if not backends:
                        st.error("List backends first.")
                    else:
                        try:
                            result = run_on_ibm(
                                qc,
                                backend_name=backend_name,
                                token=token or None,
                                channel=channel,
                                shots=int(ibm_shots),
                            )
                            st.session_state["wiz_ibm_job"] = result
                            st.success(
                                f"Submitted job `{result['job_id']}` on `{result['backend']}`. "
                                f"Status: {result['status']}"
                            )
                        except Exception as exc:
                            st.error(f"Submit failed: {sanitize_exception(exc, token)}")

                job = st.session_state.get("wiz_ibm_job")
                if (
                    job is not None
                    and not job.get("counts")
                    and st.button("Retrieve results", key="wiz_retrieve_ibm")
                ):
                    from _ibm import retrieve_ibm_job

                    try:
                        updated = retrieve_ibm_job(
                            job["job_id"], token=token or None, channel=channel
                        )
                        st.session_state["wiz_ibm_job"] = updated
                        if updated["counts"]:
                            st.session_state["wiz_counts"] = updated["counts"]
                            st.session_state["wiz_decoded"] = _unwrap_decoded(
                                encoder.decode(updated["counts"]), encoding
                            )
                            st.success(f"Job done — {len(updated['counts'])} outcomes")
                        else:
                            st.info(f"Status: {updated['status']}")
                    except Exception as exc:
                        st.error(f"Retrieve failed: {sanitize_exception(exc, token)}")

        # ---- Always-available exports ----
        st.divider()
        st.markdown("### Exports")

        def _get_qasm_cached() -> str:
            """Compute the QASM 3 export lazily, cached by circuit identity.

            For FRQI n≥5 the fallback decompose+transpile can take several
            seconds; without caching we'd pay that on every Streamlit rerun.
            """
            cache = st.session_state.get("wiz_qasm_cache")
            if cache is not None and cache[0] is qc:
                return cache[1]
            text = circuit_to_qasm3(qc)
            st.session_state["wiz_qasm_cache"] = (qc, text)
            return text

        exp_col_a, exp_col_b = st.columns(2)
        with exp_col_a:
            # Pass a callable so the QASM is built only when the user clicks.
            st.download_button(
                "⬇️ Download circuit (QASM 3)",
                data=_get_qasm_cached,
                file_name=f"qimp_{encoding.lower()}_n{n}.qasm",
                mime="text/plain",
                use_container_width=True,
            )
        with exp_col_b:
            if st.session_state.get("wiz_decoded") is not None and st.button(
                "💾 Save outputs to data/output/", use_container_width=True, key="wiz_save"
            ):
                decoded = st.session_state["wiz_decoded"]
                img_in = st.session_state["wiz_image"]
                out_dir = new_output_dir(prefix=f"workflow_{encoding.lower()}")
                panels = [("input", img_in), (f"{encoding}_decoded", decoded)]
                save_named_panels(panels, out_dir)
                (out_dir / "circuit.qasm").write_text(_get_qasm_cached(), encoding="utf-8")
                st.success(f"Saved to {out_dir}")

        # ---- Results ----
        if st.session_state.get("wiz_decoded") is not None:
            st.divider()
            decoded = st.session_state["wiz_decoded"]
            img_in = st.session_state["wiz_image"]
            from _viz import panel_grid_figure

            st.pyplot(
                panel_grid_figure([("Input", img_in), (f"{encoding} decoded", decoded)], cols=2)
            )

            from qimp.metrics import mse, psnr

            try:
                err = float(mse(img_in.astype(np.float64), decoded.astype(np.float64)))
                if err == 0:
                    fidelity = "∞ dB (exact)"
                else:
                    max_int = float(max(img_in.max(), 1.0))
                    fidelity = f"{psnr(img_in, decoded, max_intensity=max_int):.2f} dB"
            except Exception:
                err = float("nan")
                fidelity = "n/a"

            st.write(
                {
                    "shots": sum(st.session_state.get("wiz_counts", {}).values()),
                    "MSE": round(err, 4),
                    "PSNR": fidelity,
                }
            )
