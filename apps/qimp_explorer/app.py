"""QIMP Explorer — sequential workflow wizard.

Walk an image through every step of the QIMP pipeline:

  1. Load image          (upload or pick from data/immagini/)
  2. Preprocess          (resize to 2^n × 2^n, optional grayscale)
  3. Encode              (FRQI / NEQR / QPIE / MCRQI / NCQI)
  4. Process (optional)  (geometric / chromatic / QFT / QHED)
  5. Execute & export    (ideal / noisy / IBM hardware → save outputs)

Each step unlocks only when its prerequisites are met. Click "Reset" at the
top to start over. The sidebar still exposes the individual advanced pages
(Encoder Explorer, Processing Playground, Benchmark, GP-ratio, System Info)
for power-user workflows.
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

from _ibm import circuit_to_qasm3, have_ibm_runtime, list_ibm_backends, run_on_ibm
from _viz import to_display_uint8
from app_io import (
    DATASET_GRAYSCALE,
    DATASET_RGB,
    RESIZE_OPTIONS,
    discover_dataset_images,
    load_image,
    new_output_dir,
    resize_to_square,
    save_named_panels,
    to_grayscale,
)

st.set_page_config(page_title="QIMP Explorer", page_icon="🔬", layout="wide")
st.title("🔬 QIMP Explorer — Workflow")
st.caption(
    "Sequential wizard for **qimp-mi**. Each step activates as soon as the "
    "previous one is complete. Use the sidebar pages for advanced single-step "
    "exploration."
)


# ---------------------------------------------------------- State helpers ----

_STATE_KEYS = (
    "wiz_image_raw",
    "wiz_image_source",
    "wiz_image",
    "wiz_n",
    "wiz_encoding",
    "wiz_encoder",
    "wiz_circuit",
    "wiz_counts",
    "wiz_decoded",
    "wiz_run_id",
    "wiz_ibm_job",
)


def _reset_all() -> None:
    for key in _STATE_KEYS:
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
        dataset_paths = discover_dataset_images(dataset_dir, max_items=50)
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
        preview = to_display_uint8(raw)
        st.image(preview, width=200, clamp=True)
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
            st.image(to_display_uint8(img), width=240, clamp=True)
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
                    encoder: object
                    if encoding == "FRQI":
                        from qimp.encoding.frqi import FrqiEncoder

                        enc = FrqiEncoder()
                        qc = enc.encode(img.astype(np.uint8) if img.dtype != np.uint8 else img)
                        encoder = enc
                    elif encoding == "NEQR":
                        from qimp.encoding.neqr import NeqrEncoder

                        if img.max() >= (1 << q_qubits):
                            st.error(
                                f"Image max {img.max()} exceeds 2^q-1 = {(1 << q_qubits) - 1}. "
                                "Increase q."
                            )
                            st.stop()
                        enc = NeqrEncoder(q=q_qubits)
                        qc = enc.encode(img.astype(np.int64))
                        encoder = enc
                    elif encoding == "QPIE":
                        from qimp.encoding.qpie import QpieEncoder

                        enc = QpieEncoder()
                        qc = enc.encode(img.astype(np.float64))
                        encoder = enc
                    elif encoding == "MCRQI":
                        from qimp.encoding.mcrqi import McrqiEncoder

                        enc = McrqiEncoder()
                        qc = enc.encode(img.astype(np.uint8) if img.dtype != np.uint8 else img)
                        encoder = enc
                    else:  # NCQI
                        from qimp.encoding.ncqi import NcqiEncoder

                        if img.max() >= (1 << q_qubits):
                            st.error(
                                f"Image max {img.max()} exceeds 2^q-1 = {(1 << q_qubits) - 1}. "
                                "Increase q."
                            )
                            st.stop()
                        enc = NcqiEncoder(q=q_qubits)
                        qc = enc.encode(img.astype(np.int64))
                        encoder = enc

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

with st.container(border=True):
    st.subheader("4️⃣ Process (optional)")
    if not _step_done("encode"):
        st.info("Encode first.")
    else:
        encoding = st.session_state["wiz_encoding"]
        n = st.session_state["wiz_n"]

        op_options = ["(skip processing)"]
        if encoding == "FRQI":
            op_options += ["frqi_color_complement"]
        elif encoding == "NEQR":
            op_options += ["neqr_color_complement", "neqr_half_intensity"]

        # Geometric ops apply to any encoding that has a position register.
        op_options += [
            "axis_flip (X)",
            "axis_flip (Y)",
            "coord_swap (transpose)",
            "ort_rotation 180°",
        ]

        op = st.selectbox("Operation", op_options, key="wiz_op")

        if st.button("Apply", use_container_width=True, key="wiz_apply_op"):
            if op == "(skip processing)":
                st.info("No-op (proceeding).")
            else:
                qc = st.session_state["wiz_circuit"].copy()
                try:
                    # Position offset: 0 for FRQI/QPIE/MCRQI, q for NEQR, 3q for NCQI.
                    if encoding == "NEQR":
                        pos_off = st.session_state["wiz_encoder"].q
                    elif encoding == "NCQI":
                        pos_off = 3 * st.session_state["wiz_encoder"].q
                    else:
                        pos_off = 0

                    if op == "axis_flip (X)":
                        from qimp.processing.geometric import axis_flip

                        axis_flip(qc, n=n, axis="x", pos_offset=pos_off)
                    elif op == "axis_flip (Y)":
                        from qimp.processing.geometric import axis_flip

                        axis_flip(qc, n=n, axis="y", pos_offset=pos_off)
                    elif op == "coord_swap (transpose)":
                        from qimp.processing.geometric import coord_swap

                        coord_swap(qc, n=n, pos_offset=pos_off)
                    elif op == "ort_rotation 180°":
                        from qimp.processing.geometric import ort_rotation

                        ort_rotation(qc, n=n, angle=180, pos_offset=pos_off)
                    elif op == "frqi_color_complement":
                        from qimp.processing.chromatic import frqi_color_complement

                        frqi_color_complement(qc)
                    elif op == "neqr_color_complement":
                        from qimp.processing.chromatic import neqr_color_complement

                        neqr_color_complement(qc, q=st.session_state["wiz_encoder"].q)
                    elif op == "neqr_half_intensity":
                        from qimp.processing.chromatic import neqr_half_intensity

                        neqr_half_intensity(qc, q=st.session_state["wiz_encoder"].q)
                    st.session_state["wiz_circuit"] = qc
                    st.session_state.pop("wiz_counts", None)
                    st.session_state.pop("wiz_decoded", None)
                    st.success(f"Applied {op}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Processing failed: {exc}")


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
                from qimp.testing import ideal_simulation, noisy_simulation

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
                    st.session_state["wiz_decoded"] = encoder.decode(counts)
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
                        st.error(f"Couldn't list backends: {exc}")

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
                            st.error(f"Submit failed: {exc}")

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
                            st.session_state["wiz_decoded"] = encoder.decode(updated["counts"])
                            st.success(f"Job done — {len(updated['counts'])} outcomes")
                        else:
                            st.info(f"Status: {updated['status']}")
                    except Exception as exc:
                        st.error(f"Retrieve failed: {exc}")

        # ---- Always-available exports ----
        st.divider()
        st.markdown("### Exports")
        exp_col_a, exp_col_b = st.columns(2)
        with exp_col_a:
            qasm = circuit_to_qasm3(qc)
            st.download_button(
                "⬇️ Download circuit (QASM 3)",
                data=qasm,
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
                (out_dir / "circuit.qasm").write_text(qasm, encoding="utf-8")
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
