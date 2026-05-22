"""GP-ratio — Green-Purple microscopy pipeline.

Walks an RGB tile through the GP pipeline:

  1. Load an RGB tile (upload or pick from ``data/immagini/trainQML/``).
  2. Optional Gaussian + median preprocessing on the R/G channels.
  3. Classical reference image (``(G − α·R) / (G + α·R)``).
  4. Quantum sub-circuit construction (parametric, n=log2(side)).
  5. Optional run + short COBYLA optimisation.

The image is cached in ``st.session_state["gp_*"]`` so each action operates
on the same loaded data; switching parameters does not silently reload.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

_QIMP_APP_ROOT = Path(__file__).resolve().parent.parent
if str(_QIMP_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_QIMP_APP_ROOT))


import numpy as np
import streamlit as st
from _cached import cached_discover_dataset_images
from _viz import panel_grid_figure
from app_io import DATASET_RGB, new_output_dir, save_tiff
from PIL import Image

from qimp.io.image import apply_filters, calculate_gp_image
from qimp.metrics import mse, psnr, total_variation, transpile_summary
from qimp.processing.gp_ratio import (
    analytical_gp_params,
    apply_gp_function,
    classical_gp_image,
    evaluate_gp,
    optimize_gp,
)

st.set_page_config(page_title="GP-ratio", page_icon="🟢🟣", layout="wide")
st.title("🟢🟣 GP-ratio Lab Pipeline")
st.caption(
    "Green-Purple ratio: classical reference + parametric quantum circuit + "
    "an optional short COBYLA optimisation. Heavy runs belong in a notebook; "
    "this page keeps every action under a minute."
)


# --------------------------------------------------------- Reset helper -----


def _reset_gp_state() -> None:
    for key in [k for k in st.session_state if k.startswith("gp_")]:
        st.session_state.pop(key, None)


# ---------------------------------------------------------- Sidebar -----


with st.sidebar:
    st.header("Inputs")
    upload = st.file_uploader("Upload RGB TIFF/PNG", type=["tif", "tiff", "png"])
    dataset_paths = [
        Path(p)
        for p in cached_discover_dataset_images(
            str(DATASET_RGB), pattern="*.tif", max_items=50, require_nonzero=True
        )
    ]
    dataset_choice: Path | None = None
    if dataset_paths:
        labels = [p.name for p in dataset_paths]
        try:
            dataset_label = str(DATASET_RGB.relative_to(Path.cwd().parent))
        except ValueError:
            dataset_label = DATASET_RGB.name
        idx = st.selectbox(
            f"…or pick from `{dataset_label}`",
            options=list(range(len(labels))),
            format_func=lambda i: labels[i],
        )
        dataset_choice = dataset_paths[idx]

    source = st.radio("Active source", ["Upload", "Dataset"], index=1 if upload is None else 0)
    target_size = st.select_slider("Resize to (side)", options=[4, 8, 16, 32], value=8)
    alpha = st.slider("α (red-channel weight)", 0.0, 2.0, 0.5, step=0.05)

    st.markdown("**Preprocessing filters (optional)**")
    use_filters = st.checkbox("Gaussian + median on R/G before GP", value=False)
    sigma = st.slider("Gaussian σ", 0.1, 3.0, 1.0, step=0.1, disabled=not use_filters)
    median_size = st.slider("Median kernel", 1, 7, 3, step=2, disabled=not use_filters)

    if st.button("🔄 Reset", use_container_width=True):
        _reset_gp_state()
        st.rerun()


# ----------------------------------------------------- Step 1: load -----


with st.container(border=True):
    st.subheader("1️⃣ Load RGB tile")
    if st.button("Load", type="primary", use_container_width=True, key="gp_load"):
        try:
            if source == "Upload":
                if upload is None:
                    st.error("Upload a file first.")
                    st.stop()
                rgb_full = np.asarray(Image.open(io.BytesIO(upload.read())).convert("RGB"))
                source_label = upload.name
            else:
                if dataset_choice is None:
                    st.error("No dataset image available.")
                    st.stop()
                rgb_full = np.asarray(Image.open(dataset_choice).convert("RGB"))
                source_label = dataset_choice.name

            rgb_pil = Image.fromarray(rgb_full).resize(
                (target_size, target_size), Image.Resampling.LANCZOS
            )
            rgb = np.asarray(rgb_pil)
            red = rgb[:, :, 0].astype(np.float64)
            green = rgb[:, :, 1].astype(np.float64)
            blue = rgb[:, :, 2].astype(np.float64)
            if use_filters:
                red = apply_filters(red, sigma=sigma, median_size=median_size)
                green = apply_filters(green, sigma=sigma, median_size=median_size)

            st.session_state["gp_red"] = red
            st.session_state["gp_green"] = green
            st.session_state["gp_blue"] = blue
            st.session_state["gp_n"] = int(np.log2(target_size))
            st.session_state["gp_source"] = source_label
            st.session_state["gp_alpha"] = alpha
            # Invalidate downstream when re-loading.
            for k in ("gp_quantum_image", "gp_classical_image", "gp_opt_result"):
                st.session_state.pop(k, None)
            st.success(f"Loaded {source_label} → {target_size}×{target_size}")
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to load: {exc}")

    if "gp_red" in st.session_state:
        st.pyplot(
            panel_grid_figure(
                [
                    ("Red", st.session_state["gp_red"]),
                    ("Green", st.session_state["gp_green"]),
                    ("Blue", st.session_state["gp_blue"]),
                ],
                cols=3,
            )
        )
        st.caption(
            f"`{st.session_state['gp_source']}` · n = {st.session_state['gp_n']} · "
            f"α = {st.session_state['gp_alpha']:.2f} · "
            f"filters: {f'σ={sigma}, median={median_size}' if use_filters else 'none'}"
        )


# ----------------------------------------------------- Step 2: classical -----


with st.container(border=True):
    st.subheader("2️⃣ Classical GP reference")
    if "gp_red" not in st.session_state:
        st.info("Load an RGB tile first.")
    else:
        red = st.session_state["gp_red"]
        green = st.session_state["gp_green"]
        a = st.session_state["gp_alpha"]
        if st.button("Compute classical GP", type="primary", key="gp_run_classical"):
            normalized = classical_gp_image(green, red, alpha=a)
            uint8 = calculate_gp_image(green, red, G=a, output_format="uint8")
            sixteen = calculate_gp_image(green, red, G=a, output_format="16bit")
            st.session_state["gp_classical_image"] = normalized
            st.session_state["gp_classical_uint8"] = uint8
            st.session_state["gp_classical_16bit"] = sixteen
            st.rerun()

        if "gp_classical_image" in st.session_state:
            ref = st.session_state["gp_classical_image"]
            st.pyplot(
                panel_grid_figure(
                    [
                        ("GP normalized [-1, 1]", ref),
                        ("GP uint8 [0, 255]", st.session_state["gp_classical_uint8"]),
                        ("GP 16-bit [0, 4096]", st.session_state["gp_classical_16bit"]),
                    ],
                    cols=3,
                )
            )
            st.write(
                {
                    "shape": ref.shape,
                    "range (normalized)": [
                        round(float(ref.min()), 3),
                        round(float(ref.max()), 3),
                    ],
                    "TV": round(float(total_variation(ref)), 3),
                }
            )
            if st.button("💾 Save classical outputs", key="gp_save_classical"):
                out = new_output_dir(prefix="gp_classical")
                save_tiff(red, out / "00_red.tif")
                save_tiff(green, out / "01_green.tif")
                save_tiff(ref, out / "02_gp_normalized.tif")
                save_tiff(st.session_state["gp_classical_uint8"], out / "03_gp_uint8.tif")
                save_tiff(st.session_state["gp_classical_16bit"], out / "04_gp_16bit.tif")
                st.success(f"Saved to {out}")


# ----------------------------------------------------- Step 3: quantum -----


with st.container(border=True):
    st.subheader("3️⃣ Quantum GP circuit")
    if "gp_red" not in st.session_state:
        st.info("Load an RGB tile first.")
    else:
        red = st.session_state["gp_red"]
        green = st.session_state["gp_green"]
        n = st.session_state["gp_n"]

        # Cheap circuit summary — built every time the section renders.
        from qiskit.circuit import Parameter

        from qimp.encoding.frqi import frqi_circuit

        rg_stack = np.stack([red, green], axis=0)
        norm = float(rg_stack.max()) or 1.0
        base_qc = frqi_circuit(rg_stack, normalization=norm)
        num_params = 2 * (1 << (2 * n))
        symbolic = [Parameter(f"θ{i}") for i in range(num_params)]
        apply_gp_function(base_qc, n=n, m=1, params=symbolic)
        summary = transpile_summary(base_qc)

        col_a, col_b = st.columns(2)
        with col_a:
            st.write(
                {
                    "qubits": base_qc.num_qubits,
                    "params": num_params,
                    "depth (pre-transpile)": summary["depth"],
                    "depth (post-transpile)": summary["depth_transpiled"],
                }
            )
        with col_b:
            run_analytical = st.button(
                "🎯 Use analytical params (exact, ms)",
                type="primary",
                use_container_width=True,
                key="gp_run_analytical",
                help="Compute the closed-form optimal parameters and run the circuit "
                "with them — exact match to the classical target at every n, "
                "without any numerical optimisation. Sub-millisecond per call.",
            )
            run_random = st.button(
                "▶ Run with random params",
                use_container_width=True,
                key="gp_run_random",
                help="Useful only as a sanity check of the encoding — the result won't "
                "match the classical reference until you optimise or use analytical params.",
            )
            run_optim = st.button(
                "🐢 Optimise via COBYLA (slow)",
                use_container_width=True,
                key="gp_run_optim",
                help="Numerical baseline (gradient-free). The analytical solver above "
                "reaches the same optimum in microseconds; use this only to inspect "
                "the convergence curve.",
            )
            opt_iters = st.slider("COBYLA iterations", 5, 200, 50, step=5, key="gp_opt_iters")

        if run_analytical:
            try:
                params = analytical_gp_params(green, red, alpha=st.session_state["gp_alpha"])
                gp_q = evaluate_gp(green, red, params, exact=True)
                st.session_state["gp_quantum_image"] = gp_q
                st.session_state["gp_quantum_params"] = params
                st.session_state["gp_quantum_label"] = "analytical (closed-form)"
                st.success("Computed analytical parameters — exact target.")
                st.rerun()
            except Exception as exc:
                st.error(f"Analytical evaluation failed: {exc}")

        if run_random:
            rng = np.random.default_rng(0)
            params = rng.uniform(-np.pi, np.pi, size=num_params)
            try:
                gp_q = evaluate_gp(green, red, params, exact=True)
                st.session_state["gp_quantum_image"] = gp_q
                st.session_state["gp_quantum_label"] = "random params"
                st.success("Evaluated with random parameters.")
                st.rerun()
            except Exception as exc:
                st.error(f"Quantum evaluation failed: {exc}")

        if run_optim:
            with st.spinner(f"COBYLA up to {opt_iters} iterations…"):
                try:
                    result = optimize_gp(
                        green,
                        red,
                        alpha=st.session_state["gp_alpha"],
                        max_iter=int(opt_iters),
                        seed=0,
                        exact=True,
                    )
                    st.session_state["gp_opt_result"] = result
                    st.session_state["gp_quantum_image"] = evaluate_gp(
                        green, red, result.optimized_params, exact=True
                    )
                    st.session_state["gp_quantum_label"] = (
                        f"optimised, {len(result.history_combined)} fn-evals"
                    )
                    st.success(
                        f"Optimised. Final combined loss = "
                        f"{result.history_combined[-1]:.4f}, "
                        f"MSE = {result.history_mse[-1]:.4f}, "
                        f"PSNR = {result.history_psnr[-1]:.2f} dB"
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Optimisation failed: {exc}")


# ----------------------------------------------------- Step 4: compare -----


with st.container(border=True):
    st.subheader("4️⃣ Classical vs quantum")
    if "gp_classical_image" not in st.session_state or "gp_quantum_image" not in st.session_state:
        st.info("Compute both the classical reference (step 2) and a quantum result (step 3).")
    else:
        ref = st.session_state["gp_classical_image"]
        gp_q = st.session_state["gp_quantum_image"]
        st.pyplot(
            panel_grid_figure(
                [
                    ("Classical (target)", ref),
                    (f"Quantum ({st.session_state['gp_quantum_label']})", gp_q),
                    ("Pixel-wise difference", gp_q - ref),
                ],
                cols=3,
            )
        )
        st.write(
            {
                "MSE": round(float(mse(ref, gp_q)), 5),
                "PSNR (vs target)": round(float(psnr(ref, gp_q, max_intensity=2.0)), 2),
                "TV (quantum)": round(float(total_variation(gp_q)), 3),
            }
        )
        if st.button("💾 Save comparison", key="gp_save_compare"):
            out = new_output_dir(prefix="gp_compare")
            save_tiff(ref.astype(np.float32), out / "00_classical_gp.tif")
            save_tiff(gp_q.astype(np.float32), out / "01_quantum_gp.tif")
            save_tiff((gp_q - ref).astype(np.float32), out / "02_difference.tif")
            if "gp_opt_result" in st.session_state:
                np.save(
                    out / "03_optimised_params.npy",
                    st.session_state["gp_opt_result"].optimized_params,
                )
            st.success(f"Saved to {out}")
