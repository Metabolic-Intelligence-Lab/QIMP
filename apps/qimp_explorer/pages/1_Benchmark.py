"""Benchmark — run FRQI / NEQR / QPIE on the same image and compare metrics."""

from __future__ import annotations

import sys
from pathlib import Path

_QIMP_APP_ROOT = Path(__file__).resolve().parent.parent
if str(_QIMP_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_QIMP_APP_ROOT))


import time

import numpy as np
import pandas as pd
import streamlit as st
from app_io import (
    infer_n_from_image,
    new_output_dir,
    save_named_panels,
)
from _viz import bar_chart_figure, panel_grid_figure

from qimp.encoding.frqi import FrqiEncoder
from qimp.encoding.neqr import NeqrEncoder
from qimp.encoding.qpie import QpieEncoder
from qimp.metrics import psnr, total_variation, transpile_summary
from qimp.testing import ideal_simulation

st.set_page_config(page_title="Benchmark", page_icon="📊", layout="wide")
st.title("📊 Benchmark: FRQI vs NEQR vs QPIE")

image = st.session_state.get("wiz_image")
if image is None:
    image = st.session_state.get("wiz_image_raw")
if image is None:
    st.warning("Load and (optionally) preprocess an image in the **Workflow** wizard first.")
    st.stop()

if image.ndim != 2:
    st.error("Benchmark requires a 2D grayscale image.")
    st.stop()

n = infer_n_from_image(image)
if n is None:
    st.error(f"Image must be square 2^n × 2^n; got {image.shape}.")
    st.stop()


# ----------------------------------------------------------------- Sidebar ----

with st.sidebar:
    st.header("Settings")
    frqi_downsample = st.select_slider(
        "FRQI down-sample",
        options=[1, 2, 4, 8],
        value=1,
        help="FRQI gets expensive above n=3 due to multi-controlled RY gates. "
        "Bump this up if the FRQI step is too slow.",
    )
    neqr_downsample = st.select_slider("NEQR down-sample", options=[1, 2, 4], value=1)
    neqr_q = st.slider("NEQR intensity qubits q", 1, 8, 4)
    frqi_shots = st.number_input("FRQI shots", 1_000, 200_000, 40_000, step=1_000, format="%d")
    neqr_shots = st.number_input("NEQR shots", 1_000, 50_000, 8_192, step=1_000, format="%d")
    qpie_shots = st.number_input("QPIE shots", 1_000, 500_000, 100_000, step=1_000, format="%d")
    run = st.button("Run all", type="primary", use_container_width=True)

side = image.shape[0]
frqi_side = side // frqi_downsample
neqr_side = side // neqr_downsample
st.info(
    f"**Working sizes:**  \n"
    f"- FRQI: {frqi_side}×{frqi_side} (n = {int(np.log2(frqi_side))}, {2 * int(np.log2(frqi_side)) + 1} qubits)  \n"
    f"- NEQR: {neqr_side}×{neqr_side} (n = {int(np.log2(neqr_side))}, q = {neqr_q}, "
    f"{2 * int(np.log2(neqr_side)) + neqr_q} qubits)  \n"
    f"- QPIE: {side}×{side} (n = {n}, {2 * n} qubits) — always full size"
)


# ------------------------------------------------------------ Helper closures ----


def _bench_frqi() -> dict:
    img = image[::frqi_downsample, ::frqi_downsample]
    img_uint8 = img.astype(np.uint8) if img.dtype != np.uint8 else img
    start = time.perf_counter()
    enc = FrqiEncoder()
    qc = enc.encode(img_uint8)
    counts = ideal_simulation(qc, shots=int(frqi_shots))
    decoded = enc.decode(counts)
    elapsed = time.perf_counter() - start
    summary = transpile_summary(qc)
    return {
        "encoding": "FRQI",
        "qubits": qc.num_qubits,
        "depth": summary["depth"],
        "depth_transpiled": summary["depth_transpiled"],
        "shots": int(frqi_shots),
        "runtime_s": round(elapsed, 3),
        "psnr_dB": round(psnr(img_uint8, decoded, max_intensity=255.0), 2),
        "tv_decoded": round(total_variation(decoded.astype(np.float64)), 2),
        "exact": False,
        "input": img_uint8,
        "decoded": decoded,
    }


def _bench_neqr() -> dict:
    img = image[::neqr_downsample, ::neqr_downsample]
    max_int = (1 << int(neqr_q)) - 1
    if img.max() > max_int:
        raise ValueError(f"image max {img.max()} > 2^q-1 = {max_int}")
    img_int = img.astype(np.int64)
    start = time.perf_counter()
    enc = NeqrEncoder(q=int(neqr_q))
    qc = enc.encode(img_int)
    counts = ideal_simulation(qc, shots=int(neqr_shots))
    decoded = enc.decode(counts)
    elapsed = time.perf_counter() - start
    summary = transpile_summary(qc)
    return {
        "encoding": "NEQR",
        "qubits": qc.num_qubits,
        "depth": summary["depth"],
        "depth_transpiled": summary["depth_transpiled"],
        "shots": int(neqr_shots),
        "runtime_s": round(elapsed, 3),
        "psnr_dB": float("inf"),
        "tv_decoded": round(total_variation(decoded.astype(np.float64)), 2),
        "exact": bool(np.array_equal(decoded, img_int)),
        "input": img_int,
        "decoded": decoded.astype(np.float64),
    }


def _bench_qpie() -> dict:
    img = image.astype(np.float64)
    start = time.perf_counter()
    enc = QpieEncoder()
    qc = enc.encode(img)
    counts = ideal_simulation(qc, shots=int(qpie_shots))
    decoded = enc.decode(counts)
    elapsed = time.perf_counter() - start
    summary = transpile_summary(qc)
    return {
        "encoding": "QPIE",
        "qubits": qc.num_qubits,
        "depth": summary["depth"],
        "depth_transpiled": summary["depth_transpiled"],
        "shots": int(qpie_shots),
        "runtime_s": round(elapsed, 3),
        "psnr_dB": round(psnr(img, decoded, max_intensity=float(max(img.max(), 1.0))), 2),
        "tv_decoded": round(total_variation(decoded.astype(np.float64)), 2),
        "exact": False,
        "input": img,
        "decoded": decoded,
    }


# ----------------------------------------------------------------- Execute ----

if not run:
    st.info("Set parameters in the sidebar and press **Run all**.")
    st.stop()

results: list[dict] = []
with st.spinner("Running benchmarks…"):
    for fn in (_bench_frqi, _bench_neqr, _bench_qpie):
        try:
            results.append(fn())
        except Exception as exc:
            st.error(f"{fn.__name__} failed: {exc}")

if not results:
    st.stop()

# Strip the heavy arrays from the table view but keep them for plots.
table_cols = [
    "encoding",
    "qubits",
    "depth",
    "depth_transpiled",
    "shots",
    "runtime_s",
    "psnr_dB",
    "tv_decoded",
    "exact",
]
df = pd.DataFrame([{k: r[k] for k in table_cols} for r in results])

st.markdown("### Metrics")
st.dataframe(df, hide_index=True, use_container_width=True)

# Decoded images side by side.
panels = []
for r in results:
    panels.append((f"{r['encoding']} input", r["input"]))
    panels.append((f"{r['encoding']} decoded", r["decoded"]))
st.pyplot(panel_grid_figure(panels, cols=2))

# Bar charts.
col_psnr, col_depth = st.columns(2)
with col_psnr:
    psnr_values = [r["psnr_dB"] if r["psnr_dB"] != float("inf") else 100.0 for r in results]
    fig_psnr = bar_chart_figure(
        [r["encoding"] for r in results], psnr_values, title="PSNR (dB)", ylabel="dB"
    )
    st.pyplot(fig_psnr)
with col_depth:
    fig_depth = bar_chart_figure(
        [r["encoding"] for r in results],
        [r["depth_transpiled"] for r in results],
        title="Transpiled depth",
        ylabel="depth",
    )
    st.pyplot(fig_depth)

if st.button("Save table + plots", use_container_width=True):
    out_dir = new_output_dir(prefix="benchmark")
    df.to_csv(out_dir / "metrics.csv", index=False)
    fig_psnr.savefig(out_dir / "psnr.png", dpi=120, bbox_inches="tight")
    fig_depth.savefig(out_dir / "depth.png", dpi=120, bbox_inches="tight")
    save_named_panels(panels, out_dir)
    panel_grid_figure(panels, cols=2).savefig(
        out_dir / "comparison.png", dpi=120, bbox_inches="tight"
    )
    st.success(f"Saved to {out_dir}")
