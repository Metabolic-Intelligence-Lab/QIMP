"""System Info — environment, simulator capabilities, dataset stats, cache controls."""

from __future__ import annotations

import sys
from pathlib import Path

_QIMP_APP_ROOT = Path(__file__).resolve().parent.parent
if str(_QIMP_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_QIMP_APP_ROOT))


import platform

import streamlit as st
from app_io import (
    DATASET_GRAYSCALE,
    DATASET_RGB,
    OUTPUT_ROOT,
    discover_dataset_images,
)

st.set_page_config(page_title="System Info", page_icon="ℹ️", layout="wide")
st.title("ℹ️ System Info")

col_env, col_sim = st.columns(2)

# ----------------------------------------------------------- Environment ----
with col_env:
    st.subheader("Python & libraries")
    info: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for mod_name in (
        "qimp",
        "qiskit",
        "qiskit_aer",
        "numpy",
        "scipy",
        "PIL",
        "streamlit",
        "pandas",
    ):
        try:
            mod = __import__(mod_name)
            info[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[mod_name] = "(not installed)"
    st.write(info)


# ------------------------------------------------------- Simulator backend ----
with col_sim:
    st.subheader("Simulator")
    from qimp.runtime import SimulatorManager

    sim_info: dict[str, object] = {
        "GPU available": SimulatorManager.is_gpu_available(),
    }
    try:
        sim = SimulatorManager.get()
        sim_info["device"] = getattr(sim.options, "device", "unknown")
        sim_info["method"] = getattr(sim.options, "method", "automatic")
    except Exception as exc:
        sim_info["error"] = str(exc)
    st.write(sim_info)

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        st.caption("IBM Quantum Runtime is installed (`[ibm]` extra).")
        st.write({"qiskit_ibm_runtime": True})
        if st.button("Probe IBM saved account (read-only)"):
            try:
                service = QiskitRuntimeService()
                backends = [b.name for b in service.backends()][:10]
                st.success(f"Account OK. Sample backends: {backends}")
            except Exception as exc:
                st.error(f"No active IBM account: {exc}")
    except ImportError:
        st.caption("`qiskit-ibm-runtime` not installed. Install with `pip install qimp-mi[ibm]`.")


st.divider()

# ----------------------------------------------------------- Dataset stats ----
st.subheader("Dataset")
gs = discover_dataset_images(DATASET_GRAYSCALE, max_items=10_000)
rgb = discover_dataset_images(DATASET_RGB, max_items=10_000)
st.write(
    {
        "grayscale_dir": str(DATASET_GRAYSCALE),
        "grayscale_count": len(gs),
        "rgb_dir": str(DATASET_RGB),
        "rgb_count": len(rgb),
    }
)

# ----------------------------------------------------------- Output runs ----
st.subheader("Past runs")
runs: list[Path] = []
if OUTPUT_ROOT.exists():
    runs = sorted([p for p in OUTPUT_ROOT.iterdir() if p.is_dir()], reverse=True)
st.write({"output_dir": str(OUTPUT_ROOT), "run_count": len(runs)})
if runs:
    st.markdown("Recent runs:")
    for run in runs[:10]:
        n_files = sum(1 for _ in run.iterdir())
        st.markdown(f"- `{run.name}` ({n_files} files)")


st.divider()

# ---------------------------------------------------------- Cache controls ----
st.subheader("Cache controls")
st.caption(
    "QIMP keeps two process-wide caches for performance: a memory pool of "
    "reusable image buffers, and an `lru_cache` of base FRQI circuits keyed "
    "by `(n, m)`. Both safely rebuild on next use."
)
col_a, col_b = st.columns(2)
with col_a:
    if st.button("Clear memory pool", use_container_width=True):
        from qimp.runtime import clear_memory_pool

        clear_memory_pool()
        st.success("Memory pool cleared.")
with col_b:
    if st.button("Clear circuit cache", use_container_width=True):
        from qimp.runtime import clear_circuit_cache

        clear_circuit_cache()
        st.success("Circuit cache cleared.")


st.divider()

# ----------------------------------------------------------- Roadmap stubs ----
st.subheader("Library stubs (not yet implemented)")
st.markdown(
    "The following modules exist in the library but contain only `TODO` "
    "stubs — track v0.2 for their implementation:\n\n"
    "- `qimp.encoding.mcrqi` / `qimp.encoding.ncqi` — FRQI / NEQR for RGB.\n"
    "- `qimp.encoding.compression` — Boolean-function minimisation (Quine–McCluskey).\n"
    "- `qimp.processing.arithmetic` — quantum ADD/SUB, comparator, sort on NEQR.\n"
    "- `qimp.qml.classifier` — variational image classifier.\n"
    "- `qimp.io.datasets` — batch iterator over image folders."
)
