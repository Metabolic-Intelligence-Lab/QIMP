"""QIMP Explorer — Home.

Pick an image (upload or from the lab dataset) and the four pages in the
sidebar take it through every step of the QIMP pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

_QIMP_APP_ROOT = Path(__file__).resolve().parent
if str(_QIMP_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_QIMP_APP_ROOT))


import io
from pathlib import Path

import numpy as np
import streamlit as st
from app_io import (
    DATASET_GRAYSCALE,
    discover_dataset_images,
    infer_n_from_image,
    is_power_of_two,
    load_image,
)
from _viz import image_figure
from PIL import Image

st.set_page_config(
    page_title="QIMP Explorer",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 QIMP Explorer")
st.markdown(
    "Interactive playground for **qimp-mi** — pick an image, then use the pages "
    "in the sidebar to encode it (FRQI / NEQR / QPIE), apply processing, run a "
    "side-by-side benchmark, or compute the Green-Purple ratio.\n\n"
    "Source: [Metabolic-Intelligence-Lab/QIMP](https://github.com/Metabolic-Intelligence-Lab/QIMP) · "
    "Spec: [`docs/tesi.pdf`](../docs/tesi.pdf) · "
    "Library version: see `qimp.__version__`."
)


# ----------------------------------------------------------------- Sidebar ----

with st.sidebar:
    st.header("Image source")
    upload = st.file_uploader(
        "Upload a TIFF / PNG",
        type=["tif", "tiff", "png"],
        help="Square, single-channel for FRQI / NEQR / QPIE; RGB for GP-ratio.",
    )

    dataset_paths = discover_dataset_images(DATASET_GRAYSCALE)
    dataset_choice: Path | None = None
    if dataset_paths:
        labels = [p.name for p in dataset_paths]
        idx = st.selectbox(
            f"…or pick from `{DATASET_GRAYSCALE.relative_to(Path.cwd().parent)}`",
            options=list(range(len(labels))),
            format_func=lambda i: labels[i],
            index=0,
            key="dataset_idx",
        )
        dataset_choice = dataset_paths[idx]
    else:
        st.caption(
            f"No dataset images found at `{DATASET_GRAYSCALE}`. "
            "Sync OneDrive or upload an image directly."
        )

    source_choice = st.radio(
        "Active source",
        options=["Upload", "Dataset"],
        index=1 if upload is None and dataset_choice is not None else 0,
        horizontal=True,
    )

    if st.button("Load image", type="primary", use_container_width=True):
        try:
            if source_choice == "Upload":
                if upload is None:
                    st.error("Upload a file first.")
                    st.stop()
                arr = np.asarray(Image.open(io.BytesIO(upload.read())))
                source_name = upload.name
            else:
                if dataset_choice is None:
                    st.error("No dataset image available.")
                    st.stop()
                arr = load_image(dataset_choice)
                source_name = dataset_choice.name
            st.session_state["image"] = arr
            st.session_state["image_source"] = source_name
            st.success(f"Loaded {source_name}")
        except Exception as exc:
            st.error(f"Failed to load image: {exc}")


# ---------------------------------------------------------------- Main area ----

image = st.session_state.get("image")
if image is None:
    st.info("No image loaded yet. Use the sidebar to pick one.")
    st.stop()

source_name = st.session_state.get("image_source", "<unknown>")

col_image, col_info = st.columns([2, 1])
with col_image:
    if image.ndim == 2:
        fig = image_figure(image, title=source_name)
    elif image.ndim == 3 and image.shape[-1] in (3, 4):
        # Stack RGB channels next to each other to make picking obvious.
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 3, figsize=(9, 3.2))
        for i, (label, cmap) in enumerate([("R", "Reds"), ("G", "Greens"), ("B", "Blues")]):
            axs[i].imshow(image[:, :, i], cmap=cmap, interpolation="nearest")
            axs[i].set_title(label)
            axs[i].axis("off")
        fig.suptitle(source_name)
        fig.tight_layout()
    else:
        fig = None
        st.warning(f"Unsupported image shape {image.shape}.")
    if fig is not None:
        st.pyplot(fig)

with col_info:
    st.markdown("**Image info**")
    st.write(
        {
            "shape": tuple(image.shape),
            "dtype": str(image.dtype),
            "min": float(image.min()),
            "max": float(image.max()),
            "mean": round(float(image.mean()), 3),
        }
    )

    if image.ndim == 2:
        n = infer_n_from_image(image)
        if n is not None:
            st.success(f"Square 2D, side = 2^{n} → ready for FRQI/NEQR/QPIE.")
        else:
            side = image.shape[0] if image.shape[0] == image.shape[1] else None
            if side is None:
                st.warning("Not square. Crop or resize before encoding.")
            elif not is_power_of_two(side):
                st.warning(
                    f"Square ({side}×{side}) but side is not a power of two. "
                    "Use the **Processing Playground → resize** option, or pick a different tile."
                )
    elif image.ndim == 3:
        st.info("RGB image — head to **GP-ratio** in the sidebar.")

st.divider()
st.markdown(
    "### Pages\n"
    "1. **Encoder Explorer** — round-trip an image through FRQI, NEQR, or QPIE.\n"
    "2. **Processing Playground** — apply geometric / chromatic operations, "
    "or run QHED edge detection.\n"
    "3. **Benchmark** — compare all three encodings on the same image.\n"
    "4. **GP-ratio** — Green-Purple microscopy pipeline (classical + quantum)."
)
