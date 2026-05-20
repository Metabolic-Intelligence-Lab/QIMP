"""Matplotlib figure helpers used by the Streamlit pages.

Independent of `streamlit` so the helpers can be exercised in plain unit tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def upscale_for_display(arr: np.ndarray, target_min_side: int = 320) -> np.ndarray:
    """Replicate pixels with NEAREST-neighbor so a small image stays crisp in
    the browser regardless of CSS scaling.

    Streamlit's `st.image` applies smooth-scaling when the browser has to
    enlarge a tiny image like 16×16 → 320 px. Pre-enlarging the array via
    `np.repeat` makes the browser display 1:1 pixels.

    `target_min_side` is the minimum side we want before handing off to the
    browser. Returned array's side is the smallest integer multiple of the
    original side that's ≥ `target_min_side`.
    """
    a = np.asarray(arr)
    if a.ndim not in (2, 3):
        return a
    side = a.shape[0]
    if side >= target_min_side or side == 0:
        return a
    factor = (target_min_side + side - 1) // side  # ceil division
    return np.repeat(np.repeat(a, factor, axis=0), factor, axis=1)


def to_display_uint8(
    img: np.ndarray, *, vmin: float | None = None, vmax: float | None = None
) -> np.ndarray:
    """Normalize any 2-D or 3-D image to uint8 [0, 255] for ``st.image``.

    - 2-D grayscale: linear stretch from ``[min, max]`` to ``[0, 255]``.
    - 3-D uint8 RGB/RGBA: returned unchanged.
    - 3-D uint16 RGB: downshifted to uint8 (divide by 257).
    - 3-D float RGB: scaled if max ≤ 1.0, else clipped to [0, 255].

    Pass ``vmin`` / ``vmax`` to override the auto-stretch with a fixed
    intensity window — useful when rendering before/after panels that must
    share a scale (otherwise the same content can appear with different
    brightness if the two images have different ``max`` values).
    """
    arr = np.asarray(img)
    if arr.ndim == 3:
        if arr.dtype == np.uint8:
            return arr
        if arr.dtype == np.uint16:
            return (arr // 257).astype(np.uint8)
        if np.issubdtype(arr.dtype, np.floating):
            if float(arr.max()) <= 1.0:
                return np.clip(arr * 255, 0, 255).astype(np.uint8)
            return np.clip(arr, 0, 255).astype(np.uint8)
        return np.clip(arr, 0, 255).astype(np.uint8)
    # Grayscale.
    lo = float(arr.min()) if vmin is None else float(vmin)
    hi = float(arr.max()) if vmax is None else float(vmax)
    if hi > lo:
        return np.clip(np.interp(arr, [lo, hi], [0, 255]), 0, 255).astype(np.uint8)
    return np.zeros_like(arr, dtype=np.uint8)


def image_figure(
    image: np.ndarray,
    *,
    title: str | None = None,
    cmap: str = "gray",
    figsize: tuple[float, float] = (4.0, 4.0),
) -> Any:
    """Return a matplotlib Figure showing `image`."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap=cmap, interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    return fig


def panel_grid_figure(
    panels: list[tuple[str, np.ndarray]],
    *,
    cols: int = 3,
    cmap: str = "gray",
    panel_size: float = 4.0,
) -> Any:
    """Return a Figure of (title, image) panels arranged in a grid."""
    import matplotlib.pyplot as plt

    n = len(panels)
    if n == 0:
        raise ValueError("panels list is empty")
    cols = min(n, cols)
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(panel_size * cols, panel_size * rows))
    axs = np.atleast_2d(axs)
    for idx, (title, img) in enumerate(panels):
        r, c = divmod(idx, cols)
        ax = axs[r, c]
        ax.imshow(img, cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axs[r, c].axis("off")
    fig.tight_layout()
    return fig


def bar_chart_figure(
    labels: list[str],
    values: list[float],
    *,
    title: str = "",
    ylabel: str = "",
    figsize: tuple[float, float] = (6.0, 3.5),
) -> Any:
    """Return a Figure with a vertical bar chart."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, values, color="steelblue")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def safe_circuit_figure(qc: Any, max_ops: int = 40) -> Any | None:
    """Return ``qc.draw('mpl')`` only if the circuit has at most `max_ops` ops.

    Larger circuits would produce huge figures that lock up the browser.
    Returns None to signal "too big to render".
    """
    try:
        op_count = sum(qc.count_ops().values())
    except Exception:
        return None
    if op_count > max_ops:
        return None
    try:
        return qc.draw("mpl", fold=20)
    except Exception:
        return None
