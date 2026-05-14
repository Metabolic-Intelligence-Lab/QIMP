"""Matplotlib figure helpers used by the Streamlit pages.

Independent of `streamlit` so the helpers can be exercised in plain unit tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np


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
