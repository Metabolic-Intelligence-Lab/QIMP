"""Streamlit-cached wrappers around the pure helpers in ``app_io``.

Kept in its own module so the pure helpers stay testable without a Streamlit
context. The wrappers below are the only Streamlit-aware glue.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from app_io import discover_dataset_images


@st.cache_data(show_spinner=False, ttl=300)
def cached_discover_dataset_images(
    directory: str,
    pattern: str = "*.tif",
    *,
    max_items: int = 50,
    require_nonzero: bool = True,
    mtime_hint: float | None = None,
) -> list[str]:
    """Cached dataset discovery, returning string paths.

    ``directory`` is a string because :class:`pathlib.Path` isn't a hashable
    argument shape Streamlit's cache handles well across reruns. The optional
    ``mtime_hint`` lets the caller force a cache miss when the folder has
    changed (otherwise the TTL covers it). Returns string paths (not
    :class:`Path`) so the cached list is still immutable-ish.
    """
    paths = discover_dataset_images(
        Path(directory),
        pattern=pattern,
        max_items=max_items,
        require_nonzero=require_nonzero,
    )
    return [str(p) for p in paths]
