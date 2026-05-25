"""Recipe factory — one entry per encoder for the hardware sweep.

A `CircuitRecipe` bundles the quantum circuit, a counts→image decoder,
and the matching classical reference, so the sweep loop is one-shot
per (encoder, n) regardless of encoder family.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image
from qiskit import QuantumCircuit

__all__ = ["CircuitRecipe"]


@dataclass
class CircuitRecipe:
    """One row of the sweep matrix: circuit + decoder + classical reference."""

    label: str
    encoder: str  # "frqi" | "frqi_multi" | "neqr" | "qpie" | "mcrqi" | "ncqi" | "gp"
    n: int
    q: int
    m: int
    qc: QuantumCircuit
    decoder: Callable[[dict[str, int]], np.ndarray]
    reference: np.ndarray


def _downsample_to_n(img: np.ndarray, *, n: int) -> np.ndarray:
    """Lanczos-resample to ``(2^n, 2^n)``. Preserves dtype for uint8.

    Accepts 2D grayscale or 3D RGB(A) — the alpha channel is dropped.
    """
    side = 1 << n
    if img.ndim == 2:
        pil = Image.fromarray(img).resize((side, side), Image.Resampling.LANCZOS)
        return np.asarray(pil, dtype=img.dtype)
    if img.ndim == 3 and img.shape[2] in (3, 4):
        pil = Image.fromarray(img[..., :3], mode="RGB").resize((side, side), Image.Resampling.LANCZOS)
        return np.asarray(pil, dtype=img.dtype)
    raise ValueError(f"unsupported image shape {img.shape}")
