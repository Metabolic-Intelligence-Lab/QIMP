"""Figures of merit for quantum image processing.

Reference: docs/tesi.pdf §3.1.6
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

__all__ = ["mse", "psnr", "total_variation", "transpile_summary"]


def _ensure_float(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return a.astype(np.float64), b.astype(np.float64)


def mse(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Mean squared error between two images.

    Shapes must match. Computed in float64 to avoid overflow on integer images.
    """
    fa, fb = _ensure_float(image_a, image_b)
    return float(np.mean((fa - fb) ** 2))


def psnr(image_a: np.ndarray, image_b: np.ndarray, max_intensity: float | None = None) -> float:
    """Peak signal-to-noise ratio (dB).

    Parameters
    ----------
    image_a, image_b
        Images to compare. Must have the same shape.
    max_intensity
        Peak intensity used in the PSNR formula. If None, inferred from the dtype
        of `image_a` (255 for uint8, 65535 for uint16, 1.0 for float).

    Returns
    -------
    float
        PSNR in decibels; `inf` if the images are identical.
    """
    fa, fb = _ensure_float(image_a, image_b)
    err = float(np.mean((fa - fb) ** 2))
    if err == 0:
        return float("inf")
    if max_intensity is None:
        max_intensity = _infer_max_intensity(image_a)
    return float(20.0 * np.log10(max_intensity / np.sqrt(err)))


def _infer_max_intensity(image: np.ndarray) -> float:
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        return float(info.max)
    return 1.0


def total_variation(image: np.ndarray) -> float:
    """Anisotropic total variation: sum of |∇x| + |∇y| over the image."""
    f = image.astype(np.float64)
    dx = np.diff(f, axis=0, append=f[-1:])
    dy = np.diff(f, axis=1, append=f[:, -1:])
    return float(np.sum(np.sqrt(dx**2 + dy**2)))


def transpile_summary(qc: QuantumCircuit, basis_gates: list[str] | None = None) -> dict[str, Any]:
    """Return depth + op counts before and after transpiling to a basis.

    Useful for characterising scalability: rerun across n to see how depth grows.

    Parameters
    ----------
    qc
        Circuit to summarise.
    basis_gates
        Target basis. Defaults to a typical near-term superconducting basis
        (`u1`, `u2`, `u3`, `cx`).

    Returns
    -------
    dict with keys: depth, ops, depth_transpiled, ops_transpiled.
    """
    from qiskit import transpile

    if basis_gates is None:
        basis_gates = ["u1", "u2", "u3", "cx"]
    transpiled = transpile(qc, basis_gates=basis_gates, optimization_level=1)
    return {
        "depth": qc.depth(),
        "ops": OrderedDict(qc.count_ops()),
        "depth_transpiled": transpiled.depth(),
        "ops_transpiled": OrderedDict(transpiled.count_ops()),
    }
