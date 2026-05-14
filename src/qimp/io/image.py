"""Image loading and preparation utilities.

Reference: docs/tesi.pdf §3.1.6 (image_preparation module)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter

__all__ = [
    "apply_filters",
    "calculate_gp_image",
    "color_image_conv",
    "image_conv",
    "load_tiff_16bit",
]


def load_tiff_16bit(path: str | Path) -> np.ndarray:
    """Load a TIFF file as a numpy array, preserving 16-bit depth if present.

    Returns the raw pixel array (no normalization, no resize).
    """
    img = Image.open(Path(path))
    return np.asarray(img)


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def image_conv(
    path: str | Path, *, require_square: bool = True, require_pow2: bool = True
) -> np.ndarray:
    """Open a grayscale image and verify it is square with side a power of two.

    Returns the (H, W) pixel array. Converts to grayscale (L mode) if needed.

    Raises
    ------
    ValueError
        If the image is not square (when `require_square`) or its side is not a
        power of two (when `require_pow2`). These constraints come from the
        encoding requirements: image side must be 2^n.
    """
    base = Image.open(Path(path))
    img: Image.Image = base if base.mode in ("L", "I;16", "I") else base.convert("L")
    arr = np.asarray(img)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {arr.shape}")
    h, w = arr.shape
    if require_square and h != w:
        raise ValueError(f"Image is not square: {h}x{w}")
    if require_pow2 and not _is_power_of_two(h):
        raise ValueError(f"Image side {h} is not a power of two")
    return arr


def color_image_conv(
    path: str | Path, *, require_square: bool = True, require_pow2: bool = True
) -> np.ndarray:
    """Open an RGB image and verify it is square with side a power of two.

    Alpha channel, if present, is discarded. Returns shape (H, W, 3).
    """
    base = Image.open(Path(path))
    img: Image.Image = base if base.mode == "RGB" else base.convert("RGB")
    arr = np.asarray(img)
    h, w, c = arr.shape
    if c != 3:
        raise ValueError(f"Expected 3 channels, got {c}")
    if require_square and h != w:
        raise ValueError(f"Image is not square: {h}x{w}")
    if require_pow2 and not _is_power_of_two(h):
        raise ValueError(f"Image side {h} is not a power of two")
    return arr


def apply_filters(
    channel: np.ndarray,
    sigma: float = 1.0,
    median_size: int = 3,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Apply Gaussian then median filter to a single image channel.

    Parameters
    ----------
    channel
        2D image to filter.
    sigma
        Gaussian filter sigma.
    median_size
        Square median kernel side.
    out
        Optional pre-allocated output buffer.
    """
    if out is not None:
        gaussian_filter(channel, sigma=sigma, output=out)
        median_filter(out, size=median_size, output=out)
        return out
    filtered = gaussian_filter(channel, sigma=sigma)
    result: np.ndarray = median_filter(filtered, size=median_size)
    return result


def calculate_gp_image(
    green: np.ndarray,
    red: np.ndarray,
    G: float = 0.5,
    output_format: str = "normalized",
) -> np.ndarray:
    """Green-Purple ratio image: (G - α·R) / (G + α·R).

    Parameters
    ----------
    green, red
        Channel arrays with matching shape.
    G
        Weight applied to the red channel (called α in some literature).
    output_format
        - "normalized": float array in [-1, 1]
        - "16bit": uint16 array in [0, 4096]
        - "uint8": uint8 array in [0, 255]
    """
    if green.shape != red.shape:
        raise ValueError(f"Channel shape mismatch: {green.shape} vs {red.shape}")
    g = green.astype(np.float64)
    r = red.astype(np.float64)
    denom = g + G * r
    safe_denom = denom + 1e-10
    out = (g - G * r) / safe_denom
    out = np.clip(out, -1.0, 1.0)
    out[denom == 0] = 0.0
    if output_format == "normalized":
        return out
    if output_format == "16bit":
        return np.interp(out, [-1.0, 1.0], [0, 4096]).astype(np.uint16)
    if output_format == "uint8":
        return np.interp(out, [-1.0, 1.0], [0, 255]).astype(np.uint8)
    raise ValueError(f"Unknown output_format: {output_format!r}")
