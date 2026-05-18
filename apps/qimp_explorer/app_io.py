"""I/O helpers shared by the Streamlit pages.

Kept independent of `streamlit` so the functions are unit-testable.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_GRAYSCALE = REPO_ROOT / "data" / "immagini" / "trainQML" / "Train_QML_16"
DATASET_RGB = REPO_ROOT / "data" / "immagini" / "trainQML"
OUTPUT_ROOT = REPO_ROOT / "data" / "output"


def discover_dataset_images(
    directory: Path,
    pattern: str = "*.tif",
    *,
    max_items: int = 50,
    require_nonzero: bool = True,
) -> list[Path]:
    """Return a sorted list of image paths under `directory`.

    By default the search stops after `max_items` and skips files whose pixel
    max is 0 (so the dropdown isn't filled with blank tiles).
    """
    if not directory.exists():
        return []
    from PIL import Image

    kept: list[Path] = []
    for path in sorted(directory.glob(pattern)):
        if len(kept) >= max_items:
            break
        if not require_nonzero:
            kept.append(path)
            continue
        try:
            arr = np.asarray(Image.open(path))
        except (OSError, ValueError):
            continue
        if arr.size > 0 and arr.max() > 0:
            kept.append(path)
    return kept


def load_image(path: Path) -> np.ndarray:
    """Load any image as a numpy array (no resize, no normalization)."""
    from PIL import Image

    return np.asarray(Image.open(path))


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


#: Power-of-two image sizes offered by the resize utility.
RESIZE_OPTIONS: tuple[int, ...] = (2, 4, 8, 16, 32, 64)


def resize_to_square(image: np.ndarray, target_side: int) -> np.ndarray:
    """Resize ``image`` to a ``target_side × target_side`` square.

    - For grayscale (2-D) or RGB / RGBA (3-D) arrays alike.
    - Non-square inputs are center-cropped to a square first, then resized.
    - Down-scaling uses LANCZOS; up-scaling uses NEAREST (preserves discrete
      intensity values for NEQR-style encoders).
    - Preserves the dtype where possible (uint8 / uint16 round-trip cleanly).

    Parameters
    ----------
    image
        2-D ``(H, W)`` or 3-D ``(H, W, C)`` numpy array.
    target_side
        Output side in pixels. Must be a positive power of two.
    """
    from PIL import Image

    if not is_power_of_two(target_side):
        raise ValueError(f"target_side must be a power of two, got {target_side}")
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2-D or 3-D, got shape {image.shape}")

    arr = np.asarray(image)
    h, w = arr.shape[:2]

    # Center-crop to square if needed.
    if h != w:
        s = min(h, w)
        top = (h - s) // 2
        left = (w - s) // 2
        if arr.ndim == 3:
            arr = arr[top : top + s, left : left + s, :]
        else:
            arr = arr[top : top + s, left : left + s]

    current = arr.shape[0]
    if current == target_side:
        return arr

    # PIL needs a specific mode for 16-bit grayscale; everything else is auto.
    if arr.ndim == 2 and arr.dtype == np.uint16:
        pil = Image.fromarray(arr, mode="I;16")
    elif arr.ndim == 2 and np.issubdtype(arr.dtype, np.floating):
        # Float grayscale: rescale to uint16 for resize, then back.
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            arr_u16 = np.interp(arr, [lo, hi], [0, 65535]).astype(np.uint16)
        else:
            arr_u16 = np.zeros_like(arr, dtype=np.uint16)
        pil = Image.fromarray(arr_u16, mode="I;16")
        method = Image.Resampling.LANCZOS if target_side < current else Image.Resampling.NEAREST
        resized_u16 = np.asarray(pil.resize((target_side, target_side), method))
        # Map back to original float range.
        if hi > lo:
            return np.interp(resized_u16, [0, 65535], [lo, hi]).astype(arr.dtype)
        return resized_u16.astype(arr.dtype)
    else:
        pil = Image.fromarray(arr)

    method = Image.Resampling.LANCZOS if target_side < current else Image.Resampling.NEAREST
    return np.asarray(pil.resize((target_side, target_side), method))


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert 3-D RGB/RGBA images to 2-D grayscale via the ITU-R BT.601 luma.

    2-D inputs are returned unchanged. Preserves the dtype where possible.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"can't convert shape {arr.shape} to grayscale")
    r = arr[..., 0].astype(np.float64)
    g = arr[..., 1].astype(np.float64)
    b = arr[..., 2].astype(np.float64)
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return np.clip(luma, info.min, info.max).astype(arr.dtype)
    return luma.astype(arr.dtype)


def infer_n_from_image(image: np.ndarray) -> int | None:
    """Return ``log2(side)`` if `image` is square with a power-of-two side, else None."""
    if image.ndim < 2 or image.shape[0] != image.shape[1]:
        return None
    side = image.shape[0]
    if not is_power_of_two(side):
        return None
    return int(np.log2(side))


def save_tiff(img: np.ndarray, path: Path) -> None:
    """Save a 2D array as TIFF.

    Float images are renormalized to uint16. Integer images of unusual dtype are
    clipped to uint16. uint8/uint16 are written verbatim.
    """
    from PIL import Image

    arr = np.asarray(img)
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() > arr.min():
            scaled = np.interp(arr, [arr.min(), arr.max()], [0, 65535]).astype(np.uint16)
        else:
            scaled = np.zeros_like(arr, dtype=np.uint16)
        Image.fromarray(scaled).save(path)
    elif arr.dtype in (np.uint8, np.uint16):
        Image.fromarray(arr).save(path)
    else:
        Image.fromarray(np.clip(arr, 0, 65535).astype(np.uint16)).save(path)


def new_output_dir(prefix: str = "run") -> Path:
    """Create and return ``data/output/<prefix>_<timestamp>/``."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_ROOT / f"{prefix}_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_named_panels(panels: list[tuple[str, np.ndarray]], directory: Path) -> list[Path]:
    """Save a list of (label, ndarray) pairs as numbered TIFFs in `directory`.

    Returns the list of paths written.
    """
    written: list[Path] = []
    for idx, (label, arr) in enumerate(panels):
        safe_label = "".join(c if c.isalnum() else "_" for c in label).strip("_")
        path = directory / f"{idx:02d}_{safe_label}.tif"
        save_tiff(arr, path)
        written.append(path)
    return written
