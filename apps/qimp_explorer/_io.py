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
