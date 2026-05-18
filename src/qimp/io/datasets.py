"""Folder-based dataset iterators for batch image processing."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from qimp.io.image import color_image_conv, image_conv

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["ImageDataset", "batch_process_images"]


class ImageDataset:
    """Iterate over images in a folder.

    Parameters
    ----------
    root
        Directory containing the images.
    pattern
        Glob pattern (default ``*.tif``).
    color
        ``"L"`` (grayscale via `image_conv`) or ``"RGB"`` (via `color_image_conv`).
        Defaults to ``"L"``.
    require_square
        Forwarded to the loader; rejects non-square images when True.
    require_pow2
        Forwarded to the loader; rejects non-power-of-two sides when True.
    skip_blank
        If True (default) skip images whose pixel max is 0.
    limit
        Maximum number of files to enumerate. ``None`` = all.

    Examples
    --------
    >>> ds = ImageDataset("data/immagini/trainQML/Train_QML_16", color="L")
    >>> for path, arr in ds:
    ...     pass  # process each (path, image)
    """

    def __init__(
        self,
        root: str | Path,
        *,
        pattern: str = "*.tif",
        color: str = "L",
        require_square: bool = True,
        require_pow2: bool = True,
        skip_blank: bool = True,
        limit: int | None = None,
    ) -> None:
        if color not in ("L", "RGB"):
            raise ValueError(f"color must be 'L' or 'RGB', got {color!r}")
        self.root = Path(root)
        self.pattern = pattern
        self.color = color
        self.require_square = require_square
        self.require_pow2 = require_pow2
        self.skip_blank = skip_blank
        self.limit = limit

    def paths(self) -> list[Path]:
        """List image paths matching the pattern (sorted, capped to `limit`)."""
        if not self.root.exists():
            return []
        items = sorted(self.root.glob(self.pattern))
        return items[: self.limit] if self.limit is not None else items

    def __len__(self) -> int:
        return len(self.paths())

    def __iter__(self) -> Iterator[tuple[Path, np.ndarray]]:
        for path in self.paths():
            try:
                if self.color == "L":
                    arr = image_conv(
                        path,
                        require_square=self.require_square,
                        require_pow2=self.require_pow2,
                    )
                else:
                    arr = color_image_conv(
                        path,
                        require_square=self.require_square,
                        require_pow2=self.require_pow2,
                    )
            except (OSError, ValueError):
                continue
            if self.skip_blank and arr.size > 0 and arr.max() == 0:
                continue
            yield path, arr


def batch_process_images(
    dataset: ImageDataset | Sequence[tuple[Path, np.ndarray]],
    fn: Callable[[np.ndarray], object],
    *,
    on_error: str = "skip",
) -> list[tuple[Path, object]]:
    """Apply `fn` to every image in `dataset` and return [(path, result), …].

    Parameters
    ----------
    dataset
        An `ImageDataset` instance or any iterable of ``(path, image)`` tuples.
    fn
        Callable invoked as ``fn(image)`` for each image.
    on_error
        ``"skip"`` (default) silently drops images that raise; ``"raise"``
        re-raises the first exception.

    Returns
    -------
    list[tuple[Path, object]]
        Pairs of source path and the value returned by `fn`.
    """
    if on_error not in ("skip", "raise"):
        raise ValueError(f"on_error must be 'skip' or 'raise', got {on_error!r}")
    results: list[tuple[Path, object]] = []
    for path, arr in dataset:
        try:
            results.append((path, fn(arr)))
        except Exception:
            if on_error == "raise":
                raise
            continue
    return results
