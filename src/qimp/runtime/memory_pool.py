"""Pre-allocated buffer pool for batch image processing.

Avoids repeated `np.zeros((H, W))` allocations when processing large datasets
of same-shape images. Buffers are reused round-robin.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["MemoryPool", "clear_memory_pool", "get_memory_pool"]


class MemoryPool:
    """Round-robin pool of (H, W) image buffers and flattened angle buffers.

    Parameters
    ----------
    max_buffers
        How many buffers to pre-allocate.
    image_size
        Side length of square buffers. Image buffer shape = (image_size, image_size),
        angle buffer shape = (image_size * image_size,).
    dtype
        Dtype of the buffers (default float32 — half the RAM of float64).

    Notes
    -----
    No upper bound on `image_size`. With float32 and image_size=2^n × 2^n,
    one buffer costs `4 * 4^n` bytes. A typical pool with 50 buffers at
    n=7 (128×128) uses ~3.2 MB; at n=10 (1024×1024) it uses ~200 MB.
    """

    def __init__(self, max_buffers: int, image_size: int, dtype: type = np.float32) -> None:
        if max_buffers < 1:
            raise ValueError("max_buffers must be >= 1")
        if image_size < 1:
            raise ValueError("image_size must be >= 1")
        self.max_buffers = max_buffers
        self.image_size = image_size
        self.dtype = dtype
        self._image_pool: np.ndarray = np.zeros((max_buffers, image_size, image_size), dtype=dtype)
        self._angle_pool: np.ndarray = np.zeros((max_buffers, image_size * image_size), dtype=dtype)
        self._cursor = 0
        logger.debug(
            "MemoryPool: %d buffers of %dx%d (dtype=%s)",
            max_buffers,
            image_size,
            image_size,
            dtype,
        )

    def get_image_buffer(self) -> np.ndarray:
        """Return the next zeroed (image_size, image_size) buffer.

        Advances the round-robin cursor: consecutive calls return distinct
        buffers, wrapping around after `max_buffers` calls.
        """
        buf: np.ndarray = self._image_pool[self._cursor % self.max_buffers]
        buf.fill(0)
        self._cursor += 1
        return buf

    def get_angle_buffer(self) -> np.ndarray:
        """Return the next zeroed flat (image_size**2,) buffer.

        Advances the round-robin cursor like `get_image_buffer`.
        """
        buf: np.ndarray = self._angle_pool[self._cursor % self.max_buffers]
        buf.fill(0)
        self._cursor += 1
        return buf


_global_pool: MemoryPool | None = None


def get_memory_pool(image_size: int = 16, max_buffers: int = 100) -> MemoryPool:
    """Return the process-wide pool, creating or resizing it as needed."""
    global _global_pool
    if _global_pool is None or _global_pool.image_size != image_size:
        _global_pool = MemoryPool(max_buffers, image_size)
    return _global_pool


def clear_memory_pool() -> None:
    """Drop the process-wide pool. Useful between large batches to release RAM."""
    global _global_pool
    _global_pool = None
