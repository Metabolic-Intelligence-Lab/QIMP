"""Performance monitoring decorator."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

__all__ = ["performance_monitor"]


def performance_monitor(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator that logs wall-clock execution time of `func` at INFO level."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            logger.info("%s ran in %.3fs", func.__qualname__, elapsed)

    return wrapper
