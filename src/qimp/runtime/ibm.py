"""IBM Quantum Runtime helpers — service singleton, backend resolution,
hardware execution via SamplerV2 with TREX + DD mitigation, Aer
noise-model fallback, and per-run artifact persistence.

All IBM-Runtime imports are local to this module: the rest of the
library imports only the typed helpers exposed here.
"""

# ruff: noqa: F822
from __future__ import annotations

__all__ = [
    "aer_noisy_run",
    "get_service",
    "hw_run",
    "is_run_complete",
    "list_backends",
    "persist_run",
    "pick_backend",
]
