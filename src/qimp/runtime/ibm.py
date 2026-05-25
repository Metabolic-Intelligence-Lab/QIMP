"""IBM Quantum Runtime helpers — service singleton, backend resolution,
hardware execution via SamplerV2 with TREX + DD mitigation, Aer
noise-model fallback, and per-run artifact persistence.

All IBM-Runtime imports are local to this module: the rest of the
library imports only the typed helpers exposed here.
"""

from __future__ import annotations

__all__: list[str] = []
