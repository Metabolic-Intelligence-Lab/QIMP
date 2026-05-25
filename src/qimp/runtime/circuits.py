"""Recipe factory — one entry per encoder for the hardware sweep.

A `CircuitRecipe` bundles the quantum circuit, a counts→image decoder,
and the matching classical reference, so the sweep loop is one-shot
per (encoder, n) regardless of encoder family.
"""

from __future__ import annotations

__all__: list[str] = []
