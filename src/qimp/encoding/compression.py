"""Compression for FRQI and NEQR via Boolean function minimization.

Pixels with the same intensity (FRQI) or intensity bit (NEQR) are grouped,
their position strings converted to Boolean functions, and minimized via
Quine–McCluskey / Espresso. Reduces the number of CNOT/multi-controlled
gates substantially for images with redundant intensity.

Reference: docs/tesi.pdf §2.5
"""

from __future__ import annotations

# TODO(Fase 4): implement
#   class FrqiCompressor:
#       def turn_to_funct(self, image): ...
#       def print_funct(self): ...
#       def compress(self): ...
#       def print_pos_comprsd(self): ...
#       def frqi_image(self) -> QuantumCircuit: ...
#   class NeqrCompressor:  # analogous, per-intensity-bit grouping
