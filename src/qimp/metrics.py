"""Figures of merit for quantum image processing.

Reference: docs/tesi.pdf §3.1.6
"""

from __future__ import annotations

# TODO(Fase 3):
#   - mse(a, b) -> float
#   - psnr(a, b, max_intensity=None) -> float    # MAX_I defaults to dtype max
#   - tv(image) -> float                          # total variation
#   - ssim(a, b) -> float                         # via skimage if installed (optional)
#   - transpile_summary(qc, basis_gates=None) -> dict  # depth, op counts before/after
