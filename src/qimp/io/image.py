"""Image loading and preparation utilities.

Reference: docs/tesi.pdf §3.1.6 (image_preparation module)
"""

from __future__ import annotations

# TODO(Fase 3): migrate from legacy/scripts/prepare_gp_images.py + qml_and_qimp.py
#   - load_tiff_16bit(path) -> np.ndarray
#   - image_conv(path) -> np.ndarray       # squared, 1-channel check + return pixel array
#   - color_image_conv(path) -> np.ndarray # squared, 3-channel check
#   - apply_filters(channel, sigma, median_size) -> np.ndarray
#   - calculate_gp_image(green, red, G=0.5) -> np.ndarray  # 16-bit GP ratio
