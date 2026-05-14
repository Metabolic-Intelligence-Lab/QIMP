"""Example 03 — QHED edge detection on a synthetic image.

Runs Quantum Hadamard Edge Detection on a small image containing a sharp
vertical edge. Edges are detected by running QHED on the original image
(horizontal gradients) and on its transpose (vertical gradients), then
summing.

Reproduces the workflow of `docs/tesi.pdf` §3.2.3 with the modern API.

Run:
    python examples/03_qhed_edges.py
"""

from __future__ import annotations

import numpy as np

from qimp.processing.filters import qhed_filter, qhed_full_edges
from qimp.testing import ideal_simulation


def main() -> int:
    # 4x4 image: bright square (intensity 100) on a dark background.
    image = np.zeros((4, 4))
    image[1:3, 1:3] = 100.0
    print("Original image:")
    print(image)

    qc_h, n, _rms = qhed_filter(image)
    qc_v, _, _ = qhed_filter(image.T)
    print(f"\nQHED circuit: {qc_h.num_qubits} qubits (= 2n + 1 = {2 * n + 1})")

    shots = 40_000
    counts_h = ideal_simulation(qc_h, shots=shots)
    counts_v = ideal_simulation(qc_v, shots=shots)

    edges = qhed_full_edges(image, counts_h, counts_v, max_gradient=80.0)
    print("\nEdge map (rows = vertical edges, cols = horizontal edges):")
    print(np.round(edges, 1))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
