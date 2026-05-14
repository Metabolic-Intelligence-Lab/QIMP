"""Example 02 — NEQR exact round-trip.

NEQR stores intensity in a basis-state register, so encoding then measuring
recovers the image **exactly** (a few shots are enough to read each pixel).

Run:
    python examples/02_neqr_round_trip.py
"""

from __future__ import annotations

import numpy as np

from qimp.encoding.neqr import NeqrEncoder
from qimp.testing import ideal_simulation


def main() -> int:
    image = np.array(
        [
            [10, 20, 30, 40],
            [50, 60, 70, 80],
            [90, 100, 110, 120],
            [130, 140, 150, 160],
        ],
        dtype=np.int64,
    )
    print("Original image:")
    print(image)

    encoder = NeqrEncoder(q=8)
    qc = encoder.encode(image)
    print(f"\nEncoded into {qc.num_qubits} qubits (n={encoder.n}, q={encoder.q})")

    counts = ideal_simulation(qc, shots=8192)
    decoded = encoder.decode(counts)
    print("\nDecoded image:")
    print(decoded)

    if np.array_equal(decoded, image):
        print("\n✓ Exact round-trip recovered.")
    else:
        diff = decoded - image
        print(f"\n✗ Mismatch detected, max diff = {np.abs(diff).max()}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
