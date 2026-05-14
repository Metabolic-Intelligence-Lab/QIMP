"""Example 01 — FRQI fidelity round-trip.

Encode a small 4x4 grayscale image with FRQI, measure on an ideal simulator,
decode back into an image, and report PSNR.

Reproduces the workflow of `docs/tesi.pdf` §3.2.1 with the modern API.

Run:
    python examples/01_frqi_fidelity.py
"""

from __future__ import annotations

import numpy as np

from qimp.encoding.frqi import FrqiEncoder
from qimp.metrics import mse, psnr
from qimp.testing import ideal_simulation


def main() -> int:
    rng = np.random.default_rng(seed=0)
    # 4x4 image of 8-bit intensities.
    image = rng.integers(0, 256, size=(4, 4), dtype=np.uint8)
    print("Original image:")
    print(image)

    encoder = FrqiEncoder()
    qc = encoder.encode(image)
    print(
        f"\nEncoded into {qc.num_qubits} qubits "
        f"(n={encoder.n}, m={encoder.m}, depth before transpile={qc.depth()})"
    )

    counts = ideal_simulation(qc, shots=40_000)
    decoded = encoder.decode(counts)[0]
    print("\nDecoded image (float):")
    print(np.round(decoded, 1))

    fidelity_psnr = psnr(image, decoded)
    fidelity_mse = mse(image, decoded)
    print(f"\nMSE  = {fidelity_mse:.3f}")
    print(f"PSNR = {fidelity_psnr:.2f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
