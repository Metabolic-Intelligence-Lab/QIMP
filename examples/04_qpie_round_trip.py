"""Example 04 — QPIE amplitude-encoding round-trip.

QPIE stores pixel intensities directly as amplitude probabilities on 2n qubits.
The reconstructed image differs from the original up to shot noise.

Run:
    python examples/04_qpie_round_trip.py
"""

from __future__ import annotations

import numpy as np

from qimp.encoding.qpie import QpieEncoder
from qimp.metrics import psnr
from qimp.testing import ideal_simulation


def main() -> int:
    image = np.array(
        [
            [10.0, 20.0, 30.0, 40.0],
            [50.0, 60.0, 70.0, 80.0],
            [90.0, 100.0, 110.0, 120.0],
            [130.0, 140.0, 150.0, 160.0],
        ]
    )
    print("Original image:")
    print(image)

    encoder = QpieEncoder()
    qc = encoder.encode(image)
    print(f"\nEncoded into {qc.num_qubits} qubits (n={encoder.n}, depth=1+initialize)")

    counts = ideal_simulation(qc, shots=80_000)
    decoded = encoder.decode(counts)
    print("\nDecoded image:")
    print(np.round(decoded, 1))

    fidelity_psnr = psnr(image, decoded, max_intensity=image.max())
    print(f"\nPSNR = {fidelity_psnr:.2f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
