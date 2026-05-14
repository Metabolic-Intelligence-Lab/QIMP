"""Example 05 — Full QIMP pipeline on a real microscopy image.

Loads a 16×16 grayscale tile from the lab's `Train_QML_16/` folder and runs
every encoder + a geometric op + QHED edge detection. Prints numerical
fidelity at each step.

Requires `data/immagini/trainQML/Train_QML_16/` to exist (gitignored).

Run:
    python examples/05_real_image_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = REPO_ROOT / "data" / "immagini" / "trainQML" / "Train_QML_16"


def _load_first_real_tile() -> np.ndarray | None:
    if not TRAIN_DIR.exists():
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    files = sorted(TRAIN_DIR.glob("*.tif"))
    if not files:
        return None
    # Pick the first file with non-zero content so the demo isn't a wall of zeros.
    for path in files[:50]:
        arr = np.asarray(Image.open(path))
        if arr.max() > 0:
            print(f"  loaded: {path.name}")
            return arr
    return None


def main() -> int:
    print("Loading a real 16×16 grayscale tile from data/immagini/trainQML/Train_QML_16/...")
    image = _load_first_real_tile()
    if image is None:
        print(f"  ERROR: no data found in {TRAIN_DIR}", file=sys.stderr)
        print("  (sync OneDrive or run from a populated checkout)", file=sys.stderr)
        return 1
    print(
        f"  shape={image.shape}, dtype={image.dtype}, "
        f"range=[{image.min()}, {image.max()}], mean={image.mean():.2f}\n"
    )

    # ----------------------------------------------------------- FRQI ----
    print("=" * 60)
    print("FRQI on 4×4 down-sample")
    print("=" * 60)
    from qimp.encoding.frqi import FrqiEncoder
    from qimp.metrics import psnr
    from qimp.testing import ideal_simulation

    img4 = image[::4, ::4].astype(np.uint8)
    encoder = FrqiEncoder()
    qc = encoder.encode(img4)
    print(f"  circuit: {qc.num_qubits} qubits, depth pre-transpile = {qc.depth()}")
    counts = ideal_simulation(qc, shots=40_000)
    decoded = encoder.decode(counts)[0]
    print(f"  original:    {img4.flatten().tolist()}")
    print(f"  decoded:     {np.round(decoded.flatten(), 1).tolist()}")
    print(f"  PSNR: {psnr(img4, decoded, max_intensity=255.0):.2f} dB")

    # ----------------------------------------------------------- NEQR ----
    print()
    print("=" * 60)
    print("NEQR on 8×8 crop (exact retrieval)")
    print("=" * 60)
    from qimp.encoding.neqr import NeqrEncoder

    img8 = image[:8, :8].astype(np.int64)
    q = 4
    neqr = NeqrEncoder(q=q)
    qc = neqr.encode(img8)
    print(f"  circuit: {qc.num_qubits} qubits (n=3, q={q})")
    counts = ideal_simulation(qc, shots=16_384)
    decoded = neqr.decode(counts)
    print(f"  byte-for-byte recovery: {np.array_equal(decoded, img8)}")

    # ----------------------------------------------------------- QPIE ----
    print()
    print("=" * 60)
    print("QPIE on full 16×16 (amplitude encoding, 8 qubits)")
    print("=" * 60)
    from qimp.encoding.qpie import QpieEncoder

    qpie = QpieEncoder()
    qc = qpie.encode(image.astype(np.float64))
    print(f"  circuit: {qc.num_qubits} qubits, depth pre-transpile = {qc.depth()}")
    counts = ideal_simulation(qc, shots=200_000)
    decoded = qpie.decode(counts)
    fidelity = psnr(image, decoded, max_intensity=float(max(image.max(), 1.0)))
    print(f"  PSNR vs original: {fidelity:.2f} dB")

    # ------------------------------------------------------ Geometric ----
    print()
    print("=" * 60)
    print("Geometric: NEQR + axis_flip(axis='y') vs numpy.fliplr")
    print("=" * 60)
    from qimp.encoding.neqr import neqr_circuit, neqr_decode
    from qimp.processing.geometric import axis_flip

    qc = neqr_circuit(img4.astype(np.int64), q=q)
    axis_flip(qc, n=2, axis="y", pos_offset=q)
    counts = ideal_simulation(qc, shots=8192)
    flipped = neqr_decode(counts, n=2, q=q)
    print(f"  flipped matches np.fliplr(img): {np.array_equal(flipped, np.fliplr(img4))}")

    # ----------------------------------------------------------- QHED ----
    print()
    print("=" * 60)
    print("QHED edge detection on 16×16")
    print("=" * 60)
    from qimp.metrics import total_variation
    from qimp.processing.filters import qhed_decode, qhed_filter

    qc, n, rms = qhed_filter(image.astype(np.float64))
    print(f"  circuit: {qc.num_qubits} qubits (= 2n+1)")
    counts = ideal_simulation(qc, shots=40_000)
    edges = qhed_decode(counts, n=n, rms=rms)
    classical_tv = total_variation(image.astype(np.float64))
    print(f"  quantum total gradient energy: {edges.sum():.2f}")
    print(f"  classical total variation:     {classical_tv:.2f}")
    print("  edge map (rounded):")
    for row in np.round(edges, 1):
        print("   ", row.tolist())

    print()
    print("All pipeline steps completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
