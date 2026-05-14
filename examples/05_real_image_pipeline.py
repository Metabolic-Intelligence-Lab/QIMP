"""Example 05 — Full QIMP pipeline on a real microscopy image.

Loads a 16×16 grayscale tile from the lab's `Train_QML_16/` folder and runs
every encoder + a geometric op + QHED edge detection. Prints numerical
fidelity at each step **and saves every input/output as a TIFF plus a
side-by-side PNG comparison figure** under `data/output/run_<timestamp>/`.

Requires `data/immagini/trainQML/Train_QML_16/` to exist (gitignored).

Run:
    python examples/05_real_image_pipeline.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = REPO_ROOT / "data" / "immagini" / "trainQML" / "Train_QML_16"
OUTPUT_ROOT = REPO_ROOT / "data" / "output"


def _load_first_real_tile() -> tuple[np.ndarray, str] | None:
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
            return arr, path.stem
    return None


def _save_tiff(img: np.ndarray, path: Path) -> None:
    """Save a 2D array as a TIFF, picking a sensible dtype.

    Float images are renormalized to uint16 [0, 65535]; integer images are
    saved verbatim if uint8/uint16, otherwise clipped to uint16.
    """
    from PIL import Image

    arr = np.asarray(img)
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() > arr.min():
            scaled = np.interp(arr, [arr.min(), arr.max()], [0, 65535]).astype(np.uint16)
        else:
            scaled = np.zeros_like(arr, dtype=np.uint16)
        Image.fromarray(scaled).save(path)
    elif arr.dtype in (np.uint8, np.uint16):
        Image.fromarray(arr).save(path)
    else:
        # Generic integer fallback: clip to uint16.
        scaled = np.clip(arr, 0, 65535).astype(np.uint16)
        Image.fromarray(scaled).save(path)


def _save_comparison_figure(panels: list[tuple[str, np.ndarray]], path: Path) -> None:
    """Save a side-by-side matplotlib figure with one panel per (title, image)."""
    import matplotlib.pyplot as plt

    n = len(panels)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axs = np.atleast_2d(axs)
    for idx, (title, img) in enumerate(panels):
        r, c = divmod(idx, cols)
        ax = axs[r, c]
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    # Hide unused axes.
    for idx in range(len(panels), rows * cols):
        r, c = divmod(idx, cols)
        axs[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    print("Loading a real 16x16 grayscale tile from data/immagini/trainQML/Train_QML_16/...")
    loaded = _load_first_real_tile()
    if loaded is None:
        print(f"  ERROR: no data found in {TRAIN_DIR}", file=sys.stderr)
        print("  (sync OneDrive or run from a populated checkout)", file=sys.stderr)
        return 1
    image, _stem = loaded
    print(
        f"  shape={image.shape}, dtype={image.dtype}, "
        f"range=[{image.min()}, {image.max()}], mean={image.mean():.2f}\n"
    )

    # Output directory: data/output/run_<timestamp>/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to {out_dir}\n")

    panels: list[tuple[str, np.ndarray]] = []

    _save_tiff(image, out_dir / "00_original_16x16.tif")
    panels.append(("Original 16x16", image))

    # ----------------------------------------------------------- FRQI ----
    print("=" * 60)
    print("FRQI on 4x4 down-sample")
    print("=" * 60)
    from qimp.encoding.frqi import FrqiEncoder
    from qimp.metrics import psnr
    from qimp.testing import ideal_simulation

    img4 = image[::4, ::4].astype(np.uint8)
    _save_tiff(img4, out_dir / "01_frqi_input_4x4.tif")
    panels.append(("FRQI input 4x4", img4))

    encoder = FrqiEncoder()
    qc = encoder.encode(img4)
    print(f"  circuit: {qc.num_qubits} qubits, depth pre-transpile = {qc.depth()}")
    counts = ideal_simulation(qc, shots=40_000)
    decoded = encoder.decode(counts)[0]
    _save_tiff(decoded, out_dir / "02_frqi_decoded_4x4.tif")
    panels.append(("FRQI decoded", decoded))
    print(f"  PSNR: {psnr(img4, decoded, max_intensity=255.0):.2f} dB")

    # ----------------------------------------------------------- NEQR ----
    print()
    print("=" * 60)
    print("NEQR on 8x8 crop (exact retrieval)")
    print("=" * 60)
    from qimp.encoding.neqr import NeqrEncoder

    img8 = image[:8, :8].astype(np.int64)
    _save_tiff(img8, out_dir / "03_neqr_input_8x8.tif")
    panels.append(("NEQR input 8x8", img8))

    q = 4
    neqr = NeqrEncoder(q=q)
    qc = neqr.encode(img8)
    print(f"  circuit: {qc.num_qubits} qubits (n=3, q={q})")
    counts = ideal_simulation(qc, shots=16_384)
    decoded_neqr = neqr.decode(counts)
    _save_tiff(decoded_neqr, out_dir / "04_neqr_decoded_8x8.tif")
    panels.append(("NEQR decoded (exact)", decoded_neqr))
    print(f"  byte-for-byte recovery: {np.array_equal(decoded_neqr, img8)}")

    # ----------------------------------------------------------- QPIE ----
    print()
    print("=" * 60)
    print("QPIE on full 16x16 (amplitude encoding, 8 qubits)")
    print("=" * 60)
    from qimp.encoding.qpie import QpieEncoder

    qpie = QpieEncoder()
    qc = qpie.encode(image.astype(np.float64))
    print(f"  circuit: {qc.num_qubits} qubits, depth pre-transpile = {qc.depth()}")
    counts = ideal_simulation(qc, shots=200_000)
    decoded_qpie = qpie.decode(counts)
    _save_tiff(decoded_qpie, out_dir / "05_qpie_decoded_16x16.tif")
    panels.append(("QPIE decoded 16x16", decoded_qpie))
    fidelity = psnr(image, decoded_qpie, max_intensity=float(max(image.max(), 1.0)))
    print(f"  PSNR vs original: {fidelity:.2f} dB")

    # ------------------------------------------------------ Geometric ----
    print()
    print("=" * 60)
    print("Geometric: NEQR + axis_flip(axis='y') vs numpy.fliplr")
    print("=" * 60)
    from qimp.encoding.neqr import neqr_circuit, neqr_decode
    from qimp.processing.geometric import axis_flip, ort_rotation

    qc = neqr_circuit(img4.astype(np.int64), q=q)
    axis_flip(qc, n=2, axis="y", pos_offset=q)
    counts = ideal_simulation(qc, shots=8192)
    flipped = neqr_decode(counts, n=2, q=q)
    _save_tiff(flipped, out_dir / "06_neqr_flipped_y.tif")
    panels.append(("Flip Y axis", flipped))
    print(f"  flipped matches np.fliplr(img): {np.array_equal(flipped, np.fliplr(img4))}")

    qc = neqr_circuit(img4.astype(np.int64), q=q)
    ort_rotation(qc, n=2, angle=180, pos_offset=q)
    counts = ideal_simulation(qc, shots=8192)
    rotated = neqr_decode(counts, n=2, q=q)
    _save_tiff(rotated, out_dir / "07_neqr_rotated_180.tif")
    panels.append(("Rotate 180", rotated))
    print(f"  rotated matches np.rot90(img,2): {np.array_equal(rotated, np.rot90(img4, 2))}")

    # ----------------------------------------------------------- QHED ----
    print()
    print("=" * 60)
    print("QHED edge detection on 16x16")
    print("=" * 60)
    from qimp.metrics import total_variation
    from qimp.processing.filters import qhed_decode, qhed_filter

    qc, n, rms = qhed_filter(image.astype(np.float64))
    print(f"  circuit: {qc.num_qubits} qubits (= 2n+1)")
    counts = ideal_simulation(qc, shots=40_000)
    edges = qhed_decode(counts, n=n, rms=rms)
    _save_tiff(edges, out_dir / "08_qhed_edges_16x16.tif")
    panels.append(("QHED edges", edges))
    classical_tv = total_variation(image.astype(np.float64))
    print(f"  quantum total gradient energy: {edges.sum():.2f}")
    print(f"  classical total variation:     {classical_tv:.2f}")

    # Side-by-side comparison figure.
    comparison_path = out_dir / "comparison.png"
    _save_comparison_figure(panels, comparison_path)
    print()
    print(f"Comparison figure: {comparison_path}")
    print(f"All {len(panels) + 1} stage TIFFs written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
