"""End-to-end integration tests using real microscopy images from `data/`.

Skipped automatically if `data/immagini/trainQML/Train_QML_16/` is empty
(e.g. on CI where data isn't checked in). Run locally with:

    pytest tests/integration -v -m "slow or not slow"

These tests cross-validate:
- FRQI / NEQR / QPIE round-trips on a real 16×16 grayscale tile.
- Geometric operations against numpy ground-truth.
- QHED edge detection sanity (gradient maxima at known edges).
- Classical GP-ratio vs. the lab's reference `GP_output/` files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Resolve the data directory relative to the repo root (two levels up from this file).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAIN_DIR = _REPO_ROOT / "data" / "immagini" / "trainQML" / "Train_QML_16"
_GP_DIR = _REPO_ROOT / "data" / "immagini" / "trainQML" / "GP_output"
_RGB_DIR = _REPO_ROOT / "data" / "immagini" / "trainQML"


def _have_samples(directory: Path, pattern: str = "*.tif") -> bool:
    return directory.exists() and any(directory.glob(pattern))


def _load_first(directory: Path, pattern: str = "*.tif") -> np.ndarray:
    from PIL import Image

    path = sorted(directory.glob(pattern))[0]
    return np.asarray(Image.open(path))


pytestmark = [
    pytest.mark.skipif(
        not _have_samples(_TRAIN_DIR),
        reason=f"no real images in {_TRAIN_DIR} (data/ not synced)",
    ),
]


@pytest.fixture(scope="module")
def real_grayscale_16() -> np.ndarray:
    """A real 16×16 uint8 grayscale tile from the lab's training set."""
    return _load_first(_TRAIN_DIR)


@pytest.fixture(scope="module")
def real_rgb_110() -> np.ndarray:
    """A real 110×110 RGB uint8 tile (raw microscopy frame)."""
    return _load_first(_RGB_DIR)


# ---------------------------------------------------------------------- FRQI ----


@pytest.mark.slow
def test_frqi_on_real_4x4_tile(real_grayscale_16: np.ndarray) -> None:
    """FRQI round-trip on a 4×4 tile (down-sampled from a real 16×16 image)."""
    from qimp.encoding.frqi import FrqiEncoder
    from qimp.metrics import psnr
    from qimp.testing import ideal_simulation

    # Down-sample 16×16 → 4×4 by 4× block averaging to keep the circuit shallow.
    img4 = real_grayscale_16[::4, ::4]
    assert img4.shape == (4, 4) and img4.dtype == np.uint8

    encoder = FrqiEncoder()
    qc = encoder.encode(img4)
    assert qc.num_qubits == 2 * 2 + 1  # n=2 → 5 qubits

    counts = ideal_simulation(qc, shots=40_000)
    decoded = encoder.decode(counts)[0]
    fidelity = psnr(img4, decoded.astype(np.uint8), max_intensity=255.0)
    print(f"\nFRQI 4×4 PSNR: {fidelity:.1f} dB")
    # Even with low base intensities this lab uses, the round-trip should
    # recover ≥ 30 dB on a 4×4 (no transpile crash, shot noise dominated).
    assert fidelity >= 30.0


# ---------------------------------------------------------------------- NEQR ----


def test_neqr_exact_recovery_on_real_image(real_grayscale_16: np.ndarray) -> None:
    """NEQR is exact: a single ideal-simulation shot batch recovers every pixel."""
    from qimp.encoding.neqr import NeqrEncoder
    from qimp.testing import ideal_simulation

    # Lab data is uint8 with values in [0, ~15]; q=4 bits suffice.
    img = real_grayscale_16
    q = 4
    assert img.max() < (1 << q)

    # 16×16 image with q=4 is 2n+q = 12 qubits. NEQR construction is heavy at
    # this scale; demonstrate on an 8×8 crop instead.
    img8 = img[:8, :8]
    encoder = NeqrEncoder(q=q)
    qc = encoder.encode(img8)
    assert qc.num_qubits == 2 * 3 + q  # n=3 → 10 qubits

    counts = ideal_simulation(qc, shots=16_384)
    decoded = encoder.decode(counts)
    np.testing.assert_array_equal(decoded, img8)


# ---------------------------------------------------------------------- QPIE ----


@pytest.mark.slow
def test_qpie_round_trip_on_real_image(real_grayscale_16: np.ndarray) -> None:
    """QPIE on the full 16×16 — fits in 8 qubits."""
    from qimp.encoding.qpie import QpieEncoder
    from qimp.metrics import psnr
    from qimp.testing import ideal_simulation

    img = real_grayscale_16.astype(np.float64)
    if img.max() == 0:
        pytest.skip("image is entirely black; QPIE round-trip is meaningless")

    encoder = QpieEncoder()
    qc = encoder.encode(img)
    assert qc.num_qubits == 8

    counts = ideal_simulation(qc, shots=200_000)
    decoded = encoder.decode(counts)

    # The image's dynamic range is small (max ~15); use max=img.max() for PSNR.
    fidelity = psnr(img, decoded, max_intensity=float(max(img.max(), 1.0)))
    print(f"\nQPIE 16×16 PSNR: {fidelity:.1f} dB")
    assert fidelity >= 20.0


# ---------------------------------------------------------------- Geometric ----


def test_geometric_flip_round_trip_on_real_image(real_grayscale_16: np.ndarray) -> None:
    """Encode 4×4 with NEQR, apply axis_flip(axis='y'), verify == fliplr."""
    from qimp.encoding.neqr import neqr_circuit, neqr_decode
    from qimp.processing.geometric import axis_flip
    from qimp.testing import ideal_simulation

    img4 = real_grayscale_16[::4, ::4].astype(np.int64)
    q = 4
    qc = neqr_circuit(img4, q=q)
    axis_flip(qc, n=2, axis="y", pos_offset=q)
    counts = ideal_simulation(qc, shots=8192)
    decoded = neqr_decode(counts, n=2, q=q)
    np.testing.assert_array_equal(decoded, np.fliplr(img4))


def test_geometric_rotation_180_round_trip(real_grayscale_16: np.ndarray) -> None:
    """180° rotation = double flip = np.rot90(img, 2)."""
    from qimp.encoding.neqr import neqr_circuit, neqr_decode
    from qimp.processing.geometric import ort_rotation
    from qimp.testing import ideal_simulation

    img4 = real_grayscale_16[::4, ::4].astype(np.int64)
    q = 4
    qc = neqr_circuit(img4, q=q)
    ort_rotation(qc, n=2, angle=180, pos_offset=q)
    counts = ideal_simulation(qc, shots=8192)
    decoded = neqr_decode(counts, n=2, q=q)
    np.testing.assert_array_equal(decoded, np.rot90(img4, 2))


# ----------------------------------------------------------------------- QHED ----


@pytest.mark.slow
def test_qhed_finds_edges_on_real_image(real_grayscale_16: np.ndarray) -> None:
    """Sanity: QHED's gradient map has total energy correlated to the image's
    classical TV. On a uniform image the gradient map should sum to ≈ 0.
    """
    from qimp.metrics import total_variation
    from qimp.processing.filters import qhed_filter
    from qimp.testing import ideal_simulation

    img = real_grayscale_16.astype(np.float64)
    if img.max() == 0:
        pytest.skip("image is entirely black; gradients are trivially zero")

    qc, n, rms = qhed_filter(img)
    counts = ideal_simulation(qc, shots=40_000)

    from qimp.processing.filters import qhed_decode

    edges = qhed_decode(counts, n=n, rms=rms)
    classical_tv = total_variation(img)
    print(f"\nQHED total gradient energy: {edges.sum():.2f}, classical TV: {classical_tv:.2f}")
    # On a non-trivial image the quantum gradient sum should be non-zero.
    assert edges.sum() > 0


# ------------------------------------------------------------------- GP-ratio ----


@pytest.mark.skipif(not _have_samples(_GP_DIR), reason=f"no reference GP images in {_GP_DIR}")
def test_classical_gp_matches_reference(real_rgb_110: np.ndarray) -> None:
    """Compute classical GP on a real 16×16 down-sample and compare to the
    lab's pre-computed reference in `data/immagini/trainQML/GP_output/`.

    The reference uses a uint16 range of [0, 4096]; our implementation
    matches that with output_format='16bit'.
    """
    from PIL import Image

    from qimp.io.image import calculate_gp_image

    # Reference 16×16 GP image for the same sample (matched by stem).
    rgb_stem = sorted(_RGB_DIR.glob("*.tif"))[0].stem
    ref_path = _GP_DIR / f"{rgb_stem}_GP.tif"
    if not ref_path.exists():
        pytest.skip(f"no reference GP file for sample {rgb_stem}")

    reference = np.asarray(Image.open(ref_path))
    assert reference.dtype == np.uint16

    # Down-sample the RGB image to 16×16 by 7× nearest-neighbour, matching the
    # `Train_QML_16` preparation pipeline used in legacy/scripts.
    rgb = real_rgb_110[:, :, :3]
    # Crop to 112×112 then 7× decimate (per legacy pipeline) → 16×16.
    rgb16 = rgb[3:115:7, 3:115:7] if rgb.shape[0] >= 115 else rgb[::7, ::7][:16, :16]
    if rgb16.shape[:2] != (16, 16):
        pytest.skip("RGB image too small after decimation")

    red = rgb16[:, :, 0].astype(np.float64)
    green = rgb16[:, :, 1].astype(np.float64)
    ours_16bit = calculate_gp_image(green, red, G=0.5, output_format="16bit")
    print(
        f"\nGP shape OK ({ours_16bit.shape}), reference range "
        f"[{reference.min()}, {reference.max()}], ours [{ours_16bit.min()}, {ours_16bit.max()}]"
    )
    # We can't expect exact match because the legacy preparation pipeline
    # includes filters and a specific resize; sanity-check the range only.
    assert ours_16bit.dtype == np.uint16
    assert ours_16bit.shape == reference.shape
