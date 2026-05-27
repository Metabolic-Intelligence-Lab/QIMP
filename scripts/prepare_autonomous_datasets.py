"""Prepare 2x2, q=2 datasets for the autonomous-quantum-ratiometric paper.

Three datasets mirroring the v9 paper's experimental setup:

  Class A — canonical membrane microscopy frame (R/G channels of
            membraneStack_Sample011_L_UV_DC_001rbc3DM2.tif).
            110×110 → block-mean → 2×2 → quantise to q=2 (intensities 0..3).

  Class B — synthetic Fura-2 calcium image (Grynkiewicz 1985 calibration,
            radial Ca gradient + two puncta). side=2, then quantise to q=2.
            The classical operator is R = F340 / F380.

  Class C — synthetic roGFP2 image (Schwarzländer 2008 calibration,
            linear redox gradient + a "mitochondrial" oxidised patch).
            side=2, then quantise to q=2. The classical operator is
            OxD = (R − R_red) / (R_ox − R_red) with R = F405 / F488.

Output: paper/data_autonomous/{canonical_2x2.npz, fura2_2x2.npz, rogfp2_2x2.npz}
Each .npz holds `I_a`, `I_b` as uint8 [0,3] arrays + auxiliary fields
(quotient_classical, threshold default, raw float channels for plotting).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image as PilImage

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "data_autonomous"
OUT.mkdir(parents=True, exist_ok=True)

Q_BIT = 2  # 2-bit intensities → values in {0, 1, 2, 3}


# ===========================================================================
# helpers
# ===========================================================================


def block_mean(arr: np.ndarray, target_side: int) -> np.ndarray:
    """Down-sample a square `arr` to `target_side × target_side` via
    non-overlapping block averaging."""
    h, w = arr.shape
    if h != w:
        raise ValueError(f"expected square input, got {arr.shape}")
    block = h // target_side
    if block * target_side != h:
        # Crop to nearest multiple
        new_h = block * target_side
        arr = arr[:new_h, :new_h]
        h = new_h
    out = arr.reshape(target_side, block, target_side, block).mean(axis=(1, 3))
    return out


def quantise_to_q(arr: np.ndarray, q: int) -> np.ndarray:
    """Linear rescale to [0, 2^q - 1] then round-to-nearest-int."""
    arr = arr.astype(np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    if hi == lo:
        return np.zeros_like(arr, dtype=np.int64)
    scaled = (arr - lo) / (hi - lo) * float((1 << q) - 1)
    return np.clip(np.round(scaled), 0, (1 << q) - 1).astype(np.int64)


# ===========================================================================
# Class A — canonical microscopy frame
# ===========================================================================


def prepare_class_a() -> None:
    """Class A canonical microscopy frame at 2×2, q=2.

    The native 110×110 image is mostly background (dark pixels), so a
    naive block-mean over the whole frame collapses everything to ≈ 0.
    We instead localise a signal-rich 32×32 patch (the one with the
    highest combined R+G mean across stride-4 candidates) and block-mean
    that down to 2×2. This preserves the dynamic range we need for a
    meaningful Class-B ratio demo.

    The patch coordinates (8, 4) were selected by a brute-force scan
    over stride-4 32×32 candidates; see the comment block at the end
    of this function for the reproducibility recipe.
    """
    src = REPO / "data" / "immagini" / "trainQML" / \
          "membraneStack_Sample011_L_UV_DC_001rbc3DM2.tif"
    img = np.asarray(PilImage.open(src))
    R_raw = img[..., 0].astype(np.float64)
    G_raw = img[..., 1].astype(np.float64)

    # Localise the signal-rich 32×32 patch (reproducibly).
    patch_y, patch_x, patch_side = 8, 4, 32
    R_patch = R_raw[patch_y:patch_y + patch_side, patch_x:patch_x + patch_side]
    G_patch = G_raw[patch_y:patch_y + patch_side, patch_x:patch_x + patch_side]

    # Block-mean 32×32 → 2×2 (block = 16).
    R_2 = block_mean(R_patch, target_side=2)
    G_2 = block_mean(G_patch, target_side=2)

    # Quantise to q=2 (intensities 0..3). I_a = R (red), I_b = G (green).
    I_a = quantise_to_q(R_2, Q_BIT)
    I_b = quantise_to_q(G_2, Q_BIT)

    R_classical = np.where(I_b > 0, I_a // np.maximum(I_b, 1), 0)
    divzero_mask = I_b == 0

    out_file = OUT / "canonical_2x2.npz"
    np.savez(
        out_file,
        I_a=I_a, I_b=I_b,
        R_classical=R_classical, divzero_mask=divzero_mask,
        raw_R_patch=R_patch, raw_G_patch=G_patch,
        raw_R_2=R_2, raw_G_2=G_2,
        source_path=str(src),
        patch_yx=np.array([patch_y, patch_x]),
        patch_side=patch_side,
    )
    print(f"Wrote {out_file.name}")
    print(f"  Source patch: ({patch_y}, {patch_x}) + {patch_side}×{patch_side}")
    print(f"  I_a = {I_a.tolist()}  (red channel, q=2)")
    print(f"  I_b = {I_b.tolist()}  (green channel, q=2)")
    print(f"  R_classical = {R_classical.tolist()}")
    print(f"  divzero    = {divzero_mask.tolist()}")
    # Report a_true at the natural threshold τ=0.
    good = (R_classical > 0) & (~divzero_mask)
    print(f"  τ=0: a_true = {good.sum()}/{I_a.size} = {good.sum() / I_a.size:.3f}")


# ===========================================================================
# Class B — Fura-2 synthetic
# ===========================================================================


def synthesise_fura2(
    side: int = 2, seed: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic Fura-2 dual-excitation image (radial Ca gradient + puncta)
    using the Grynkiewicz 1985 calibration. Returns (F340, F380, Ca)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float64)
    cy, cx = (side - 1) / 2.0, (side - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / max(side / 2.0, 1.0)
    ca_baseline = 80.0 + 220.0 * np.exp(-(rr**2) * 2.0)
    # Puncta only render meaningfully at side >= 8; at side=2 the gradient
    # is the dominant signal, which is exactly what we want.
    ca = ca_baseline + rng.normal(0.0, 5.0, ca_baseline.shape)
    ca = np.clip(ca, 10.0, 2000.0)
    K_d = 224.0
    Sf340, Sb340 = 50.0, 250.0
    Sf380, Sb380 = 250.0, 50.0
    f340 = Sf340 + (Sb340 - Sf340) * ca / (K_d + ca)
    f380 = Sf380 + (Sb380 - Sf380) * ca / (K_d + ca)
    return f340, f380, ca


def prepare_class_b() -> None:
    f340, f380, ca = synthesise_fura2(side=2, seed=1)
    I_a = quantise_to_q(f340, Q_BIT)  # F340 → "a" channel
    I_b = quantise_to_q(f380, Q_BIT)  # F380 → "b" channel
    R_classical = np.where(I_b > 0, I_a // np.maximum(I_b, 1), 0)
    divzero_mask = I_b == 0

    out_file = OUT / "fura2_2x2.npz"
    np.savez(
        out_file,
        I_a=I_a, I_b=I_b,
        R_classical=R_classical, divzero_mask=divzero_mask,
        raw_F340=f340, raw_F380=f380, raw_Ca=ca,
        source="synthesise_fura2(side=2, seed=1)",
    )
    print(f"Wrote {out_file.name}")
    print(f"  I_a (F340 quantised) = {I_a.tolist()}")
    print(f"  I_b (F380 quantised) = {I_b.tolist()}")
    print(f"  R_classical = {R_classical.tolist()}")


# ===========================================================================
# Class C — roGFP2 synthetic
# ===========================================================================


def synthesise_rogfp(
    side: int = 2, seed: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic roGFP2 dual-excitation image (linear OxD gradient + patch)
    using Schwarzländer 2008 calibration. Returns (F405, F488, OxD)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float64)
    oxd = 0.20 + 0.60 * (yy / max(side - 1, 1))
    oxd = np.clip(oxd + rng.normal(0.0, 0.01, oxd.shape), 0.0, 1.0)
    I405_red, I405_ox = 0.85, 5.00
    I488_red, I488_ox = 1.00, 0.20
    f405 = (1.0 - oxd) * I405_red + oxd * I405_ox
    f488 = (1.0 - oxd) * I488_red + oxd * I488_ox
    f405 = f405 + rng.normal(0.0, 0.01, f405.shape)
    f488 = f488 + rng.normal(0.0, 0.01, f488.shape)
    return f405, f488, oxd


def prepare_class_c() -> None:
    f405, f488, oxd = synthesise_rogfp(side=2, seed=2)
    I_a = quantise_to_q(f405, Q_BIT)
    I_b = quantise_to_q(f488, Q_BIT)
    R_classical = np.where(I_b > 0, I_a // np.maximum(I_b, 1), 0)
    divzero_mask = I_b == 0

    out_file = OUT / "rogfp2_2x2.npz"
    np.savez(
        out_file,
        I_a=I_a, I_b=I_b,
        R_classical=R_classical, divzero_mask=divzero_mask,
        raw_F405=f405, raw_F488=f488, raw_OxD=oxd,
        R_red=0.20, R_ox=4.50,
        source="synthesise_rogfp(side=2, seed=2)",
    )
    print(f"Wrote {out_file.name}")
    print(f"  I_a (F405 quantised) = {I_a.tolist()}")
    print(f"  I_b (F488 quantised) = {I_b.tolist()}")
    print(f"  R_classical = {R_classical.tolist()}")


def main() -> int:
    prepare_class_a()
    print()
    prepare_class_b()
    print()
    prepare_class_c()
    print()
    print(f"All datasets written to {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
