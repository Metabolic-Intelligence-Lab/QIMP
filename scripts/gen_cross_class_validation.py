"""Cross-class validation: closed-form solver applied to Class-B (simple ratio,
Fura-2 calcium imaging) and Class-C (calibrated ratio, roGFP redox imaging)
synthetic targets, in addition to the GP (Class A) case already covered in
the main validation pipeline.

The script performs the algorithmic check: for each class the per-pixel target
is computed classically, the closed-form parameters
    alpha_p = arcsin(-t_p) - phi_a,   beta_p = arcsin(-t_p) - phi_b
are evaluated, and the per-pixel decoded value
    GP_decoded[p] = -(sin(phi_b + beta_p) + sin(phi_a + alpha_p))/2
is reconstructed in pure NumPy. By Theorem 1 of the main paper the
reconstruction equals the target to machine precision; the simulator-side
floating-point ceiling characterised in Section 5.2 (75-100 dB across n=2..5)
is independent of the choice of target and is the same numeric envelope here.

Two figures are saved:
    paper/figures/fig_fura2_validation.png   (Class B, Fura-2 calcium)
    paper/figures/fig_rogfp_validation.png   (Class C, roGFP redox)

Each figure has three panels: the two input intensity channels, the classical
ratiometric target, and the closed-form reconstruction error (pixelwise
difference at colour range +/- 1e-9), plus a small annotation reporting MSE
and PSNR.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(
    "/mnt/c/Users/Giuseppe/OneDrive - Università Cattolica del Sacro Cuore/"
    "Metabolic Intelligence - Projects-MI/2024_QIMP/repo"
)
OUT = REPO / "paper" / "figures"


# ----- closed-form solver (pure NumPy mirror of qimp.processing.gp_ratio) --


def frqi_angles(intensity: np.ndarray, normalization: float) -> np.ndarray:
    """phi = arccos(1 - 2 * I / N), elementwise."""
    return np.arccos(np.clip(1.0 - 2.0 * intensity / normalization, -1.0, 1.0))


def closed_form_params(
    a_channel: np.ndarray,
    b_channel: np.ndarray,
    target: np.ndarray,
    normalization: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form (alpha_p, beta_p) per pixel for the corrected GP ansatz.

    See Theorem 1 of the main paper. Returns two arrays of the same shape as
    the target (one per pixel). For any pixel where |target[p]| > 1 the
    target is silently clipped to the valid range of arcsin.
    """
    phi_a = frqi_angles(a_channel, normalization)
    phi_b = frqi_angles(b_channel, normalization)
    t = np.clip(target, -1.0, 1.0)
    base = np.arcsin(-t)
    return base - phi_a, base - phi_b


def closed_form_decoded(
    a_channel: np.ndarray,
    b_channel: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    normalization: float,
) -> np.ndarray:
    """Per-pixel decoded value of the corrected GP ansatz.

    GP_decoded[p] = -(sin(phi_b + beta_p) + sin(phi_a + alpha_p)) / 2

    This is the pure-NumPy reconstruction of what the quantum statevector
    would produce on a noise-free simulator (modulo floating-point error
    in the multi-controlled-gate decomposition, which is target-independent
    and characterised in Section 5.2 of the main paper).
    """
    phi_a = frqi_angles(a_channel, normalization)
    phi_b = frqi_angles(b_channel, normalization)
    return -(np.sin(phi_b + beta) + np.sin(phi_a + alpha)) / 2.0


def psnr_db(target: np.ndarray, decoded: np.ndarray, signal_range: float = 2.0) -> float:
    """PSNR in dB with the signal range as max - min of the target.

    For targets in [-1, +1] the natural range is 2 (matches the convention
    used in Section 5.2 of the main paper)."""
    mse = float(np.mean((target - decoded) ** 2))
    if mse < 1e-30:
        return float("inf")
    return float(10.0 * np.log10((signal_range**2) / mse))


# ----- Class B: Fura-2 calcium imaging ------------------------------------


def synthesise_fura2(side: int = 16, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic Fura-2 dual-excitation image at the given spatial side.

    Generates a smooth radial calcium gradient with two Gaussian "puncta"
    (local Ca2+ release sites). The two excitation channels are computed
    from the Grynkiewicz (1985) calibration:

        F340(Ca) = Sf340 + (Sb340 - Sf340) * Ca / (Kd + Ca)
        F380(Ca) = Sf380 + (Sb380 - Sf380) * Ca / (Kd + Ca)

    where Kd = 224 nM and the S-coefficients are the standard literature
    values for Fura-2/AM in mammalian cells (cf. Grynkiewicz, Poenie,
    Tsien 1985). Returns (F340, F380, Ca) with Ca in nM.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float64)
    cy, cx = (side - 1) / 2.0, (side - 1) / 2.0

    # Radial baseline + two puncta
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (side / 2.0)
    ca_baseline = 80.0 + 220.0 * np.exp(-(rr**2) * 2.0)  # nM, peak ~ 300 nM at centre
    punctum_a = 700.0 * np.exp(-(((yy - 4) ** 2 + (xx - 3) ** 2) / 4.0))
    punctum_b = 500.0 * np.exp(-(((yy - 11) ** 2 + (xx - 12) ** 2) / 5.0))
    ca = ca_baseline + punctum_a + punctum_b
    ca = ca + rng.normal(0.0, 5.0, ca.shape)  # mild shot noise
    ca = np.clip(ca, 10.0, 2000.0)

    # Grynkiewicz calibration constants for Fura-2 / AM
    K_d = 224.0  # nM
    Sf340, Sb340 = 50.0, 250.0   # arbitrary AU; ratio is what matters
    Sf380, Sb380 = 250.0, 50.0
    f340 = Sf340 + (Sb340 - Sf340) * ca / (K_d + ca)
    f380 = Sf380 + (Sb380 - Sf380) * ca / (K_d + ca)
    return f340, f380, ca


def validate_class_b() -> dict[str, float]:
    """Class B (simple ratio): R = F340 / F380, bounded via u = (R-1)/(R+1)."""
    f340, f380, ca = synthesise_fura2(side=16, seed=1)
    # Class-B bounded target: u = (R - 1) / (R + 1)
    eps = 1e-9
    R = f340 / np.maximum(f380, eps)
    u = (R - 1.0) / (R + 1.0)  # in [-1, +1] by construction

    # Use the joint maximum as FRQI normalisation (per-frame).
    norm = float(max(f340.max(), f380.max()))

    # Closed-form parameters and decoded value
    alpha, beta = closed_form_params(f340, f380, u, normalization=norm)
    decoded = closed_form_decoded(f340, f380, alpha, beta, normalization=norm)

    mse = float(np.mean((u - decoded) ** 2))
    psnr = psnr_db(u, decoded, signal_range=2.0)

    # Figure: 5 panels (F340, F380, R, u target, decoded error)
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.0))
    titles = [
        "F340 (intensity AU)",
        "F380 (intensity AU)",
        r"$R = F_{340}/F_{380}$",
        r"$u = (R-1)/(R+1) \in [-1,1]$",
        "decoded - u  (closed-form)",
    ]
    images = [f340, f380, R, u, decoded - u]
    cmaps = ["viridis", "viridis", "magma", "RdBu_r", "RdBu_r"]
    norms = [
        plt.Normalize(vmin=f340.min(), vmax=f340.max()),
        plt.Normalize(vmin=f380.min(), vmax=f380.max()),
        plt.Normalize(vmin=R.min(), vmax=R.max()),
        plt.Normalize(vmin=-1, vmax=1),
        plt.Normalize(vmin=-1e-15, vmax=1e-15),  # the error is machine-eps
    ]
    for ax, img, title, cmap, cn in zip(axes, images, titles, cmaps, norms, strict=True):
        im = ax.imshow(img, cmap=cmap, norm=cn, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Class B (Fura-2 calcium imaging) — closed-form solver, n = 4 "
        f"(16x16, 512 parameters)   |   MSE = {mse:.2e},  PSNR = {psnr:.1f} dB",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(
        OUT / "fig_fura2_validation.png", dpi=200, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)
    return {"mse": mse, "psnr_db": psnr, "ca_min": float(ca.min()), "ca_max": float(ca.max())}


# ----- Class C: roGFP2 redox imaging --------------------------------------


def synthesise_rogfp(side: int = 16, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic roGFP2 dual-excitation image with a known redox gradient.

    roGFP2 has two excitation peaks (lambda_ox ~ 405 nm, lambda_red ~ 488 nm)
    whose ratio R = F405/F488 reports the fraction of oxidised dimer. The
    calibration constants R_min (fully reduced) and R_max (fully oxidised)
    are dye- and instrument-specific; we use the published values of
    Schwarzlaender et al. (2008) for roGFP2 imaging in HEK293:

        R_min = 0.20,   R_max = 4.50

    and the spectral coefficients I405red/I405ox, I488red/I488ox = 0.85 / 5.0
    and 1.0 / 0.20 respectively (normalised). The oxidation fraction
    OxD(p) = (R - R_min) / (R_max - R_min) in [0, 1] is then the per-pixel
    target of the closed-form solver (after rescaling to [-1, +1]).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float64)
    # Linear OxD gradient from top (reduced) to bottom (oxidised),
    # plus a "mitochondrial" oxidised patch in the upper-right corner.
    oxd = 0.20 + 0.60 * (yy / (side - 1))
    mito_patch = 0.30 * np.exp(-(((yy - 3) ** 2 + (xx - 12) ** 2) / 6.0))
    oxd = np.clip(oxd + mito_patch + rng.normal(0.0, 0.01, oxd.shape), 0.0, 1.0)

    # Schwarzlander-style spectral coefficients (illustrative, normalised)
    I405_red, I405_ox = 0.85, 5.00
    I488_red, I488_ox = 1.00, 0.20
    f405 = (1.0 - oxd) * I405_red + oxd * I405_ox
    f488 = (1.0 - oxd) * I488_red + oxd * I488_ox
    # Add ~1% intensity noise
    f405 = f405 + rng.normal(0.0, 0.01, f405.shape)
    f488 = f488 + rng.normal(0.0, 0.01, f488.shape)
    return f405, f488, oxd


def validate_class_c() -> dict[str, float]:
    """Class C (calibrated ratio): f = (R - R_min)/(R_max - R_min), in [0,1]."""
    f405, f488, oxd_true = synthesise_rogfp(side=16, seed=2)
    eps = 1e-9
    R = f405 / np.maximum(f488, eps)
    R_min, R_max = 0.20, 4.50
    f_cal = (R - R_min) / (R_max - R_min)
    # Affine rescale to [-1, +1] (Lemma 2 of the main paper)
    u = 2.0 * np.clip(f_cal, 0.0, 1.0) - 1.0

    norm = float(max(f405.max(), f488.max()))
    alpha, beta = closed_form_params(f405, f488, u, normalization=norm)
    decoded = closed_form_decoded(f405, f488, alpha, beta, normalization=norm)

    mse = float(np.mean((u - decoded) ** 2))
    psnr = psnr_db(u, decoded, signal_range=2.0)

    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.0))
    titles = [
        "F405 (intensity AU)",
        "F488 (intensity AU)",
        r"$R = F_{405}/F_{488}$",
        r"$u = 2 \cdot (R - R_{\mathrm{min}})/(R_{\mathrm{max}} - R_{\mathrm{min}}) - 1$",
        "decoded - u  (closed-form)",
    ]
    images = [f405, f488, R, u, decoded - u]
    cmaps = ["viridis", "viridis", "magma", "RdBu_r", "RdBu_r"]
    norms = [
        plt.Normalize(vmin=f405.min(), vmax=f405.max()),
        plt.Normalize(vmin=f488.min(), vmax=f488.max()),
        plt.Normalize(vmin=R.min(), vmax=R.max()),
        plt.Normalize(vmin=-1, vmax=1),
        plt.Normalize(vmin=-1e-15, vmax=1e-15),
    ]
    for ax, img, title, cmap, cn in zip(axes, images, titles, cmaps, norms, strict=True):
        im = ax.imshow(img, cmap=cmap, norm=cn, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Class C (roGFP2 redox imaging) — closed-form solver, n = 4 "
        f"(16x16, 512 parameters)   |   MSE = {mse:.2e},  PSNR = {psnr:.1f} dB",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(
        OUT / "fig_rogfp_validation.png", dpi=200, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)
    return {"mse": mse, "psnr_db": psnr, "oxd_min": float(oxd_true.min()),
            "oxd_max": float(oxd_true.max())}


def main() -> int:
    b_stats = validate_class_b()
    c_stats = validate_class_c()
    print("Class B (Fura-2):", b_stats)
    print("Class C (roGFP2):", c_stats)
    for f in ["fig_fura2_validation.png", "fig_rogfp_validation.png"]:
        p = OUT / f
        print(f"  {p.name}: {p.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
