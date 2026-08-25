"""Recipe factory — one entry per encoder for the hardware sweep.

A `CircuitRecipe` bundles the quantum circuit, a counts→image decoder,
and the matching classical reference, so the sweep loop is one-shot
per (encoder, n) regardless of encoder family.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image
from qiskit import QuantumCircuit

__all__ = ["CircuitRecipe", "build_recipes"]


@dataclass
class CircuitRecipe:
    """One row of the sweep matrix: circuit + decoder + classical reference."""

    label: str
    encoder: str  # "frqi" | "frqi_multi" | "neqr" | "qpie" | "mcrqi" | "ncqi" | "gp"
    n: int
    q: int
    m: int
    qc: QuantumCircuit
    decoder: Callable[[dict[str, int]], np.ndarray]
    reference: np.ndarray


def _downsample_to_n(img: np.ndarray, *, n: int) -> np.ndarray:
    """Lanczos-resample to ``(2^n, 2^n)``. Preserves dtype for uint8.

    Accepts 2D grayscale or 3D RGB(A) — the alpha channel is dropped.
    """
    side = 1 << n
    if img.ndim == 2:
        pil = Image.fromarray(img).resize((side, side), Image.Resampling.LANCZOS)
        return np.asarray(pil, dtype=img.dtype)
    if img.ndim == 3 and img.shape[2] in (3, 4):
        pil = Image.fromarray(img[..., :3], mode="RGB").resize(
            (side, side), Image.Resampling.LANCZOS
        )
        return np.asarray(pil, dtype=img.dtype)
    raise ValueError(f"unsupported image shape {img.shape}")


def build_recipes(
    image: np.ndarray, *, n: int, q: int = 2, alpha: float = 0.5
) -> list[CircuitRecipe]:
    """Build one CircuitRecipe per encoder.

    Parameters
    ----------
    image
        Source frame — 2D grayscale or 3D RGB. Downsampled to 2^n × 2^n.
    n
        Spatial qubits per axis.
    q
        Intensity qubits for NEQR / NCQI (default 2 = NISQ-friendly).
    alpha
        G-factor for the GP recipe.
    """
    from qimp.encoding.frqi import frqi_circuit, frqi_decode

    if image.ndim == 2:
        gray = _downsample_to_n(image, n=n)
        rgb = np.stack([gray, gray, gray], axis=-1)
    else:
        rgb = _downsample_to_n(image, n=n)
        gray = rgb[..., 1]  # green channel as grayscale source

    recipes: list[CircuitRecipe] = []

    # --- FRQI single-image ---
    norm = float(gray.max()) or 1.0
    qc = frqi_circuit(gray, normalization=norm)

    def _decode_frqi(counts: dict[str, int], _n: int = n, _norm: float = norm) -> np.ndarray:
        out = frqi_decode(counts, n=_n, m=0, normalization=_norm)
        assert isinstance(out, np.ndarray)  # m=0 contract
        return out

    recipes.append(
        CircuitRecipe(
            label=f"frqi_n{n}",
            encoder="frqi",
            n=n,
            q=0,
            m=0,
            qc=qc,
            decoder=_decode_frqi,
            reference=gray.astype(np.float64),
        )
    )

    # --- FRQI multi-image (m=1, two channels = R, G) ---
    red = rgb[..., 0]
    green = rgb[..., 1]
    stack = np.stack([red, green], axis=0).astype(np.float64)
    norm_multi = float(stack.max()) or 1.0
    qc_multi = frqi_circuit(stack, normalization=norm_multi)

    def _decode_multi(counts: dict[str, int], _n: int = n, _norm: float = norm_multi) -> np.ndarray:
        imgs = frqi_decode(counts, n=_n, m=1, normalization=_norm)
        assert isinstance(imgs, list)  # m>0 contract
        return imgs[1]

    recipes.append(
        CircuitRecipe(
            label=f"frqi_multi_n{n}",
            encoder="frqi_multi",
            n=n,
            q=0,
            m=1,
            qc=qc_multi,
            decoder=_decode_multi,
            reference=green.astype(np.float64),
        )
    )

    # --- NEQR ---
    from qimp.encoding.neqr import neqr_circuit, neqr_decode

    max_q_val = (1 << q) - 1
    gray_max = max(int(gray.max()), 1)
    neqr_img = (gray.astype(np.float64) / gray_max * max_q_val).round().astype(np.uint8)
    qc_neqr = neqr_circuit(neqr_img, q=q)

    def _decode_neqr(counts: dict[str, int], _n: int = n, _q: int = q) -> np.ndarray:
        return neqr_decode(counts, n=_n, q=_q)

    recipes.append(
        CircuitRecipe(
            label=f"neqr_n{n}",
            encoder="neqr",
            n=n,
            q=q,
            m=0,
            qc=qc_neqr,
            decoder=_decode_neqr,
            reference=neqr_img.astype(np.float64),
        )
    )

    # --- QPIE ---
    from qimp.encoding.qpie import normalize_amplitudes, qpie_circuit, qpie_decode

    _, _, rms = normalize_amplitudes(gray.astype(np.float64))
    qc_qpie = qpie_circuit(gray.astype(np.float64))

    def _decode_qpie(counts: dict[str, int], _n: int = n, _rms: float = float(rms)) -> np.ndarray:
        return qpie_decode(counts, n=_n, rms=_rms)

    recipes.append(
        CircuitRecipe(
            label=f"qpie_n{n}",
            encoder="qpie",
            n=n,
            q=0,
            m=0,
            qc=qc_qpie,
            decoder=_decode_qpie,
            reference=gray.astype(np.float64),
        )
    )

    # --- MCRQI ---
    from qimp.encoding.mcrqi import mcrqi_circuit, mcrqi_decode

    norm_mc = float(rgb.max()) or 1.0
    qc_mc = mcrqi_circuit(rgb, normalization=norm_mc)

    def _decode_mcrqi(counts: dict[str, int], _n: int = n, _norm: float = norm_mc) -> np.ndarray:
        return mcrqi_decode(counts, n=_n, normalization=_norm)

    recipes.append(
        CircuitRecipe(
            label=f"mcrqi_n{n}",
            encoder="mcrqi",
            n=n,
            q=0,
            m=0,
            qc=qc_mc,
            decoder=_decode_mcrqi,
            reference=rgb.astype(np.float64),
        )
    )

    # --- NCQI ---
    from qimp.encoding.ncqi import ncqi_circuit, ncqi_decode

    rgb_max = max(int(rgb.max()), 1)
    ncqi_img = (rgb.astype(np.float64) / rgb_max * max_q_val).round().astype(np.uint8)
    qc_ncqi = ncqi_circuit(ncqi_img, q=q)

    def _decode_ncqi(counts: dict[str, int], _n: int = n, _q: int = q) -> np.ndarray:
        return ncqi_decode(counts, n=_n, q=_q)

    recipes.append(
        CircuitRecipe(
            label=f"ncqi_n{n}",
            encoder="ncqi",
            n=n,
            q=q,
            m=0,
            qc=qc_ncqi,
            decoder=_decode_ncqi,
            reference=ncqi_img.astype(np.float64),
        )
    )

    # --- GP corrected (paper centre-piece) ---
    from qimp.processing.gp_ratio import (
        analytical_gp_params,
        apply_gp_function,
        classical_gp_image,
        decode_gp_counts,
    )

    g_chan = rgb[..., 1].astype(np.float64)
    r_chan = rgb[..., 0].astype(np.float64)
    # Stack order [red, green] matches `qimp.processing.gp_ratio.evaluate_gp`
    # so the m=1 selection qubit maps R -> |0>, G -> |1>; decode_gp_counts
    # marginalises over selection without needing a channel argument.
    rg_stack = np.stack([r_chan, g_chan], axis=0)
    norm_gp = float(rg_stack.max()) or 1.0

    qc_gp = frqi_circuit(rg_stack, normalization=norm_gp)
    gp_params = analytical_gp_params(g_chan, r_chan, alpha=alpha, normalization=norm_gp)
    apply_gp_function(qc_gp, n=n, m=1, params=list(gp_params))

    def _decode_gp(counts: dict[str, int], _n: int = n) -> np.ndarray:
        return decode_gp_counts(counts, n=_n, m=1)

    recipes.append(
        CircuitRecipe(
            label=f"gp_n{n}",
            encoder="gp",
            n=n,
            q=0,
            m=1,
            qc=qc_gp,
            decoder=_decode_gp,
            reference=classical_gp_image(g_chan, r_chan, alpha=alpha),
        )
    )

    return recipes
