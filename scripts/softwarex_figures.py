"""Figures and numbers of the SoftwareX software paper.

Every number quoted in ``paper/softwarex/`` that is not a static property of
the code (qubit counts, round-trip errors, map agreement) is produced here,
from the Laurdan frame shipped in ``data/immagini/trainQML``, on the exact
statevector simulator. No IBM account or GPU is needed.

    python scripts/softwarex_figures.py  # writes paper/softwarex/figures/*.png
                                         # and paper/softwarex/figures/numbers.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from qiskit import transpile

from qimp.encoding.frqi import FrqiEncoder
from qimp.encoding.mcrqi import McrqiEncoder
from qimp.encoding.ncqi import NcqiEncoder
from qimp.encoding.neqr import NeqrEncoder
from qimp.encoding.qpie import QpieEncoder
from qimp.io.image import load_tiff_16bit
from qimp.metrics import psnr
from qimp.processing.filters import qhed_filter, qhed_full_edges
from qimp.processing.geometric import axis_flip
from qimp.testing import exact_counts, ideal_simulation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "softwarex" / "figures"
FRAME = ROOT / "data" / "immagini" / "trainQML" / "membraneStack_Sample011_L_UV_DC_001rbc3DM2.tif"


def _quantise(x: np.ndarray, q: int) -> np.ndarray:
    """Linear map of a non-negative array onto the integers 0 .. 2^q - 1."""
    x = x.astype(float)
    top = float(x.max()) or 1.0
    return np.clip(np.round(x / top * ((1 << q) - 1)), 0, (1 << q) - 1).astype(int)


def _patch(n: int, row0: int = 10, col0: int = 18) -> tuple[np.ndarray, np.ndarray]:
    """Two co-registered Laurdan channels, a 2^n x 2^n patch of the frame."""
    rgb = load_tiff_16bit(FRAME)
    side = 1 << n
    sl = (slice(row0, row0 + side), slice(col0, col0 + side))
    return rgb[sl][..., 0], rgb[sl][..., 1]


def fig_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    layers = [
        (
            5.0,
            "Applications",
            ["Streamlit explorer (qimp ui)", "hardware drivers & analysis scripts"],
            "#e8eef7",
        ),
        (3.7, "Library core", ["encoding", "processing", "qml", "io"], "#dfeee3"),
        (
            2.4,
            "Runtime & utilities",
            ["runtime (Aer, IBM)", "testing", "metrics", "qft"],
            "#f6efdc",
        ),
        (
            1.1,
            "Dependencies",
            ["Qiskit >= 1.0", "qiskit-aer", "qiskit-ibm-runtime (opt.)", "NumPy / SciPy / Pillow"],
            "#eeeeee",
        ),
    ]
    for y, title, boxes, colour in layers:
        ax.add_patch(
            FancyBboxPatch(
                (0.2, y - 0.5), 9.6, 1.0, boxstyle="round,pad=0.02", fc=colour, ec="#555555"
            )
        )
        ax.text(0.35, y + 0.32, title, fontsize=10, weight="bold", va="center")
        w = 9.0 / len(boxes)
        for i, b in enumerate(boxes):
            x = 0.5 + i * w
            ax.add_patch(
                FancyBboxPatch(
                    (x, y - 0.38),
                    w - 0.2,
                    0.55,
                    boxstyle="round,pad=0.02",
                    fc="white",
                    ec="#777777",
                )
            )
            ax.text(
                x + (w - 0.2) / 2,
                y - 0.1,
                b,
                fontsize=8.5,
                ha="center",
                va="center",
                family="monospace",
            )
    for y in (4.45, 3.15, 1.85):
        ax.annotate(
            "",
            xy=(9.6, y - 0.25),
            xytext=(9.6, y + 0.15),
            arrowprops=dict(arrowstyle="->", color="#555555"),
        )
    ax.text(
        5, 5.85, "qimp-mi: each layer depends only on the layers below it", ha="center", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def roundtrip_numbers(n: int = 3, q: int = 8) -> dict[str, float]:
    """FRQI (shot-free, exact) and NEQR round trip on one 8x8 channel."""
    a, _ = _patch(n)
    img8 = _quantise(a, q).astype(np.uint8)
    out: dict[str, float] = {}
    neqr = NeqrEncoder()
    qc = neqr.encode(img8)
    counts = exact_counts(qc)
    rec = neqr.decode(counts)
    out["neqr_qubits"] = qc.num_qubits
    out["neqr_exact"] = float(np.array_equal(rec, img8))
    frqi = FrqiEncoder()
    qc = frqi.encode(img8)
    counts = exact_counts(qc)
    rec = frqi.decode(counts)
    out["frqi_qubits"] = qc.num_qubits
    out["frqi_psnr_db"] = float(psnr(img8, rec, max_intensity=255))
    return out


def fig_examples(path: Path, n: int = 3, q: int = 8, shots: int = 8192) -> dict[str, float]:
    """Encode / process / decode one 8x8 channel of the Laurdan frame."""
    a, _ = _patch(n)
    img = _quantise(a, q).astype(np.uint8)
    out: dict[str, float] = {"side": 1 << n, "q": q, "shots": shots}

    frqi = FrqiEncoder()
    qc = frqi.encode(img)
    rec_frqi = frqi.decode(ideal_simulation(qc, shots=shots))
    out["frqi_qubits"] = qc.num_qubits
    out["frqi_psnr_shots_db"] = float(psnr(img, rec_frqi, max_intensity=255))

    neqr = NeqrEncoder()
    qc = neqr.encode(img)
    rec_neqr = neqr.decode(exact_counts(qc))
    out["neqr_qubits"] = qc.num_qubits
    out["neqr_exact"] = float(np.array_equal(rec_neqr, img))

    qc = neqr.encode(img)
    axis_flip(qc, n, axis="x", pos_offset=q)
    flipped = neqr.decode(exact_counts(qc))
    out["neqr_flip_exact"] = float(
        np.array_equal(flipped, img[:, ::-1]) or np.array_equal(flipped, img[::-1, :])
    )

    qc_h, _, _ = qhed_filter(img.astype(float))
    qc_v, _, _ = qhed_filter(img.T.astype(float))
    edges = qhed_full_edges(img.astype(float), exact_counts(qc_h), exact_counts(qc_v))
    out["qhed_qubits"] = qc_h.num_qubits

    fig, axs = plt.subplots(1, 5, figsize=(13, 3))
    panels = [
        (img, "input (8-bit, 8x8)"),
        (
            rec_frqi,
            f"FRQI, {out['frqi_qubits']} qubits\n{shots} shots, PSNR {out['frqi_psnr_shots_db']:.1f} dB",
        ),
        (rec_neqr, f"NEQR, {out['neqr_qubits']} qubits\nexact round trip"),
        (flipped, "NEQR + axis_flip"),
        (edges[:, :-1], f"QHED edges, {out['qhed_qubits']} qubits\n(wrap-around column dropped)"),
    ]
    for ax, (im_, title) in zip(axs, panels, strict=True):
        im = ax.imshow(im_, cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return out


def resource_table(sizes: tuple[int, ...] = (2, 3)) -> list[dict[str, object]]:
    """Qubits and transpiled depth / CX count per encoder (8-bit, random image)."""
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    for n in sizes:
        side = 1 << n
        gray = rng.integers(0, 256, (side, side)).astype(np.uint8)
        rgb = rng.integers(0, 256, (side, side, 3)).astype(np.uint8)
        for name, qc in (
            ("FRQI", FrqiEncoder().encode(gray)),
            ("NEQR", NeqrEncoder().encode(gray)),
            ("QPIE", QpieEncoder().encode(gray)),
            ("MCRQI", McrqiEncoder().encode(rgb)),
            ("NCQI", NcqiEncoder().encode(rgb)),
        ):
            t = transpile(
                qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1, seed_transpiler=0
            )
            rows.append(
                {
                    "n": n,
                    "encoder": name,
                    "qubits": qc.num_qubits,
                    "depth": t.depth(),
                    "cx": t.count_ops().get("cx", 0),
                }
            )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_architecture(OUT / "fig1_architecture.png")
    numbers = {
        "roundtrip": roundtrip_numbers(),
        "examples": fig_examples(OUT / "fig2_examples.png"),
        "resources": resource_table(),
    }
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=2))
    print(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()
