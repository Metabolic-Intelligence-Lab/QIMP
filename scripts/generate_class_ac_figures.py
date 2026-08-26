"""Figures for the end-to-end Class-A and Class-C runs of §5.3.

Both figures were originally produced by hand and had no generator in the
repository, which left two of the manuscript's main-text figures the only ones
that could not be rebuilt from committed data. This script reproduces them from
the npz written by `run_autonomous_class_a_gp_mps.py` and
`run_autonomous_class_c_rogfp_mps.py`.

Four panels each, left to right: the two quantised input channels on a shared
intensity scale, the classical reference observable, and the observable decoded
from the quantum circuit. The two right-hand panels share their colour scale so
that "identical at every pixel" is something the reader can see rather than take
on trust.

Colour follows the data's job: viridis for the intensity panels (sequential,
one hue light to dark, perceptually uniform); a diverging map with a neutral
midpoint for Laurdan GP, which spans [-1, +1] about a meaningful zero; and a
sequential map for the roGFP redox index, which spans [0, 1] with no special
midpoint. All three are colourblind-safe and survive greyscale conversion.

Usage:
    python scripts/generate_class_ac_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "paper" / "data_autonomous"
FIGS = REPO / "paper" / "figures_autonomous"

DPI = 300


def cell_labels(ax, values, fmt, threshold_map, cmap, vmin, vmax):
    """Write each cell's value on it, in whichever ink stays legible."""
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    for (r, c), v in np.ndenumerate(values):
        lum = sum(cmap(norm(threshold_map[r, c]))[:3]) / 3
        ax.text(
            c, r, fmt.format(v), ha="center", va="center", fontsize=11,
            color="white" if lum < 0.5 else "black",
        )


def panel(ax, values, title, cmap, vmin, vmax, fmt):
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    cell_labels(ax, values, fmt, values, plt.get_cmap(cmap), vmin, vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def class_a() -> None:
    d = np.load(DATA / "class_a_gp_n1_q4.npz")
    q = int(d["q"])
    hi = (1 << q) - 1
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), constrained_layout=True)
    panel(axes[0], d["I_a"], f"$I_a$ (red, q={q})", "viridis", 0, hi, "{:.0f}")
    panel(axes[1], d["I_b"], f"$I_b$ (green, q={q})", "viridis", 0, hi, "{:.0f}")
    panel(axes[2], d["gp_classical"], "classical GP", "RdBu_r", -1, 1, "{:+.3f}")
    panel(
        axes[3], d["gp_quantum"], f"quantum GP (MPS, {int(d['qubits'])}q)",
        "RdBu_r", -1, 1, "{:+.3f}",
    )
    match = int(np.isclose(d["gp_quantum"], d["gp_classical"]).sum())
    fig.suptitle(
        "Autonomous Class-A Laurdan GP, end-to-end on AerSimulator(mps): "
        f"bit-exact {match}/{d['gp_classical'].size}",
        fontsize=12,
    )
    out = FIGS / "fig_class_a_gp_e2e.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}  ({match}/{d['gp_classical'].size} bit-exact)")


def class_c() -> None:
    d = np.load(DATA / "class_c_rogfp_n1_q4.npz")
    q = int(d["q"])
    hi = (1 << q) - 1
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), constrained_layout=True)
    panel(axes[0], d["I_a"], f"$I_a$ = F405 (q={q})", "viridis", 0, hi, "{:.0f}")
    panel(axes[1], d["I_b"], f"$I_b$ = F488 (q={q})", "viridis", 0, hi, "{:.0f}")
    panel(axes[2], d["rc_classical"], "classical $R_C$ (redox)", "magma", 0, 1, "{:.3f}")
    panel(
        axes[3], d["rc_quantum"], f"quantum $R_C$ (MPS, {int(d['qubits'])}q)",
        "magma", 0, 1, "{:.3f}",
    )
    match = int(np.isclose(d["rc_quantum"], d["rc_classical"]).sum())
    fig.suptitle(
        "Autonomous Class-C roGFP calibrated redox $R_C$, end-to-end on "
        f"AerSimulator(mps): bit-exact {match}/{d['rc_classical'].size}",
        fontsize=12,
    )
    out = FIGS / "fig_class_c_rogfp_e2e.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}  ({match}/{d['rc_classical'].size} bit-exact)")


if __name__ == "__main__":
    class_a()
    class_c()
