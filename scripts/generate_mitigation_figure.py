"""Figure: dynamical decoupling suppresses the quotient signal, and the
per-pixel match rate cannot see it.

Left  -- the decoded quotient histogram of one pixel under the paper's default
         mitigation (TREX + XY4 DD) and under TREX alone. Same circuit, same
         patch, same device; the default flattens the distribution onto the
         uniform null while TREX alone leaves it clearly peaked on the true
         value.
Right -- signal margin against match rate for the four mitigation settings.
         The match rate spans 3.0-4.0 out of 4 while the margin behind it
         spans 3.8-12.8 sigma: the two are decoupled, which is why a match
         count on its own cannot support a recovery claim.

Reads paper/data_autonomous/hw_signal_audit.json (written by
scripts/analyse_hw_signal.py).

Usage:
    python scripts/generate_mitigation_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "paper" / "data_autonomous" / "hw_signal_audit.json"
OUT = REPO / "paper" / "figures_autonomous" / "fig_mitigation_signal.png"

# Fixed hue order, validated for CVD separation (dE 24.6 protan) and paired
# with hatch so the panels survive greyscale printing.
COL_ON, COL_OFF = "#1f77b4", "#ff7f0e"

CONFIGS = [
    ("TREX + XY4 DD", "j7rep_canonical_nonrestoring", True),
    ("XY4 DD only", "j7x_dd_only", True),
    ("TREX only", "j7x_trex_only", False),
    ("no mitigation", "j7x_marrakesh_nomit", False),
]


def runs(audit: dict, sub: str) -> list[dict]:
    return [v for k, v in audit.items() if sub in k]


def main() -> int:
    audit = json.loads(AUDIT.read_text())
    fig, (axL, axR) = plt.subplots(figsize=(9.6, 4.0), ncols=2, constrained_layout=True)

    # ---- left: one pixel's quotient histogram, DD on vs DD off -------------
    pixel = 2  # (1,0): a true R = 1 pixel, away from the divide-by-zero corner
    hist_on = np.mean(
        [r["pixels"][pixel]["histogram"] for r in runs(audit, CONFIGS[0][1])], axis=0
    )
    hist_off = np.mean(
        [r["pixels"][pixel]["histogram"] for r in runs(audit, CONFIGS[2][1])], axis=0
    )
    x = np.arange(len(hist_on))
    w = 0.38
    axL.bar(
        x - w / 2, hist_on * 100, w, label="TREX + XY4 DD (paper default)",
        color=COL_ON, edgecolor="white", linewidth=0.8,
    )
    axL.bar(
        x + w / 2, hist_off * 100, w, label="TREX only", color=COL_OFF,
        edgecolor="white", linewidth=0.8, hatch="///",
    )
    axL.axhline(25, color="0.35", ls="--", lw=1.0)
    axL.annotate(
        "uniform null (25 %)", (3.46, 26.2), fontsize=7.5, color="0.35", ha="right"
    )
    axL.annotate(
        "true value", (1 + w / 2, hist_off[1] * 100 + 2.0), fontsize=8,
        ha="center", va="bottom", color="0.15", fontweight="bold",
    )
    axL.set_xticks(x)
    axL.set_xticklabels([str(i) for i in x])
    axL.set_xlabel("decoded quotient value")
    axL.set_ylabel("share of shots at pixel (1,0)  [%]")
    axL.set_ylim(0, 55)
    axL.set_title("Dynamical decoupling flattens the read-out", fontsize=10)
    axL.legend(fontsize=7.5, loc="upper right")
    axL.grid(True, axis="y", alpha=0.25)

    # ---- right: margin vs match rate, four mitigation settings -------------
    # Labels are placed per point rather than by a rule: the DD-only spread is
    # wide enough (sd 1.73) that a uniform offset collides with its whiskers.
    OFFSETS = {
        "TREX + XY4 DD": (0, -20, "center"),
        "XY4 DD only": (0, 14, "center"),
        "TREX only": (-12, 4, "right"),
        "no mitigation": (0, -20, "center"),
    }
    for label, sub, dd in CONFIGS:
        rs = runs(audit, sub)
        sig = np.mean([r["mean_sigma_over_null"] for r in rs])
        match = np.mean([r["match"] for r in rs])
        err = np.std([r["match"] for r in rs], ddof=1)
        colour = COL_ON if dd else COL_OFF
        axR.errorbar(
            match, sig, xerr=err, fmt="o", ms=9, capsize=3, lw=1.0,
            ecolor=colour, alpha=0.85, color=colour,
            markerfacecolor="white" if dd else colour,
            markeredgecolor=colour, markeredgewidth=1.8, zorder=3,
        )
        dx, dy, ha = OFFSETS[label]
        axR.annotate(
            label, (match, sig), textcoords="offset points",
            xytext=(dx, dy), fontsize=8, ha=ha, color="0.15", zorder=4,
        )
    axR.axhline(3, color="0.35", ls="--", lw=1.0)
    axR.annotate("3σ", (4.55, 3.4), fontsize=7.5, color="0.35", ha="right")
    axR.annotate(
        "DD on: open   DD off: filled", (1.05, 14.6), fontsize=7.5, color="0.35"
    )
    axR.set_xlabel("per-pixel match rate  [/ 4]   (mean ± sd over 5 runs)")
    axR.set_ylabel("signal margin above the uniform null  [σ]")
    # Wide enough to show the DD-only whisker in full: its sd of 1.73 is part
    # of the point, that setting swings between 0/4 and 4/4 run to run.
    axR.set_xlim(0.9, 4.65)
    axR.set_ylim(0, 15.8)
    axR.set_title("The match rate does not track the signal", fontsize=10)
    axR.grid(True, alpha=0.25)

    fig.suptitle(
        "Class-B non-restoring pipeline, canonical Laurdan patch, "
        "ibm_marrakesh, ~650 CX, 5 repeats per setting",
        fontsize=9,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
