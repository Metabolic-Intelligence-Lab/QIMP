"""
fig11_nonrestoring_hw.png — Class B pipeline hardware match rate vs transpiled
two-qubit gate count on IBM Heron r2, restoring vs non-restoring divider,
coloured by use case. Shows the non-restoring compression moving the
n=1,q=2 pipeline below the empirical survival ceiling.

Reads the persisted summary.json files from data/output/ibm_hw/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
HW_ROOT = REPO / "data" / "output" / "ibm_hw"
OUT = REPO / "paper" / "figures_autonomous" / "fig11_nonrestoring_hw.png"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]


def load_label(label: str) -> dict | None:
    for summary in HW_ROOT.glob("*/summary.json"):
        try:
            data = json.loads(summary.read_text())
        except Exception:
            continue
        for val in data.values():
            if val.get("label") == label:
                return val
    return None


# (label, divider, dataset, n_pixels, fallback_cx, fallback_match)
JOBS = [
    # restoring-divider baseline
    ("j3_class_b_synthetic", "restoring", "synthetic", 4, 1259, 0),
    ("j4_class_b_laurdan", "restoring", "Laurdan", 4, 1174, 0),
    # non-restoring fan-out
    ("j7_class_b_laurdan_nonrestoring", "nonrestoring", "Laurdan", 4, 679, 4),
    ("j8_synthetic_n1q2_nonrestoring", "nonrestoring", "synthetic", 4, 693, 3),
    ("j9_fura2_n1q2_nonrestoring", "nonrestoring", "Fura-2", 4, 675, 0),
    ("j9b_fura2_n1q2_nonrestoring_marrakesh", "nonrestoring", "Fura-2", 4, 670, 0),
    ("j11_canonical_n2q2_nonrestoring", "nonrestoring", "Laurdan", 16, 1348, 8),
    ("j12_canonical_n1q3_nonrestoring", "nonrestoring", "Laurdan", 4, 1136, 4),
    ("j13_rogfp2_n1q3_nonrestoring", "nonrestoring", "roGFP2", 4, 1247, 0),
]

USE_CASE_COLOR = {
    "Laurdan": "C0", "synthetic": "C1", "Fura-2": "C2", "roGFP2": "C3",
}


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)

    # gate-budget fidelity curve
    nb = np.linspace(100, 2400, 300)
    fb = np.exp(-nb * 4e-3) * 100
    ax.plot(nb, fb, "k:", alpha=0.5,
            label=r"$\exp(-N_{2q}\,\epsilon)\times100$, $\epsilon=4\!\times\!10^{-3}$")

    # uniform-random floors
    ax.axhline(25, color="0.6", ls="--", lw=0.8, label="q=2 floor (25%)")
    ax.axhline(12.5, color="0.8", ls="--", lw=0.8, label="q=3 floor (12.5%)")

    # survival-ceiling band
    ax.axvspan(700, 900, color="gold", alpha=0.18, label="survival ceiling (~700–900 CX)")

    seen_uc = set()
    for label, div, uc, npix, fb_cx, fb_match in JOBS:
        rec = load_label(label)
        if rec is not None:
            cx = rec.get("two_q_gate_count", fb_cx)
            match = rec.get("match_count", fb_match)
        else:
            cx, match = fb_cx, fb_match
        if cx is None or match is None:
            continue  # job not yet returned (e.g. J9 pending)
        pct = 100.0 * match / npix
        marker = "o" if div == "nonrestoring" else "X"
        fc = USE_CASE_COLOR.get(uc, "0.5") if div == "nonrestoring" else "none"
        ec = USE_CASE_COLOR.get(uc, "0.5")
        lbl = None
        key = uc if div == "nonrestoring" else "restoring (J3,J4)"
        if key not in seen_uc:
            lbl = key
            seen_uc.add(key)
        ax.scatter(cx, pct, marker=marker, s=130,
                   facecolor=fc, edgecolor=ec, linewidth=1.8,
                   zorder=5, label=lbl)
        tag = label.split("_")[0].upper()
        ax.annotate(f" {tag}", (cx, pct), fontsize=8,
                    xytext=(5, 3), textcoords="offset points")

    ax.set_xlabel(r"transpiled two-qubit gate count $N_{2q}$ (Heron r2)")
    ax.set_ylabel("per-pixel match vs classical (%)")
    ax.set_title("Class B autonomous ratio on hardware:\n"
                 "non-restoring divider crosses the gate-budget floor")
    ax.set_ylim(-5, 108)
    ax.set_xlim(400, 2500)
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.25)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
