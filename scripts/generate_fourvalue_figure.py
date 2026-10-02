"""Fig. 8 (v2): four-value target, per-pixel quotient histograms on two devices,
raw and after flat-field correction, mean over the campaign-3 runs.
Usage: .venv/bin/python scripts/generate_fourvalue_figure.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from decode_bias_corrected import collect, run_matrices
from analyse_hw_signal import load_images

images = load_images("fourvalue", 1, 2)
arms = [("ibm_marrakesh, 16 384 shots, 5 runs", ("c3_c2_",)), ("ibm_kingston, 4096 shots, 3 runs", ("c3_c3_",))]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharey="row")
cols = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
for j, (title, pref) in enumerate(arms):
    mats = run_matrices(collect(pref), images, "nonrestoring", 1, 2)
    P = np.mean([M for _, M, _ in mats], axis=0)          # 4 px x 4 bins
    t = mats[0][2]
    Q = P - P.mean(axis=0, keepdims=True)
    for i, (M, ylabel) in enumerate([(P, "share of shots"), (Q, "share minus pixel mean")]):
        ax = axes[i, j]
        x = np.arange(4)
        for px in range(4):
            ax.bar(x + (px - 1.5) * 0.2, M[px], width=0.2, color=cols[px], label=f"pixel {px}, true R = {t[px]}")
            ax.plot([t[px] + (px - 1.5) * 0.2], [M[px, t[px]] + (0.02 if i == 0 else 0.012)], marker="v", color="k", ms=4)
        if i == 0:
            ax.axhline(0.25, ls=":", c="k", lw=0.8); ax.set_title(title, fontsize=9)
        else:
            ax.axhline(0, c="k", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels([f"bin {v}" for v in x], fontsize=8)
        if j == 0: ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
axes[0, 0].legend(fontsize=6.5, loc="upper right", ncol=2)
fig.suptitle("Four-value target R = [[0,1],[2,3]], ~650 CX, TREX only.  Top: raw read-out.  Bottom: flat-field corrected.  ▼ = true bin", fontsize=8.5)
fig.tight_layout()
out = REPO / "paper/figures_autonomous/fig_fourvalue_hw.png"
fig.savefig(out, dpi=200); print("wrote", out)
