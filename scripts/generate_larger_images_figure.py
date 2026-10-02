"""Fig. 9 (v2): larger images with the uniformly controlled load on ibm_kingston.
Classical quotient image vs flat-field-decoded hardware quotient image, for the
4x4 tiled four-value target, the 4x4 random image, the 4x4 Laurdan patch and the
8x8 random image (campaign 7). Usage: .venv/bin/python scripts/generate_larger_images_figure.py
"""
from __future__ import annotations
import sys, glob, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from decode_bias_corrected import collect, run_matrices
from analyse_hw_signal import load_images, analyse

panels = [("4 × 4 four-value (tiled)", ("c7_a_",), "fourvalue", 2),
          ("4 × 4 random image", ("c7_b_",), "random4", 2),
          ("4 × 4 Laurdan patch", ("c7_c_", "c10_c_"), "canonical_shared", 2),
          ("8 × 8 random image", ("c7_d_", "c8_a_"), "random4", 3),
          ("8 × 8 Laurdan patch", ("c10_d_",), "canonical_shared", 3)]
fig, axes = plt.subplots(2, 5, figsize=(12.0, 5.0))
cmap = ListedColormap(["#f0f0f0", "#9ecae1", "#3182bd", "#08519c"])
for j, (title, pref, ds, n) in enumerate(panels):
    images = load_images(ds, n, 2); I_a, I_b = images
    runs = collect(pref); mats = run_matrices(runs, images, "nonrestoring", n, 2)
    # best run by flat-field match, to show what a single job returns
    best = None
    for label, P, t in mats:
        dec = (P - P.mean(axis=0, keepdims=True)).argmax(axis=1)
        m = int(((dec == t) & (t >= 0)).sum())
        if best is None or m > best[0]: best = (m, label, dec, t)
    m, label, dec, t = best
    side = 2 ** n
    R = np.where(I_b > 0, I_a // np.maximum(I_b, 1), -1)
    Q = np.full((side, side), -1)
    r = analyse(runs[[l for l, _, _ in mats].index(label)][1], images, "nonrestoring", n, 2)
    for k, px in enumerate(r["pixels"]):
        rr, cc = px["pixel"]; Q[rr, cc] = dec[k]
    valid = int((R >= 0).sum())
    for i, (M, sub) in enumerate([(R, "classical quotient"), (Q, f"hardware, flat-field decode: {m}/{valid}")]):
        ax = axes[i, j]
        Mm = np.ma.masked_less(M, 0)
        ax.imshow(Mm, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        for rr in range(side):
            for cc in range(side):
                if M[rr, cc] >= 0 and side <= 4:
                    ok = (i == 0) or (Q[rr, cc] == R[rr, cc])
                    ax.text(cc, rr, str(M[rr, cc]), ha="center", va="center", fontsize=9,
                            color="white" if M[rr, cc] >= 2 else "black", fontweight="bold" if (i == 1 and not ok) else "normal")
                elif M[rr, cc] < 0:
                    ax.text(cc, rr, "÷0", ha="center", va="center", fontsize=6, color="gray")
        if i == 1 and side == 8:
            wrong = (Q != R) & (R >= 0)
            ys, xs = np.where(wrong); ax.scatter(xs, ys, marker="x", c="red", s=12, lw=0.8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title if i == 0 else sub, fontsize=8)
fig.suptitle("Uniformly controlled load, ibm_kingston, TREX only, 16 384 shots (4 × 4) and 65 536 shots (8 × 8); best run shown, red × = wrong pixel", fontsize=8)
fig.tight_layout()
out = REPO / "paper/figures_autonomous/fig_larger_images_hw.png"
fig.savefig(out, dpi=200); print("wrote", out)
