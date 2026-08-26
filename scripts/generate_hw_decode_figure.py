"""
fig10_hw_decode.png — classical vs hardware-decoded Class B ratio.

Visualises J3 (synthetic 2×2) and J4 (Laurdan 2×2) hardware runs side-by-side
with their classical references, so the noise-floor collapse is visible at a
glance alongside the J2 NEQR-only result which preserved the signal.

Reads the persisted summary.json files from data/output/ibm_hw/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
HW_ROOT = REPO / "data" / "output" / "ibm_hw"
OUT = REPO / "paper" / "figures_autonomous" / "fig10_hw_decode.png"


def _load(label: str) -> dict:
    for summary in HW_ROOT.glob("*/summary.json"):
        data = json.loads(summary.read_text())
        for val in data.values():
            if val.get("label") == label:
                return val
    raise FileNotFoundError(f"no summary entry with label={label!r}")


def _draw_grid(ax, values, title, *, vmin=0, vmax=3, cmap="viridis"):
    arr = np.asarray(values, dtype=float)
    im = ax.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{int(arr[i, j])}", ha="center", va="center",
                    color="white" if arr[i, j] < (vmin + vmax) / 2 else "black",
                    fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    return im


def main() -> None:
    j3 = _load("j3_class_b_synthetic")
    j4 = _load("j4_class_b_laurdan")

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.6), constrained_layout=True)

    _draw_grid(axes[0, 0], j3["classical_R"],
               f"J3 synthetic — classical R\n(truth)", vmax=3)
    _draw_grid(axes[0, 1], j3["quantum_quotient"],
               f"J3 synthetic — hardware decode\n"
               f"ibm_kingston, {j3['two_q_gate_count']} CX, 0/4 match",
               vmax=3)

    _draw_grid(axes[1, 0], j4["classical_R"],
               f"J4 Laurdan — classical R\n(truth)", vmax=3)
    _draw_grid(axes[1, 1], j4["quantum_quotient"],
               f"J4 Laurdan — hardware decode\n"
               f"ibm_fez, {j4['two_q_gate_count']} CX, 0/4 match",
               vmax=3)

    fig.suptitle(
        "Class B autonomous ratio: classical truth vs hardware decode\n"
        "Both Heron r2 runs collapse to noise floor "
        f"(predicted $F \\approx e^{{-N_{{2q}} \\epsilon}} \\sim 10^{{-2}}$)",
        fontsize=11,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
