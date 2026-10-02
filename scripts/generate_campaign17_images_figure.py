"""Fig. 12 (v2): the four 64-pixel targets of campaign 17 on ibm_kingston with the truth-table
divider. Left column: the classical quotient image (divide-by-zero pixels blank). Other columns:
the flat-field decode of each pre-registered round, one fixed compilation per target, wrong
pixels marked with a red cross. The per-run counts reproduce scripts/score_campaign17.py.
Usage: .venv/bin/python scripts/generate_campaign17_images_figure.py
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from analyse_hw_signal import analyse
from run_hardware_class_b_nonrestoring import load_dataset

TARGETS = [("random4", "8 × 8 random image"), ("canonical_shared", "8 × 8 Laurdan patch"),
           ("fura2", "8 × 8 Fura-2 image"), ("balanced8", "8 × 8 balanced")]


def runs_for(dataset: str) -> list[tuple[dict, dict]]:
    out = []
    for d in sorted(glob.glob(str(REPO / "data/output/ibm_hw/*/runs" / "c17_*_hw"))):
        m = json.load(open(Path(d) / "metadata.json"))
        if m["label"].startswith("c17_") and m["dataset"] == dataset and m["n"] == 3:
            out.append((m, json.load(open(Path(d) / "counts.json"))))
    return sorted(out, key=lambda mc: (mc[0].get("round") or 0, mc[0]["label"]))


def decode(m: dict, c: dict) -> tuple[np.ndarray, np.ndarray]:
    """Classical quotient image R and the flat-field decode Q, -1 at divide-by-zero pixels."""
    Ia, Ib = load_dataset(m["dataset"], m["n"], m["q"])
    r = analyse(c, (Ia, Ib), m["divider"], m["n"], m["q"])
    P = np.array([px["histogram"] for px in r["pixels"]])
    ff = (P - P.mean(axis=0, keepdims=True)).argmax(axis=1)
    side = 2 ** m["n"]
    R = np.full((side, side), -1); Q = np.full((side, side), -1)
    for k, px in enumerate(r["pixels"]):
        rr, cc = px["pixel"]
        if px["true"] is not None:
            R[rr, cc] = px["true"]; Q[rr, cc] = ff[k]
    return R, Q


def main() -> None:
    cmap = ListedColormap(["#f0f0f0", "#9ecae1", "#3182bd", "#08519c"])
    fig, axes = plt.subplots(len(TARGETS), 4, figsize=(8.6, 9.0))
    pooled = {}
    for i, (ds, title) in enumerate(TARGETS):
        runs = runs_for(ds)
        assert len(runs) == 3, (ds, len(runs))
        decoded = [decode(m, c) for m, c in runs]
        R = decoded[0][0]; valid = int((R >= 0).sum())
        ax = axes[i, 0]
        ax.imshow(np.ma.masked_less(R, 0), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        ax.set_title(f"{title}\nclassical quotient, {valid} valid pixels", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        tot = 0
        for j, ((m, _), (_, Q)) in enumerate(zip(runs, decoded)):
            ax = axes[i, j + 1]
            ax.imshow(np.ma.masked_less(Q, 0), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
            wrong = (Q != R) & (R >= 0)
            ys, xs = np.where(wrong); ax.scatter(xs, ys, marker="x", c="red", s=14, lw=0.9)
            ok = valid - int(wrong.sum()); tot += ok
            ax.set_title(f"round {m.get('round', j + 1)}: {ok}/{valid} ({100 * ok / valid:.0f} %)", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        pooled[ds] = (tot, 3 * valid, m["two_q_gate_count"])
        print(f"{ds:17s} {m['two_q_gate_count']:4d} 2q  pooled flat-field {tot}/{3 * valid} = {100 * tot / (3 * valid):.1f} %")
    fig.suptitle("Campaign 17, truth-table divider, ibm_kingston, TREX only, 65 536 shots; "
                 "flat-field decode of each pre-registered round, red × = wrong pixel", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(k)) for k in range(4)]
    fig.legend(handles, [f"quotient {k}" for k in range(4)], loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out = REPO / "paper/figures_autonomous/fig_campaign17_images.png"
    fig.savefig(out, dpi=300); print("wrote", out)
    # cross-check against the scorer's pooled numbers
    scores = json.load(open(REPO / "paper/data_autonomous/campaign17_scores.json"))
    for ds, (tot, n, _) in pooled.items():
        s = scores[f"{ds}_n3"]
        assert (s["flatfield"], s["pooled"]) == (tot, n), (ds, s["flatfield"], s["pooled"], tot, n)
    print("per-target pooled counts match paper/data_autonomous/campaign17_scores.json")


if __name__ == "__main__":
    main()
