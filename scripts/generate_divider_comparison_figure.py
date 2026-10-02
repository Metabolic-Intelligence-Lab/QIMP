"""Figure for §6.7: what the truth-table divider changes on hardware.

Left: flat-field decode accuracy by image size, long-division divider (campaigns 7 to 12)
against truth-table divider (campaign 17), each target with its constant-read-out null.
Right: routed two-qubit gates for the same circuits. Both panels read the archived runs, so
the figure regenerates from data.

Usage: .venv/bin/python scripts/generate_divider_comparison_figure.py
"""
from __future__ import annotations
import collections, glob, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from analyse_hw_signal import analyse
from run_hardware_class_b_nonrestoring import load_dataset

# (label, dataset, n, long-division run glob(s), truth-table run glob); the long-division runs are
# pooled as in Table 14
TARGETS = [
    ("$2\\times2$ four-value", "fourvalue", 1, "c4_a_fourvalue_16k_kingston_r*_hw", "c17_fourvalue_n1_lookup_kingston_r*_hw"),
    ("$4\\times4$ tiled", "fourvalue", 2, "c7_a_fourvalue_n2_ucry_kingston_r*_hw", "c17_fourvalue_n2_lookup_kingston_r*_hw"),
    ("$4\\times4$ random", "random4", 2, "c7_b_random4_n2_ucry_kingston_r*_hw", "c17_random4_n2_lookup_kingston_r*_hw"),
    ("$4\\times4$ Laurdan", "canonical_shared", 2, ("c7_c_laurdan_n2_ucry_kingston_r*_hw", "c10_c_laurdan_n2_ucry_kingston_r*_hw"), "c17_canonical_shared_n2_lookup_kingston_r*_hw"),
    ("$8\\times8$ random", "random4", 3, ("c7_d_random4_n3_ucry_kingston_r*_hw", "c8_a_random4_n3_ucry_kingston_r*_hw"), "c17_random4_n3_lookup_kingston_r*_hw"),
    ("$8\\times8$ Laurdan", "canonical_shared", 3, "c10_d_laurdan_n3_ucry_kingston_r*_hw", "c17_canonical_shared_n3_lookup_kingston_r*_hw"),
    ("$8\\times8$ Fura-2", "fura2", 3, "c8_b_fura2_n3_ucry_kingston_r*_hw", "c17_fura2_n3_lookup_kingston_r*_hw"),
    ("$8\\times8$ balanced", "balanced8", 3, "c12_b_balanced8_n3_ucry_kingston_r*_hw", "c17_balanced8_n3_lookup_kingston_r*_hw"),
]

def score(patterns, dataset: str, n: int):
    hits, cx, null = [], [], None
    patterns = (patterns,) if isinstance(patterns, str) else patterns
    for d in [d for pat in patterns for d in sorted(glob.glob(str(REPO / "data/output/ibm_hw/*/runs" / pat)))]:
        d = Path(d); m = json.load(open(d / "metadata.json")); c = json.load(open(d / "counts.json"))
        Ia, Ib = load_dataset(dataset, n, 2)
        r = analyse(c, (Ia, Ib), m.get("divider", "nonrestoring"), n, 2)
        P = np.array([px["histogram"] for px in r["pixels"]]); truth = [px["true"] for px in r["pixels"]]
        valid = [i for i, x in enumerate(truth) if x is not None]
        ff = (P - P.mean(axis=0, keepdims=True)).argmax(axis=1)
        hits.append(sum(ff[i] == truth[i] for i in valid) / len(valid))
        cx.append(m.get("two_q_gate_count"))
        null = max(collections.Counter(truth[i] for i in valid).values()) / len(valid)
    cxs = [c for c in cx if c]
    return (float(np.mean(hits)) if hits else None, float(np.mean(cxs)) if cxs else None, null, len(hits),
            (min(cxs), max(cxs)) if cxs else None)

def main():
    rows = []
    for label, ds, n, old_glob, new_glob in TARGETS:
        old = score(old_glob, ds, n); new = score(new_glob, ds, n)
        rows.append(dict(label=label, dataset=ds, n=n, old_acc=old[0], old_cx=old[1], old_cx_range=old[4], new_acc=new[0], new_cx=new[1],
                         new_cx_range=new[4], null=old[2] if old[2] is not None else new[2], old_runs=old[3], new_runs=new[3]))
        print(f"{label:22s} long division {str(round(old[0]*100,1)) + ' %' if old[0] is not None else '   -   '} "
              f"({old[3]} runs, {old[4]} 2q)   truth table {str(round(new[0]*100,1)) + ' %' if new[0] is not None else '   -   '} "
              f"({new[3]} runs, {new[1]} 2q)   null {round(rows[-1]['null']*100)} %", flush=True)
    json.dump(rows, open(REPO / "paper/data_autonomous/divider_comparison.json", "w"), indent=2)
    have = [r for r in rows if r["new_acc"] is not None]
    if not have:
        print("no campaign-17 runs yet; figure not drawn"); return
    x = np.arange(len(have)); w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].bar(x - w / 2, [100 * (r["old_acc"] or 0) for r in have], w, label="long division", color="C7")
    ax[0].bar(x + w / 2, [100 * r["new_acc"] for r in have], w, label="truth-table synthesis", color="C0")
    for i, r in enumerate(have):
        ax[0].hlines(100 * r["null"], i - 0.5, i + 0.5, color="C3", ls="--", lw=1.2)
    ax[0].hlines([], [], [], color="C3", ls="--", label="constant-read-out null")
    ax[0].set_xticks(x); ax[0].set_xticklabels([r["label"] for r in have], rotation=30, ha="right", fontsize=8)
    ax[0].set_ylabel("flat-field decode (% of valid pixels)"); ax[0].legend(fontsize=8); ax[0].set_ylim(0, 105)
    ax[1].bar(x - w / 2, [r["old_cx"] or 0 for r in have], w, label="long division", color="C7")
    ax[1].bar(x + w / 2, [r["new_cx"] or 0 for r in have], w, label="truth-table synthesis", color="C0")
    ax[1].set_xticks(x); ax[1].set_xticklabels([r["label"] for r in have], rotation=30, ha="right", fontsize=8)
    ax[1].set_ylabel("routed two-qubit gates"); ax[1].legend(fontsize=8)
    fig.tight_layout(); out = REPO / "paper/figures_autonomous/fig_divider_comparison.png"
    fig.savefig(out, dpi=200); print("wrote", out)

if __name__ == "__main__":
    main()
