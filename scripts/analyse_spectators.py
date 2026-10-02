"""Campaigns 11–14 analysis: active read-out and spectator excitation under the
padding conditions, grouped by device. Usage: .venv/bin/python scripts/analyse_spectators.py
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from analyse_hw_signal import analyse, load_images

ORDER = ("nodd", "dd_active", "dd_spect", "dd_all", "dd_spect_matched", "dd_active_bunched", "dd_spect_matched_bunched", "dd_active_f025", "dd_active_f050", "dd_active_f075", "dd_active_f100")

def main():
    images = load_images("canonical_shared", 1, 2)
    runs = {}
    for pat in ("c11_*_hw", "c12_dd_*_hw", "c13_*_hw", "c14_*_hw", "c15_*_hw"):
        for d in sorted(glob.glob(str(REPO / "data/output/ibm_hw/*/runs" / pat))):
            m = json.load(open(Path(d) / "metadata.json")); c = json.load(open(Path(d) / "counts.json"))
            runs.setdefault((m["backend"], m["condition"]), []).append((m, c))
    out = {}
    print(f"{'device':10s} {'condition':24s} runs  top-bin   σ/null   flat-field   separation   spectator P(1) mean ± sd(runs)")
    for be in ("ibm_marrakesh", "ibm_kingston"):
        base = None
        for cond in ORDER:
            key = (be, cond)
            if key not in runs: continue
            tops, sig, ff, d1, d0, sp = [], [], [], [], [], []
            for m, c in runs[key]:
                r = analyse(c, images, "nonrestoring", 1, 2)
                tops.append(r["mean_top_share"]); sig.append(r["mean_sigma_over_null"])
                P = np.array([px["histogram"] for px in r["pixels"]]); t = np.array([px["true"] for px in r["pixels"]])
                Q = P - P.mean(axis=0, keepdims=True); ff.append(int((Q.argmax(axis=1) == t).sum()))
                for px in range(4):
                    (d1 if t[px] == 1 else d0).append(P[px, 1] - P[px, 0])
                sp.append(float(np.mean(m["spectator_p1"])))
            sep = float(np.mean(d1) - np.mean(d0))
            row = dict(n_runs=len(runs[key]), top_share=float(np.mean(tops)), sigma_over_null=float(np.mean(sig)),
                       flatfield_match=float(np.mean(ff)), separation=sep, spectator_p1_mean=float(np.mean(sp)),
                       spectator_p1_sd_runs=float(np.std(sp, ddof=1)) if len(sp) > 1 else 0.0,
                       pulses=int(runs[key][0][0]["single_q_xy"]))
            out[f"{be}:{cond}"] = row
            print(f"{be[4:]:10s} {cond:24s} {row['n_runs']:4d}  {row['top_share']:.3f}   {row['sigma_over_null']:6.2f}   {row['flatfield_match']:.2f}/4     {sep:+.3f}      {row['spectator_p1_mean']:.4f} ± {row['spectator_p1_sd_runs']:.4f}")
            if cond == "nodd": base = row
            elif base:
                print(f"{'':10s}   -> vs nodd: σ/null ratio {row['sigma_over_null']/base['sigma_over_null']:.2f}, separation retained {row['separation']/base['separation']*100:.0f} %")
    json.dump(out, open(REPO / "paper/data_autonomous/campaign11_spectators.json", "w"), indent=2)

if __name__ == "__main__":
    main()
