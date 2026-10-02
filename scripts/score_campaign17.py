"""Score campaign 17 against the bands of paper/HW_CAMPAIGN_17_PROTOCOL.md.

Per run: flat-field and argmax accuracy over the valid pixels against the target's
constant-read-out null (P2/P3), and the decoder-free criterion P1 (the most frequent
(q, r, d) triple true in more pixels than the constant null, with a positive identity
excess over the cross-pixel null on the pixels whose quotient is at least 1).
"""
from __future__ import annotations
import argparse, collections, glob, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from analyse_hw_signal import analyse
from arithmetic_identity import analyse_run
from run_hardware_class_b_nonrestoring import load_dataset

def score_run(d: Path) -> dict:
    m = json.load(open(d / "metadata.json")); c = json.load(open(d / "counts.json"))
    Ia, Ib = load_dataset(m["dataset"], m["n"], m["q"])
    r = analyse(c, (Ia, Ib), m["divider"], m["n"], m["q"])
    P = np.array([px["histogram"] for px in r["pixels"]]); truth = [px["true"] for px in r["pixels"]]
    valid = [i for i, x in enumerate(truth) if x is not None]
    am, ff = P.argmax(axis=1), (P - P.mean(axis=0, keepdims=True)).argmax(axis=1)
    null = max(collections.Counter(truth[i] for i in valid).values())
    _, px, (const, const1) = analyse_run(d)
    p1 = [p for p in px if p["q_ge1"]]
    excess1 = float(np.nanmean([p["identity_rate"] - p["identity_cross"] for p in p1])) if p1 else float("nan")
    joint = sum(p["joint_argmax_true"] for p in px)
    return dict(label=m["label"], dataset=m["dataset"], n=m["n"], round=m.get("round"), two_q=m["two_q_gate_count"],
                n_valid=len(valid), null=null, argmax=int(sum(am[i] == truth[i] for i in valid)),
                flatfield=int(sum(ff[i] == truth[i] for i in valid)), joint=joint, joint_null=const,
                joint_q1=sum(p["joint_argmax_true"] for p in p1), n_q1=len(p1), joint_q1_null=const1,
                excess_q1=excess1, P1=bool(joint > const and excess1 > 0))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prefix", default="c17_")
    a = ap.parse_args()
    runs = [score_run(Path(d)) for d in sorted(glob.glob(str(REPO / "data/output/ibm_hw/*/runs" / f"{a.prefix}*_hw")))]
    by_target: dict[str, list[dict]] = {}
    for r in runs:
        by_target.setdefault(f"{r['dataset']}_n{r['n']}", []).append(r)
    print(f"{'target':24s} {'runs':>4} {'2q':>5} {'flat-field':>18} {'argmax':>14} {'null':>8} {'joint (null)':>16} {'P1':>7}")
    out = {}
    for k, rs in by_target.items():
        nv = rs[0]["n_valid"]; ff = sum(r["flatfield"] for r in rs); am = sum(r["argmax"] for r in rs)
        tot = nv * len(rs); above = sum(1 for r in rs if r["flatfield"] > r["null"])
        p1 = sum(r["P1"] for r in rs)
        print(f"{k:24s} {len(rs):4d} {rs[0]['two_q']:5d} {ff:5d}/{tot:<5d} ({100*ff/tot:5.1f} %) {am:5d}/{tot:<5d} ({100*am/tot:4.1f} %) {rs[0]['null']:3d}/{nv:<4d} "
              f"{sum(r['joint'] for r in rs):5d}/{tot:<5d} ({sum(r['joint_null'] for r in rs):4d}) {p1}/{len(rs)}")
        out[k] = dict(runs=len(rs), two_q=rs[0]["two_q"], n_valid=nv, null=rs[0]["null"], flatfield=ff, argmax=am,
                      joint=sum(r["joint"] for r in rs), joint_null=sum(r["joint_null"] for r in rs),
                      pooled=tot, runs_above_null=above, P1_runs=p1,
                      per_run=[{kk: r[kk] for kk in ("label", "round", "flatfield", "argmax", "joint", "joint_q1", "excess_q1", "P1")} for r in rs])
    json.dump(out, open(REPO / "paper/data_autonomous/campaign17_scores.json", "w"), indent=2)

if __name__ == "__main__":
    main()
