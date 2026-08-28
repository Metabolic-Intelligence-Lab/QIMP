"""The balanced-target campaign behind Table 9 (§7.4).

§7.3 shows the canonical Laurdan patch cannot separate a recovered quotient
from a read-out bias, because its quotient is 1 at every non-divzero pixel.
§7.4 shows TREX-only is the mitigation at which signal survives. This
script pools the runs that combine the two -- the discriminating target at
the winning mitigation -- and computes everything Table 9 reports:

  * the match distribution against the constant-read-out null, with an
    exact binomial test of "does the match exceed that null more often
    than chance";
  * the peakedness (top bin, sigma over the uniform null) per run;
  * the per-pixel contrast d = p(bin 1) - p(bin 0), split by the pixel's
    true quotient. Its separation says whether the read-out carries ratio
    information at all; the global mean says how much bias rides on top.

The last of these is the point of the whole exercise. A match count is an
argmax and collapses the histogram to one integer; d keeps the shape, and
on this target the two disagree.

Usage:
    python scripts/analyse_balanced_target.py
    python scripts/analyse_balanced_target.py --json out.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from analyse_hw_signal import analyse, load_images  # noqa: E402

HW = REPO / "data" / "output" / "ibm_hw"

# The balanced shared-scale target: R = [[1,0],[0,1]], no divzero pixel, so
# every pixel is a genuine quotient match and a constant emitter caps at 2/4.
ARMS = {
    "nonrestoring": ("j16_", "j17_", "j17b_"),
    "restoring": ("j18_",),
}
CONST_NULL = 2
N_PIXELS = 4


def collect(prefixes: tuple[str, ...]) -> list[tuple[dict, dict]]:
    out = []
    for pref in prefixes:
        for d in sorted(glob.glob(str(HW / "*" / "runs" / f"{pref}*_hw"))):
            dp = Path(d)
            try:
                counts = json.load(open(dp / "counts.json"))
                meta = json.load(open(dp / "metadata.json"))
            except OSError:
                continue
            out.append((meta, counts))
    return out


def binom_p(k: int, n: int, p: float = 0.5) -> float:
    """Two-sided exact binomial tail, without a scipy dependency."""
    from math import comb

    if n == 0:
        return float("nan")
    probs = [comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(n + 1)]
    return float(sum(x for x in probs if x <= probs[k] + 1e-12))


def analyse_arm(divider: str, prefixes: tuple[str, ...], images) -> dict | None:
    runs = collect(prefixes)
    if not runs:
        return None
    per_run, d_by_truth = [], {1: [], 0: []}
    for meta, counts in runs:
        r = analyse(counts, images, divider, 1, 2)
        match = sum(1 for px in r["pixels"] if px["argmax_is_true"])
        for px in r["pixels"]:
            h = px["histogram"]
            d_by_truth[int(px["true"])].append(h[1] - h[0])
        per_run.append({
            "label": meta["label"],
            "cx": meta["two_q_gate_count"],
            "top_share": r["mean_top_share"],
            "sigma": r["mean_sigma_over_null"],
            "match": match,
            "decoded": r["pixels"],
        })

    m = np.array([x["match"] for x in per_run], float)
    ones, zeros = np.array(d_by_truth[1]), np.array(d_by_truth[0])
    sep = ones.mean() - zeros.mean()
    se = float(np.sqrt(ones.var(ddof=1) / len(ones) + zeros.var(ddof=1) / len(zeros)))
    above = int((m > CONST_NULL).sum())

    return {
        "divider": divider,
        "n_runs": len(per_run),
        "cx_range": [int(min(x["cx"] for x in per_run)), int(max(x["cx"] for x in per_run))],
        "match_mean": round(float(m.mean()), 3),
        "match_sd": round(float(m.std(ddof=1)), 3),
        "match_hist": {str(int(v)): int((m == v).sum()) for v in sorted(set(m))},
        "constant_readout_null": CONST_NULL,
        "runs_above_null": above,
        "binomial_p_above_null": round(binom_p(above, len(m)), 4),
        "top_share_range": [round(min(x["top_share"] for x in per_run), 3),
                            round(max(x["top_share"] for x in per_run), 3)],
        "sigma_range": [round(min(x["sigma"] for x in per_run), 1),
                        round(max(x["sigma"] for x in per_run), 1)],
        "d_true_one": [round(float(ones.mean()), 4), round(float(ones.std(ddof=1)), 4)],
        "d_true_zero": [round(float(zeros.mean()), 4), round(float(zeros.std(ddof=1)), 4)],
        "separation": round(float(sep), 4),
        "separation_se": round(se, 4),
        "separation_sigma": round(float(sep / se), 2) if se > 0 else None,
        "global_bias": round(float(np.concatenate([ones, zeros]).mean()), 4),
        "per_run": [{k: v for k, v in x.items() if k != "decoded"} for x in per_run],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="analyse_balanced_target", description=__doc__)
    ap.add_argument("--json", type=Path,
                    default=REPO / "paper" / "data_autonomous" / "balanced_target.json")
    args = ap.parse_args(argv)

    images = load_images("canonical_shared", 1, 2)
    out = {}
    for divider, prefixes in ARMS.items():
        res = analyse_arm(divider, prefixes, images)
        if res is None:
            print(f"  (no runs yet for the {divider} arm)")
            continue
        out[divider] = res
        print(f"\n=== {divider}, balanced target, TREX only ({res['n_runs']} runs, "
              f"{res['cx_range'][0]}–{res['cx_range'][1]} CX) ===")
        print(f"  match           = {res['match_mean']:.2f} ± {res['match_sd']:.2f} "
              f"/ {N_PIXELS}   (constant-read-out null {CONST_NULL}/{N_PIXELS})")
        print(f"  distribution    = " +
              ", ".join(f"{k}/4 x{v}" for k, v in res["match_hist"].items()))
        print(f"  above the null  = {res['runs_above_null']}/{res['n_runs']} runs, "
              f"exact binomial p = {res['binomial_p_above_null']:.3f}")
        print(f"  top bin         = {100*res['top_share_range'][0]:.1f}–"
              f"{100*res['top_share_range'][1]:.1f} %   "
              f"({res['sigma_range'][0]}–{res['sigma_range'][1]} σ over uniform)")
        print(f"  d at R=1 pixels = {res['d_true_one'][0]:+.4f} ± {res['d_true_one'][1]:.4f}")
        print(f"  d at R=0 pixels = {res['d_true_zero'][0]:+.4f} ± {res['d_true_zero'][1]:.4f}")
        print(f"  separation      = {res['separation']:+.4f} ± {res['separation_se']:.4f}"
              f"  = {res['separation_sigma']} σ")
        print(f"  global bias     = {res['global_bias']:+.4f}"
              f"   (argmax collapses when this exceeds the separation)")

    args.json.write_text(json.dumps(out, indent=2))
    print(f"\nWritten to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
