"""Is the -0.9 slope the scaling, or the budget? (§6.5.7, extended)

§6.5.7 fits an MLQAE RMSE slope of ~M^-0.9 over a budget range topping out
at k = 128, and reports it as "empirically steeper than Monte Carlo over
the tested budget". A reviewer's obvious question is whether -0.9 is the
asymptotic behaviour or a finite-budget artefact: an estimator approaching
-1 from above would look exactly like this over one decade.

The test is to extend the schedule by a further decade and refit the slope
in sliding windows. If -0.9 is finite-budget bias the windowed slope drifts
toward -1; if it is a genuine plateau it stays put.

Scope. This extends the *estimator* study, which resamples Bernoulli hits
from the ideal p_k = sin^2((2k+1) theta). It says nothing about whether the
autonomous oracle still realises those p_k past k = 128 -- Table S2 verifies
the circuit only that far, and extending *that* is a separate MPS run. The
two questions are deliberately kept apart: this one is about the estimator,
and it is answerable now.

Usage:
    python scripts/qae_slope_drift.py
    python scripts/qae_slope_drift.py --seeds 2000 --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from qae_scaling_study import bootstrap_slope_ci, p_k, slope  # noqa: E402


def refined_mle(hits, shots, ks, n_coarse: int, n_refine: int = 4000,
                stages: int = 2) -> float:
    """Maximum-likelihood theta without a grid floor.

    The published study maximises the Bernoulli log-likelihood over a fixed
    4000-point theta grid. That grid quantises the estimate at ~3.4e-4 in a,
    which the estimator reaches by M ~ 1.7e4 -- inside the published budget
    range -- after which the measured RMSE is the grid spacing rather than
    the estimator's error. Two problems follow, and both are fixed here:

      * **resolution**: refine locally around the coarse argmax, so the
        floor drops by (2/n_refine) per stage and leaves the estimator's
        own error as the only thing measured.
      * **aliasing**: sin^2((2k+1) theta) oscillates with period
        pi/(2k+1), so the coarse grid has to resolve the *fastest* power
        in the schedule or the argmax can land in the wrong mode. n_coarse
        is set from k_max by the caller for exactly this reason.
    """
    lo, hi = 1e-6, np.pi / 2 - 1e-6
    n = n_coarse
    best = None
    for _ in range(stages + 1):
        grid = np.linspace(lo, hi, n)
        ll = np.zeros_like(grid)
        for h, sh, k in zip(hits, shots, ks):
            p = np.clip(np.sin((2 * k + 1) * grid) ** 2, 1e-12, 1 - 1e-12)
            ll += h * np.log(p) + (sh - h) * np.log(1 - p)
        i = int(np.argmax(ll))
        best = grid[i]
        step = grid[1] - grid[0]
        lo, hi = max(1e-6, best - step), min(np.pi / 2 - 1e-6, best + step)
        n = n_refine
    return float(np.sin(best) ** 2)


def run_scaling_refined(a_true, schedules, shots_per_power, seeds, rng):
    """run_scaling from the published study, with the grid floor removed."""
    theta = np.arcsin(np.sqrt(a_true))
    rows = []
    qae_err_mat = np.empty((len(schedules), seeds))
    mc_err_mat = np.empty((len(schedules), seeds))
    for i, ks in enumerate(schedules):
        ks = np.array(ks)
        M = int(shots_per_power * np.sum(2 * ks + 1))
        # Resolve the fastest oscillation in the schedule: >= 40 coarse
        # points per period of the k_max term.
        n_coarse = max(4000, int(40 * (2 * ks.max() + 1)))
        qae_err = np.empty(seeds)
        mc_err = np.empty(seeds)
        for s in range(seeds):
            hits = rng.binomial(shots_per_power, p_k(theta, ks))
            a_hat = refined_mle(hits, np.full(len(ks), shots_per_power), ks, n_coarse)
            qae_err[s] = a_hat - a_true
            mc_err[s] = rng.binomial(M, a_true) / M - a_true
        qae_err_mat[i] = qae_err
        mc_err_mat[i] = mc_err
        rows.append({
            "max_k": int(ks.max()), "n_powers": len(ks), "M": M,
            "n_coarse_grid": n_coarse,
            "qae_rmse": float(np.sqrt(np.mean(qae_err ** 2))),
            "mc_rmse": float(np.sqrt(np.mean(mc_err ** 2))),
        })
    return rows, qae_err_mat, mc_err_mat

DATASETS = (("synthetic_a0.25", 0.25), ("laurdan_a0.75", 0.75), ("fura2_a0.50", 0.50))


def windowed_slopes(Ms, rmse, width: int = 5) -> list[dict]:
    """Refit the log-log slope in overlapping windows along the budget axis."""
    out = []
    for i in range(len(Ms) - width + 1):
        xs, ys = Ms[i:i + width], rmse[i:i + width]
        out.append({
            "M_lo": int(xs[0]),
            "M_hi": int(xs[-1]),
            "slope": round(float(slope(xs, ys)), 3),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qae_slope_drift", description=__doc__)
    ap.add_argument("--seeds", type=int, default=1000)
    ap.add_argument("--shots-per-power", type=int, default=64)
    ap.add_argument("--json", type=Path,
                    default=REPO / "paper" / "data_autonomous" / "qae_slope_drift.json")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(20260827)
    # The published schedule stops at k = 128; four more doublings add
    # rather more than a decade of query budget.
    eis_published = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    eis_extended = eis_published + [256, 512, 1024, 2048]
    schedules = [eis_extended[:j] for j in range(1, len(eis_extended) + 1)]

    out = {"seeds": args.seeds, "shots_per_power": args.shots_per_power,
           "k_max_published": max(eis_published), "k_max_extended": max(eis_extended),
           "datasets": {}}

    for name, a in DATASETS:
        rows, qae_err_mat, _ = run_scaling_refined(
            a, schedules, args.shots_per_power, args.seeds, rng)
        Ms = np.array([r["M"] for r in rows])
        qae = np.array([r["qae_rmse"] for r in rows])

        n_pub = len(eis_published)
        s_pub = float(slope(Ms[:n_pub], qae[:n_pub]))
        s_all = float(slope(Ms, qae))
        ci_all = bootstrap_slope_ci(Ms, qae_err_mat)
        wins = windowed_slopes(Ms, qae)

        out["datasets"][name] = {
            "a_true": a,
            "slope_published_range": round(s_pub, 3),
            "slope_extended_range": round(s_all, 3),
            "slope_extended_ci95": [round(v, 3) for v in ci_all["slope_ci95"]],
            "windowed": wins,
            "rows": rows,
        }

        print(f"\n=== {name} (a = {a}) ===")
        print(f"  slope over the published range (k<=128):  {s_pub:+.3f}")
        print(f"  slope over the extended range (k<=2048):  {s_all:+.3f}  "
              f"[95% CI {ci_all['slope_ci95'][0]:+.3f}, {ci_all['slope_ci95'][1]:+.3f}]")
        print("  windowed slopes along the budget axis:")
        for w in wins:
            print(f"    M {w['M_lo']:>8d} .. {w['M_hi']:>9d}   slope {w['slope']:+.3f}")

    args.json.write_text(json.dumps(out, indent=2))
    print(f"\nWritten to {args.json}")
    print(
        "\n  A windowed slope drifting toward -1 as M grows means the published\n"
        "  -0.9 is a finite-budget reading of an asymptotically Heisenberg\n"
        "  estimator; a flat sequence means -0.9 is what the schedule delivers.\n"
        "  Either way the comparison to Monte Carlo's -0.5 is unaffected."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
