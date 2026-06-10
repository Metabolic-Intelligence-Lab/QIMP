"""
Bootstrap confidence intervals on the MLQAE / Monte-Carlo log-log RMSE slopes
(addresses the reviewer's request for CIs on the fitted scaling exponents).

Replicates the estimator-scaling experiment of qae_scaling_study.py exactly
(same RNG seed, schedule, shots-per-power, grid, 1000 seeds) but RETAINS the
per-seed error matrix so the slope can be re-fit on bootstrap resamples of the
seed dimension. Reports point slope and the 2.5/97.5 percentile CI.

Outputs paper/data_autonomous/qae_slope_ci.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

# Reuse the exact sampling / MLE / log-log-fit helpers from the scaling study,
# so the point estimate is guaranteed to match it (single source of truth).
from qae_scaling_study import mlqae_mle, p_k, slope  # noqa: E402

# Match qae_scaling_study.py exactly for reproducibility of the point estimate.
RNG = np.random.default_rng(20260529)
SEEDS = 1000
SHOTS_PER_POWER = 64
EIS = [0, 1, 2, 4, 8, 16, 32, 64, 128]
SCHEDULES = [EIS[:j] for j in range(1, len(EIS) + 1)]
GRID = np.linspace(1e-6, np.pi / 2 - 1e-6, 4000)
N_BOOT = 2000
BOOT_RNG = np.random.default_rng(7)


def collect_errors(a_true):
    """Return (Ms, qae_err[n_budgets, seeds], mc_err[n_budgets, seeds])."""
    theta = np.arcsin(np.sqrt(a_true))
    Ms, qae_rows, mc_rows = [], [], []
    for ks in SCHEDULES:
        ks = np.array(ks)
        M = int(SHOTS_PER_POWER * np.sum(2 * ks + 1))
        qae_e = np.empty(SEEDS)
        mc_e = np.empty(SEEDS)
        for s in range(SEEDS):
            hits = RNG.binomial(SHOTS_PER_POWER, p_k(theta, ks))
            a_hat = mlqae_mle(hits, np.full(len(ks), SHOTS_PER_POWER), ks, GRID)
            qae_e[s] = a_hat - a_true
            mc_e[s] = RNG.binomial(M, a_true) / M - a_true
        Ms.append(M)
        qae_rows.append(qae_e)
        mc_rows.append(mc_e)
    return np.array(Ms, float), np.array(qae_rows), np.array(mc_rows)


def rmse_slope(Ms, err_matrix, seed_idx=None):
    if seed_idx is None:
        rmse = np.sqrt(np.mean(err_matrix ** 2, axis=1))
    else:
        rmse = np.sqrt(np.mean(err_matrix[:, seed_idx] ** 2, axis=1))
    return slope(Ms, rmse)


def bootstrap_ci(Ms, err_matrix):
    point = rmse_slope(Ms, err_matrix)
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = BOOT_RNG.integers(0, SEEDS, SEEDS)  # resample seeds w/ replacement
        boot[b] = rmse_slope(Ms, err_matrix, idx)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi), float(np.std(boot))


def main():
    out = {"seeds": SEEDS, "shots_per_power": SHOTS_PER_POWER,
           "n_boot": N_BOOT, "datasets": {}}
    for name, a in (("synthetic_a0.25", 0.25),
                    ("laurdan_a0.75", 0.75),
                    ("fura2_a0.50", 0.50)):
        Ms, qae_e, mc_e = collect_errors(a)
        qp, qlo, qhi, qsd = bootstrap_ci(Ms, qae_e)
        mp, mlo, mhi, msd = bootstrap_ci(Ms, mc_e)
        out["datasets"][name] = {
            "a_true": a,
            "qae_slope": qp, "qae_ci95": [qlo, qhi], "qae_boot_sd": qsd,
            "mc_slope": mp, "mc_ci95": [mlo, mhi], "mc_boot_sd": msd,
        }
        print(f"{name}: QAE {qp:.3f}  (95% CI [{qlo:.3f}, {qhi:.3f}], "
              f"±{qsd:.3f})   MC {mp:.3f} (95% CI [{mlo:.3f}, {mhi:.3f}])")
    (REPO / "paper" / "data_autonomous" / "qae_slope_ci.json").write_text(
        json.dumps(out, indent=2))
    print("wrote qae_slope_ci.json")


if __name__ == "__main__":
    main()
