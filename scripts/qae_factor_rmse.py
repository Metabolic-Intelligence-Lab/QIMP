"""
Rigorous per-dataset query-efficiency factor at the fixed two-power budget
(k in {0,1}, 128 shots each -> M = 512 queries), replacing the single-run
|a_hat - a| / sigma_MC comparison (review point #2) with an RMSE over many
seeds: factor = sigma_MC(M) / RMSE_MLQAE(M).

The MLQAE estimator is the same maximum-likelihood fit used in
scripts/qae_demo_class_b.py and scripts/qae_scaling_study.py; the per-power
success probabilities p_k = sin^2((2k+1)theta) are the ones verified on the
real circuit up to k=128 (scripts/verify_pk_mps.py, paper Table S2).

Outputs paper/data_autonomous/qae_factor_rmse.json.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(20260610)

# Fixed two-power demonstration schedule used throughout paper Sec. 6.5.
KS = (0, 1)
SHOTS_PER_K = 128
M = SHOTS_PER_K * sum(2 * k + 1 for k in KS)   # 128*1 + 128*3 = 512
SEEDS = 20000
GRID = np.linspace(1e-6, np.pi / 2 - 1e-6, 8000)

DATASETS = {
    "synthetic": 0.25,
    "laurdan": 0.75,
    "fura2": 0.50,
}


def mlqae_mle(hits: np.ndarray, ks: tuple[int, ...], shots: int) -> float:
    ll = np.zeros_like(GRID)
    for h, k in zip(hits, ks):
        p = np.clip(np.sin((2 * k + 1) * GRID) ** 2, 1e-12, 1 - 1e-12)
        ll += h * np.log(p) + (shots - h) * np.log(1 - p)
    return float(np.sin(GRID[np.argmax(ll)]) ** 2)


def factor_for(a_true: float) -> dict:
    theta = np.arcsin(np.sqrt(a_true))
    pk = np.sin((2 * np.array(KS) + 1) * theta) ** 2
    qae_err = np.empty(SEEDS)
    mc_err = np.empty(SEEDS)
    for s in range(SEEDS):
        hits = RNG.binomial(SHOTS_PER_K, pk)
        qae_err[s] = mlqae_mle(hits, KS, SHOTS_PER_K) - a_true
        mc_err[s] = RNG.binomial(M, a_true) / M - a_true
    qae_rmse = float(np.sqrt(np.mean(qae_err ** 2)))
    qae_bias = float(np.mean(qae_err))
    mc_sigma = float(np.sqrt(a_true * (1 - a_true) / M))
    mc_rmse = float(np.sqrt(np.mean(mc_err ** 2)))
    return dict(a_true=a_true, M=M, seeds=SEEDS,
                qae_rmse=qae_rmse, qae_bias=qae_bias,
                mc_sigma_analytic=mc_sigma, mc_rmse=mc_rmse,
                factor_rmse=mc_sigma / qae_rmse)


def main() -> int:
    out = dict(ks=list(KS), shots_per_k=SHOTS_PER_K, M=M, seeds=SEEDS, datasets={})
    print(f"Two-power budget M={M} ({SHOTS_PER_K} shots x k in {KS}); "
          f"{SEEDS} seeds.\n")
    print(f"{'dataset':<12} {'a':>5} {'MLQAE RMSE':>11} {'bias':>9} "
          f"{'MC sigma':>9} {'factor':>7}")
    for name, a in DATASETS.items():
        r = factor_for(a)
        out["datasets"][name] = r
        print(f"{name:<12} {a:>5.2f} {r['qae_rmse']:>11.5f} {r['qae_bias']:>+9.5f} "
              f"{r['mc_sigma_analytic']:>9.5f} {r['factor_rmse']:>6.2f}x")
    dest = REPO / "paper" / "data_autonomous" / "qae_factor_rmse.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
