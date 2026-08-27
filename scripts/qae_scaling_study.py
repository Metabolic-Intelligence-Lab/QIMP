"""
MLQAE error scaling in the query budget M (addresses the reviewer's
'k={0,1} does not demonstrate the scaling' point). M counts oracle calls,
not shots -- see the cost convention of the manuscript's §6.4.

Two parts:
 (1) Empirical verification that the autonomous oracle realises the ideal
     per-power success probability p_k = sin^2((2k+1) theta) on the
     noise-free MPS simulator. This is performed by the companion script
     scripts/verify_pk_mps.py, which runs the actual A.Q^k circuit at
     k = 0,1,2,3,4 on all three datasets and confirms agreement with the
     analytic p_k within binomial 3-sigma (paper Table S2). Given that
     match, the estimator-scaling statistics below are a property of these
     p_k and can be evaluated by Bernoulli resampling.
 (2) MLQAE estimator scaling: for an increasing Grover schedule and a range
     of total *query* budgets M = sum_j n_j (2 k_j + 1), run S independent
     seeds, fit a-hat by maximum likelihood, and report RMSE(M). Compare to
     classical Monte Carlo (RMSE = sqrt(a(1-a)/M)). Fit log-log slopes:
     QAE ~ -1, MC ~ -1/2.

Outputs paper/data_autonomous/qae_scaling.json and
paper/figures_autonomous/fig_qae_scaling.png.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

RNG = np.random.default_rng(20260529)
BOOT_RNG = np.random.default_rng(20260610)   # bootstrap-CI generator (separate stream)

def p_k(theta, k):
    return np.sin((2*k+1)*theta)**2

def mlqae_mle(hits, shots, ks, grid):
    # combined Bernoulli log-likelihood over a fine theta grid
    ll = np.zeros_like(grid)
    for h, n, k in zip(hits, shots, ks):
        p = np.clip(np.sin((2*k+1)*grid)**2, 1e-12, 1-1e-12)
        ll += h*np.log(p) + (n-h)*np.log(1-p)
    return np.sin(grid[np.argmax(ll)])**2

def run_scaling(a_true, schedules, shots_per_power, seeds, grid):
    theta = np.arcsin(np.sqrt(a_true))
    rows = []
    qae_err_mat = np.empty((len(schedules), seeds))   # per-seed errors, kept for bootstrap
    mc_err_mat = np.empty((len(schedules), seeds))
    for i, ks in enumerate(schedules):
        ks = np.array(ks)
        M = int(shots_per_power * np.sum(2*ks+1))   # query budget
        qae_err = np.empty(seeds); mc_err = np.empty(seeds)
        for s in range(seeds):
            hits = RNG.binomial(shots_per_power, p_k(theta, ks))
            a_hat = mlqae_mle(hits, np.full(len(ks), shots_per_power), ks, grid)
            qae_err[s] = a_hat - a_true
            # classical MC at the SAME query budget M
            mc_hat = RNG.binomial(M, a_true) / M
            mc_err[s] = mc_hat - a_true
        qae_err_mat[i] = qae_err
        mc_err_mat[i] = mc_err
        rows.append({
            "max_k": int(ks.max()), "n_powers": len(ks), "M": M,
            "qae_rmse": float(np.sqrt(np.mean(qae_err**2))),
            "qae_mae": float(np.mean(np.abs(qae_err))),
            "qae_p05": float(np.percentile(np.abs(qae_err),5)),
            "qae_p95": float(np.percentile(np.abs(qae_err),95)),
            "mc_rmse": float(np.sqrt(np.mean(mc_err**2))),
        })
    return rows, qae_err_mat, mc_err_mat

def slope(xs, ys):
    lx, ly = np.log10(xs), np.log10(ys)
    return float(np.polyfit(lx, ly, 1)[0])

def bootstrap_slope_ci(Ms, err_mat, n_boot=2000):
    """Nonparametric bootstrap CI on the log-log RMSE slope.

    Each RMSE(M) point is estimated from `seeds` i.i.d. per-seed errors
    (err_mat row). For each bootstrap replicate we resample the seeds with
    replacement *independently per M* (the original draws are independent
    across M), recompute RMSE(M), and refit the slope. Returns the mean,
    standard error, and 95% percentile interval of the bootstrap slope
    distribution. Reproducible via the module-level BOOT_RNG seed.
    """
    lx = np.log10(np.asarray(Ms, dtype=float))
    n_pts, seeds = err_mat.shape
    slopes = np.empty(n_boot)
    for b in range(n_boot):
        rmse_b = np.empty(n_pts)
        for i in range(n_pts):
            idx = BOOT_RNG.integers(0, seeds, size=seeds)
            rmse_b[i] = np.sqrt(np.mean(err_mat[i, idx]**2))
        slopes[b] = np.polyfit(lx, np.log10(rmse_b), 1)[0]
    return {
        "slope_boot_mean": float(np.mean(slopes)),
        "slope_se": float(np.std(slopes, ddof=1)),
        "slope_ci95": [float(np.percentile(slopes, 2.5)),
                       float(np.percentile(slopes, 97.5))],
        "n_boot": n_boot,
    }

def main():
    grid = np.linspace(1e-6, np.pi/2-1e-6, 4000)
    seeds = 1000
    shots_per_power = 64
    # EIS (exponentially-increasing) Grover schedules — prefixes of
    # k = 0,1,2,4,8,16,...; these realise the Heisenberg-like ~M^{-1}
    # scaling (Suzuki et al. 2020), unlike a linearly-increasing schedule.
    eis = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    schedules = [eis[:j] for j in range(1, len(eis)+1)]
    out = {"seeds": seeds, "shots_per_power": shots_per_power, "datasets": {}}
    for name, a in (("synthetic_a0.25", 0.25), ("laurdan_a0.75", 0.75),
                    ("fura2_a0.50", 0.50)):
        rows, qae_err_mat, mc_err_mat = run_scaling(
            a, schedules, shots_per_power, seeds, grid)
        Ms = np.array([r["M"] for r in rows])
        qae = np.array([r["qae_rmse"] for r in rows])
        mc  = np.array([r["mc_rmse"] for r in rows])
        qae_ci = bootstrap_slope_ci(Ms, qae_err_mat)
        mc_ci  = bootstrap_slope_ci(Ms, mc_err_mat)
        out["datasets"][name] = {
            "a_true": a, "rows": rows,
            "qae_slope": slope(Ms, qae), "mc_slope": slope(Ms, mc),
            "qae_slope_ci": qae_ci, "mc_slope_ci": mc_ci,
        }
        lo, hi = qae_ci["slope_ci95"]
        print(f"{name}: QAE slope={slope(Ms,qae):.2f} "
              f"[95% CI {lo:.2f}, {hi:.2f}, SE {qae_ci['slope_se']:.3f}]  "
              f"MC slope={slope(Ms,mc):.2f} "
              f"[95% CI {mc_ci['slope_ci95'][0]:.2f}, {mc_ci['slope_ci95'][1]:.2f}]")
    (REPO/"paper"/"data_autonomous"/"qae_scaling.json").write_text(
        json.dumps(out, indent=2))

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7,5), constrained_layout=True)
    colors = {"synthetic_a0.25":"C1","laurdan_a0.75":"C0","fura2_a0.50":"C2"}
    for name, d in out["datasets"].items():
        Ms = np.array([r["M"] for r in d["rows"]])
        qae = np.array([r["qae_rmse"] for r in d["rows"]])
        mc  = np.array([r["mc_rmse"] for r in d["rows"]])
        lo, hi = d["qae_slope_ci"]["slope_ci95"]
        ax.loglog(Ms, qae, "o-", color=colors[name],
                  label=f"{name} MLQAE (slope {d['qae_slope']:.2f}, "
                        f"95% CI [{lo:.2f},{hi:.2f}])")
        ax.loglog(Ms, mc, "s--", color=colors[name], alpha=0.5,
                  label=f"{name} MC (slope {d['mc_slope']:.2f})")
    # reference slopes -1 (Heisenberg/QAE) and -1/2 (MC), anchored at the
    # first QAE point of the Laurdan series for visual comparison.
    allM = np.array([r["M"] for r in out["datasets"]["laurdan_a0.75"]["rows"]])
    q0 = out["datasets"]["laurdan_a0.75"]["rows"][0]["qae_rmse"]
    Mr = np.array([allM.min(), allM.max()], dtype=float)
    ax.loglog(Mr, q0*(Mr/Mr[0])**-1.0, "k-", lw=1, alpha=0.5, label="slope −1 (ideal QAE)")
    ax.loglog(Mr, q0*(Mr/Mr[0])**-0.5, "k:", lw=1, alpha=0.5, label="slope −1/2 (MC)")
    ax.set_xlabel("total query budget M = Σ nⱼ(2kⱼ+1)")
    ax.set_ylabel("RMSE of â over 1000 seeds")
    ax.set_title("MLQAE vs Monte Carlo error scaling in the query budget $M$\n"
                 "(exponentially-increasing Grover schedule k=0,1,2,4,8,…)")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(REPO/"paper"/"figures_autonomous"/"fig_qae_scaling.png", dpi=300)
    print("wrote fig_qae_scaling.png and qae_scaling.json")

if __name__ == "__main__":
    main()
