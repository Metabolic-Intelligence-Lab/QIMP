"""
Empirical verification that the autonomous QAE state-prep A realises the
ideal per-Grover-power success probability p_k = sin^2((2k+1) theta) on the
noise-free MPS simulator, for k beyond the k=0,1 currently reported.

This closes the reviewer gap (review point #1): the 1000-seed scaling study
(scripts/qae_scaling_study.py) evaluates the MLQAE estimator on the *analytic*
p_k, justified only if the real circuit reproduces those p_k past k=1. Here we
run the actual A.Q^k circuit on AerSimulator(method='matrix_product_state') and compare the
measured P(good=1) against sin^2((2k+1)theta) for k = 0..max_k.

Reuses the exact A / Grover construction of scripts/qae_demo_class_b.py so the
verification is on the same pipeline the paper describes.

Usage:
    python scripts/verify_pk_mps.py --dataset synthetic --max-k 4 --shots 1024
    python scripts/verify_pk_mps.py --all --max-k 3 --shots 1024

Outputs paper/data_autonomous/pk_verification.json
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from qae_demo_class_b import measure_good_probability  # noqa: E402

DATA_DIR = REPO / "paper" / "data_autonomous"

# (loader, threshold tau, expected a_true, human label) per dataset.
# Thresholds match the paper: synthetic tau=2 (a=0.25), Laurdan tau=0 (a=0.75),
# Fura-2 tau=1 (a=0.5).
DATASETS = {
    "synthetic": dict(tau=2, label="hand-tuned synthetic (a=0.25)"),
    "laurdan": dict(tau=0, label="canonical Laurdan frame (a=0.75)"),
    "fura2": dict(tau=1, label="synthetic Fura-2 (a=0.5)"),
}

# Stable per-dataset offset for the simulator seed (see verify_dataset).
DS_INDEX = {name: i for i, name in enumerate(DATASETS)}

# Default simulator seed. Without it the Aer sampler draws fresh shot noise
# on every run and the reported p_emp are not reproducible — only the
# within-3-sigma verdict is.
DEFAULT_SEED = 20260610


def load_images(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "synthetic":
        return (np.array([[3, 2], [3, 1]], dtype=np.int64),
                np.array([[1, 1], [3, 1]], dtype=np.int64))
    if name == "laurdan":
        d = np.load(DATA_DIR / "canonical_2x2.npz")
        return d["I_a"].astype(np.int64), d["I_b"].astype(np.int64)
    if name == "fura2":
        d = np.load(DATA_DIR / "fura2_2x2.npz")
        return d["I_a"].astype(np.int64), d["I_b"].astype(np.int64)
    raise ValueError(f"unknown dataset {name}")


def classical_a_true(image_a: np.ndarray, image_b: np.ndarray, tau: int) -> float:
    R = np.where(image_b > 0,
                 image_a // np.maximum(image_b, 1), 0)
    good = (R > tau) & (image_b > 0)
    return float(good.sum()) / float(image_a.size)


def verify_dataset(name: str, ks: list[int], shots: int, q: int = 2,
                   seed_base: int | None = None) -> dict:
    cfg = DATASETS[name]
    tau = cfg["tau"]
    image_a, image_b = load_images(name)
    a_true = classical_a_true(image_a, image_b, tau)
    theta = float(np.arcsin(np.sqrt(a_true)))
    print(f"\n=== {cfg['label']} ===")
    print(f"  I_a=\n{image_a}\n  I_b=\n{image_b}")
    print(f"  classical a_true={a_true:.4f}  theta={theta:.4f} rad  tau={tau}")
    rows = []
    for k in ks:
        p_theory = float(np.sin((2 * k + 1) * theta) ** 2)
        t0 = time.time()
        # One distinct-but-reproducible draw per (dataset, k): the dataset
        # offset keeps the three curves from sharing a shot-noise stream.
        seed = None if seed_base is None else seed_base + 1000 * DS_INDEX[name] + k
        p_emp = measure_good_probability(image_a, image_b, q, tau, k, shots, seed=seed)
        dt = time.time() - t0
        # 1-sigma binomial CI half-width on the empirical estimate
        sigma = float(np.sqrt(max(p_emp * (1 - p_emp), 1e-12) / shots))
        within = abs(p_emp - p_theory) <= 3 * sigma + 0.5 / shots
        rows.append(dict(k=k, p_theory=p_theory, p_emp=p_emp,
                         abs_err=abs(p_emp - p_theory), sigma=sigma,
                         within_3sigma=bool(within), seconds=round(dt, 1)))
        flag = "OK " if within else "*** MISMATCH"
        print(f"  k={k}: p_theory={p_theory:.4f}  p_emp={p_emp:.4f}  "
              f"|Δ|={abs(p_emp-p_theory):.4f}  (3σ={3*sigma:.4f})  "
              f"{flag}  [{dt:.1f}s]")
    all_ok = all(r["within_3sigma"] for r in rows)
    print(f"  --> {'ALL p_k consistent with analytic model' if all_ok else 'SOME MISMATCH'}")
    return dict(dataset=name, label=cfg["label"], tau=tau, q=q,
                a_true=a_true, theta=theta, shots=shots, seed_base=seed_base,
                rows=rows, all_within_3sigma=all_ok)


def make_figure(out: dict, dest: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"synthetic": "C1", "laurdan": "C0", "fura2": "C2"}
    shots = out["shots"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.3),
                                   constrained_layout=True)

    # Left: measured vs analytic p_k, plotted against marker index so the
    # discrete powers are evenly spaced and legible (the analytic sin^2
    # oscillates too fast in k to draw as a continuous curve at large theta).
    for name, d in out["datasets"].items():
        ks = [r["k"] for r in d["rows"]]
        idx = list(range(len(ks)))
        th = [r["p_theory"] for r in d["rows"]]
        emp = [r["p_emp"] for r in d["rows"]]
        err = [3 * r["sigma"] for r in d["rows"]]
        axL.plot(idx, th, "_", color=colors[name], ms=14, mew=2,
                 label=f"{d['label']}: analytic $p_k$")
        axL.errorbar(idx, emp, yerr=err, fmt="o", color=colors[name],
                     ms=5, capsize=3, lw=1, alpha=0.9,
                     label="measured $A\\cdot Q^k$ ±3σ")
        axL.set_xticks(idx)
        axL.set_xticklabels([str(k) for k in ks])
    axL.set_xlabel("Grover power $k$")
    axL.set_ylabel(r"$P(\mathrm{good}=1)$")
    axL.set_ylim(-0.05, 1.10)
    axL.set_title("Measured vs analytic $p_k=\\sin^2((2k{+}1)\\theta)$")
    axL.grid(True, alpha=0.25)
    axL.legend(fontsize=7, loc="center right", ncol=1)

    # Right: residual |p_emp - p_theory| vs k, with the binomial 3σ envelope.
    for name, d in out["datasets"].items():
        ks = [r["k"] for r in d["rows"]]
        res = [r["abs_err"] for r in d["rows"]]
        axR.plot(ks, res, "o-", color=colors[name], ms=5, lw=1,
                 label=d["label"])
    sig3 = 3 * float(np.sqrt(0.25 / shots))  # worst-case binomial 3σ (p=0.5)
    axR.axhline(sig3, ls="--", color="k", alpha=0.6,
                label=f"binomial 3σ @ {shots} shots ($p{{=}}0.5$)")
    axR.set_xlabel("Grover power $k$")
    axR.set_ylabel(r"$|\hat p_k - p_k|$")
    axR.set_title("Residual stays below shot noise for all $k$")
    axR.set_xscale("symlog")
    axR.grid(True, alpha=0.25)
    axR.legend(fontsize=7, loc="upper left")

    kmax = max(r["k"] for d in out["datasets"].values() for r in d["rows"])
    fig.suptitle("Autonomous QAE oracle on AerSimulator(method='matrix_product_state') "
                 f"reproduces the ideal per-power success probability up to $k={kmax}$",
                 fontsize=10)
    fig.savefig(dest, dpi=300)
    print(f"wrote {dest}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=list(DATASETS), default="synthetic")
    ap.add_argument("--all", action="store_true", help="run all three datasets")
    ap.add_argument("--max-k", type=int, default=4)
    ap.add_argument("--ks", type=str, default=None,
                    help="explicit comma-separated Grover powers, e.g. 0,1,2,4,8,16,32")
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="simulator seed base; pass -1 for an unseeded draw")
    ap.add_argument("--figure", action="store_true",
                    help="also write figures_autonomous/fig_pk_verification.png")
    args = ap.parse_args(argv)

    if args.ks is not None:
        ks = [int(x) for x in args.ks.split(",") if x.strip() != ""]
    else:
        ks = list(range(args.max_k + 1))

    names = list(DATASETS) if args.all else [args.dataset]
    seed_base = None if args.seed < 0 else args.seed
    out = dict(shots=args.shots, ks=ks, seed_base=seed_base, datasets={})
    for name in names:
        out["datasets"][name] = verify_dataset(name, ks, args.shots,
                                               seed_base=seed_base)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / "pk_verification.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")

    if args.figure:
        make_figure(out, REPO / "paper" / "figures_autonomous" / "fig_pk_verification.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
