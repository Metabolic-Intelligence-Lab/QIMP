"""
QAE extension to n=3 and n=4 on the Laurdan canonical frame via MPS.

The original qae_demo_class_b.py demonstrated the quadratic-in-queries
advantage at n=1 (2x2) and n=2 (4x4). This script extends the same
MLQAE pipeline to n=3 (8x8) and optionally n=4 (16x16). The total
state-prep ranges from ~35 qubits (n=3) to ~39 qubits (n=4); MPS is
expected to handle n=3 comfortably and to be the practical ceiling
at n=4.

Each run samples at Grover power k in {0, 1} with 512 shots per k by
default, applies the same divzero-aware correction, and reports the
empirical MLQAE estimate vs the classical truth + classical MC stderr
at the same query budget.

Output: paper/data_autonomous/qae_extension.json
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

# qae_demo_class_b lives at scripts/; we reuse build_A/Q/measure routines.
from qae_demo_class_b import (  # noqa: E402
    measure_good_probability,
    mlqae_estimate,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

DATA_OUT = REPO / "paper" / "data_autonomous"


def run_qae_at_n(n: int, shots_per_k: int = 512, max_k: int = 1,
                 threshold: int = 0) -> dict:
    side = 1 << n
    npz_path = DATA_OUT / f"canonical_{side}x{side}.npz"
    d = np.load(npz_path)
    image_a = d["I_a"].astype(np.int64)
    image_b = d["I_b"].astype(np.int64)
    R = np.where(image_b > 0,
                 image_a // np.maximum(image_b, 1), 0)
    valid_mask = image_b > 0
    good_mask = (R > threshold) & valid_mask
    n_pixels = image_a.size
    a_true = float(good_mask.sum()) / float(n_pixels)
    print(f"\n=== QAE Laurdan n={n} ({side}x{side}) ============================")
    print(f"  classical a_true = {a_true:.4f} "
          f"({int(good_mask.sum())} / {n_pixels} good pixels)")
    print(f"  shots per Grover power: {shots_per_k}, max_k = {max_k}")

    probs: dict[int, float] = {}
    timings: dict[int, float] = {}
    for k in range(max_k + 1):
        print(f"  k={k}: building circuit + MPS sampling...", flush=True)
        t0 = time.time()
        p = measure_good_probability(image_a, image_b, q=2,
                                     threshold=threshold,
                                     grover_k=k, shots=shots_per_k)
        timings[k] = time.time() - t0
        probs[k] = p
        print(f"    P(good=1 | k={k}) = {p:.4f}  "
              f"(theoretical sin^2((2k+1)*theta) with theta="
              f"{np.arcsin(np.sqrt(a_true)):.3f})  "
              f"({timings[k]:.1f} s)")

    a_est = mlqae_estimate(probs, shots_per_k)
    total_queries = (max_k + 1) * shots_per_k
    mc_stderr = np.sqrt(a_true * (1 - a_true) / total_queries)
    qae_err = abs(a_est - a_true)
    advantage = mc_stderr / max(qae_err, 1e-9)
    print(f"  MLQAE a_hat = {a_est:.4f}  vs truth a = {a_true:.4f}  "
          f"(|err| = {qae_err:.4g})")
    print(f"  Classical MC stderr at {total_queries} queries: {mc_stderr:.4g}")
    print(f"  Empirical advantage: {advantage:.2f}x")
    return {
        "n": n,
        "side": side,
        "n_pixels": n_pixels,
        "shots_per_k": shots_per_k,
        "max_k": max_k,
        "total_queries": total_queries,
        "a_true": a_true,
        "probs": probs,
        "timings_seconds": timings,
        "a_hat": a_est,
        "abs_err_qae": qae_err,
        "mc_stderr_same_budget": mc_stderr,
        "advantage_ratio": advantage,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-values", type=int, nargs="+", default=[3, 4],
                    help="Which n values to run (default: 3 4).")
    ap.add_argument("--shots-per-k", type=int, default=512)
    ap.add_argument("--max-k", type=int, default=1,
                    help="Largest Grover power. 1 means k in {0,1}, the "
                         "smallest MLQAE protocol with quadratic gain.")
    ap.add_argument("--threshold", type=int, default=0)
    args = ap.parse_args(argv)

    out_path = DATA_OUT / "qae_extension.json"
    results: list[dict] = []
    for n in args.n_values:
        try:
            results.append(run_qae_at_n(n,
                                        shots_per_k=args.shots_per_k,
                                        max_k=args.max_k,
                                        threshold=args.threshold))
        except Exception as e:
            print(f"  *** n={n} FAILED: {e}")
            results.append({"n": n, "error": str(e)[:200]})
        # checkpoint after each n in case the next one crashes
        out_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"  (checkpointed {out_path})")

    print(f"\nFinal write: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
