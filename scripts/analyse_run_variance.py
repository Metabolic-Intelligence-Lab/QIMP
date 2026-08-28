"""Is the reported sigma the right sigma? (§7.2 metric, checked against drift)

Every margin in §7 is a *binomial* sigma: the winning bin's lead over the
uniform null, in units of sqrt(p(1-p)/shots) at that position branch. That
is the correct scale for counting statistics within one job, and it is the
only one available for a single run. But the runs are not single: the
mitigation factorial and the fan-out re-runs submit the same circuit five
times, minutes apart, and a superconducting device drifts between jobs.
If the between-run spread of the top-bin share exceeds what the binomial
predicts, the binomial sigma understates the true uncertainty and the
reported margins are optimistic.

This compares the two directly on every repeated arm in the archive:

  * **binomial sd** -- sqrt(p(1-p)/shots), the within-job counting scale.
  * **between-run sd** -- the observed scatter of the top-bin share over
    the repeats of the same arm.

Their ratio is an excess-variance factor. A factor near 1 means the
published sigmas stand as they are; a factor well above 1 means the arm
should be quoted with an error bar built from the repeats instead.

Usage:
    python scripts/analyse_run_variance.py
    python scripts/analyse_run_variance.py --json out.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from analyse_hw_signal import analyse, load_images  # noqa: E402

HW = REPO / "data" / "output" / "ibm_hw"


def arm_key(meta: dict) -> tuple:
    """Group repeats of one arm: strip the _rN / _smoke suffix off the label."""
    label = re.sub(r"_(r\d+|smoke)$", "", meta["label"])
    return (label, meta.get("dataset"), meta.get("mitigation"),
            meta.get("divider"), meta.get("n"), meta.get("q"),
            meta.get("backend"))


def collect() -> dict[tuple, list[dict]]:
    arms: dict[tuple, list[dict]] = collections.defaultdict(list)
    for summary_path in sorted(glob.glob(str(HW / "*" / "summary.json"))):
        run_dir = Path(summary_path).parent / "runs"
        payload = json.load(open(summary_path))
        for label, meta in payload.items():
            if label == "_aggregate" or not isinstance(meta, dict):
                continue
            if "job_id" not in meta or meta.get("dataset") is None:
                continue
            counts_file = run_dir / f"{label}_hw" / "counts.json"
            if not counts_file.exists():
                continue
            try:
                counts = json.load(open(counts_file))
            except Exception:
                continue
            meta = dict(meta)
            meta["_counts"] = counts
            arms[arm_key(meta)].append(meta)
    return arms


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="analyse_run_variance", description=__doc__)
    ap.add_argument("--min-repeats", type=int, default=3)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    arms = collect()
    rows = []
    for key, runs in sorted(arms.items()):
        if len(runs) < args.min_repeats:
            continue
        label, dataset, mitigation, divider, n, q, backend = key
        try:
            images = load_images(dataset, n, q)
        except Exception:
            continue

        per_run_top, per_run_binom, per_run_sigma = [], [], []
        for meta in runs:
            try:
                res = analyse(meta["_counts"], images, divider, n, q)
            except Exception:
                continue
            shares = [p["top_share"] for p in res["pixels"]]
            shots = meta["shots"] / max(len(res["pixels"]), 1)
            per_run_top.append(float(np.mean(shares)))
            per_run_binom.append(
                float(np.mean([np.sqrt(s * (1 - s) / shots) for s in shares]))
            )
            per_run_sigma.append(res["mean_sigma_over_null"])
        if len(per_run_top) < args.min_repeats:
            continue

        top = np.array(per_run_top)
        binom_sd = float(np.mean(per_run_binom))
        between_sd = float(top.std(ddof=1))
        excess = between_sd / binom_sd if binom_sd > 0 else float("inf")
        null = 1.0 / (1 << q)
        # Margin quoted against the scatter of the repeats rather than the
        # counting noise of one job: the standard error of the arm's mean.
        sem = between_sd / np.sqrt(len(top))
        sigma_repeats = (top.mean() - null) / sem if sem > 0 else float("inf")

        rows.append({
            "arm": label, "dataset": dataset, "mitigation": mitigation,
            "divider": divider, "n": n, "q": q, "backend": backend,
            "n_repeats": len(top),
            "mean_top_share": round(float(top.mean()), 4),
            "binomial_sd": round(binom_sd, 5),
            "between_run_sd": round(between_sd, 5),
            "excess_variance_factor": round(excess, 2),
            "sigma_binomial_reported": round(float(np.mean(per_run_sigma)), 2),
            "sigma_from_repeats": round(float(sigma_repeats), 2),
        })

    hdr = (f"{'arm':38s} {'mit':9s} {'N':>2s} {'top':>7s} {'binom sd':>9s} "
           f"{'run sd':>8s} {'excess':>7s} {'sig_bin':>8s} {'sig_rep':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['arm'][:38]:38s} {str(r['mitigation'])[:9]:9s} {r['n_repeats']:2d} "
              f"{r['mean_top_share']:7.3f} {r['binomial_sd']:9.4f} "
              f"{r['between_run_sd']:8.4f} {r['excess_variance_factor']:7.2f} "
              f"{r['sigma_binomial_reported']:8.1f} {r['sigma_from_repeats']:8.1f}")
    print(
        "\n  'excess' = between-run sd / binomial sd. Near 1: the published\n"
        "  binomial sigmas stand. Well above 1: the device drifts between\n"
        "  jobs by more than counting noise, and 'sig_rep' -- the margin\n"
        "  quoted against the scatter of the repeats -- is the honest one."
    )

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"\nWritten to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
