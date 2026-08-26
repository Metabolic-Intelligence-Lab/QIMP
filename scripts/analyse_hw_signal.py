"""Is there quotient signal in a hardware run, or is the decode reading noise?

The per-pixel "match" figure used throughout §7 is an *argmax* over the
quotient histogram of each position branch. An argmax always returns a
crisp integer, including when the histogram it is taken over is flat, so a
match count on its own cannot distinguish recovery from a lucky coin flip.
Two things have to be reported alongside it:

  * **peakedness** -- the share of the largest bin against the uniform null
    1/2^q, in units of the binomial sigma at the branch's shot count. A
    per-pixel readout claim needs the argmax to win by a wide margin, not
    by one sigma.
  * **the null baseline** -- the match count a device that has lost the
    signal still obtains by emitting one constant value at every pixel.
    On a target whose true values are all equal, that baseline is a
    perfect score.

Both are computed here from the counts archived under `data/output/ibm_hw/`,
so any published run can be re-checked without touching a QPU.

Usage:
    python scripts/analyse_hw_signal.py                  # every archived run
    python scripts/analyse_hw_signal.py --run j7         # substring filter
    python scripts/analyse_hw_signal.py --json out.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from qimp.processing.ratiometric_circuit import class_b_ratio  # noqa: E402

HW = REPO / "data" / "output" / "ibm_hw"

# Runs whose metadata predates the `dataset` field but whose inputs are known.
LEGACY_DATASET = {
    "j7_class_b_laurdan_nonrestoring": ("canonical", "nonrestoring", 1, 2),
    "j4_class_b_laurdan": ("canonical", "restoring", 1, 2),
    "j3_class_b_synthetic": ("synthetic", "restoring", 1, 2),
}


def load_images(dataset: str, n: int, q: int) -> tuple[np.ndarray, np.ndarray]:
    from run_hardware_class_b_nonrestoring import load_dataset

    return load_dataset(dataset, n, q)


def analyse(counts: dict[str, int], images, divider: str, n: int, q: int) -> dict:
    I_a, I_b = images
    qc, layout = class_b_ratio(I_a, I_b, q=q, divider=divider)
    total = qc.num_qubits
    pos = layout["position"]
    quo = layout["quotient"]
    n_bits = len(quo)
    n_vals = 1 << n_bits
    null = 1.0 / n_vals

    R_ref = np.where(I_b > 0, I_a // np.maximum(I_b, 1), 0)
    divzero = I_b == 0

    hist: dict[tuple[int, int], np.ndarray] = collections.defaultdict(lambda: np.zeros(n_vals))
    for state, c in counts.items():
        flat = state.replace(" ", "")

        def bit(i: int, flat: str = flat) -> int:
            return int(flat[total - 1 - i])

        col = sum(bit(pos[i]) << i for i in range(n))
        row = sum(bit(pos[n + i]) << i for i in range(n))
        hist[(row, col)][sum(bit(quo[i]) << i for i in range(n_bits))] += c

    pixels = []
    for (row, col), h in sorted(hist.items()):
        shots = float(h.sum())
        p = h / shots
        sigma = np.sqrt(null * (1 - null) / shots)
        top = int(np.argmax(p))
        # margin over the runner-up: how close the argmax was to flipping
        order = np.sort(p)[::-1]
        pixels.append(
            {
                "pixel": [row, col],
                "true": None if divzero[row, col] else int(R_ref[row, col]),
                "divzero": bool(divzero[row, col]),
                "histogram": p.round(4).tolist(),
                "argmax": top,
                "top_share": round(float(p.max()), 4),
                "sigma_over_null": round(float((p.max() - null) / sigma), 2),
                "margin_over_runner_up": round(float(order[0] - order[1]), 4),
                "argmax_is_true": (not divzero[row, col]) and top == int(R_ref[row, col]),
            }
        )

    # Null baseline: best score obtainable by emitting one constant value.
    n_px = R_ref.size
    best_const = max(
        max(int(((R_ref == v) & ~divzero).sum()) for v in range(n_vals)),
        int(divzero.sum()),
    )
    return {
        "n_vals": n_vals,
        "null_share": null,
        "mean_top_share": round(float(np.mean([p["top_share"] for p in pixels])), 4),
        "mean_sigma_over_null": round(float(np.mean([p["sigma_over_null"] for p in pixels])), 2),
        "min_margin_over_runner_up": round(
            float(min(p["margin_over_runner_up"] for p in pixels)), 4
        ),
        "constant_readout_baseline": f"{best_const}/{n_px}",
        "pixels": pixels,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=str, default=None, help="substring filter on label")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true", help="per-pixel histograms")
    args = ap.parse_args(argv)

    out = {}
    print(
        f"{'run':40s} {'CX':>5s} {'match':>7s} {'top bin':>8s} {'sigma':>7s} "
        f"{'min gap':>8s} {'const null':>10s}"
    )
    for md_path in sorted(glob.glob(str(HW / "*" / "runs" / "*" / "metadata.json"))):
        m = json.load(open(md_path))
        label = m.get("label", "")
        if args.run and args.run not in label:
            continue
        if label in LEGACY_DATASET:
            dataset, divider, n, q = LEGACY_DATASET[label]
        else:
            dataset, divider = m.get("dataset"), m.get("divider")
            n, q = m.get("n", 1), m.get("q", 2)
            if not dataset or not divider:
                continue
        try:
            images = load_images(dataset, n, q)
            counts = json.load(open(md_path.replace("metadata.json", "counts.json")))
            res = analyse(counts, images, divider, n, q)
        except Exception as e:
            print(f"  {label:38s} skipped ({type(e).__name__}: {str(e)[:40]})")
            continue
        res.update(
            label=label,
            backend=m.get("backend"),
            cx=m.get("two_q_gate_count"),
            match=m.get("match_count"),
            dataset=dataset,
            divider=divider,
        )
        out[label + "@" + Path(md_path).parts[-2]] = res
        print(
            f"  {label:38s} {res['cx']!s:>5s} "
            f"{res['match']!s:>3s}/{res['n_vals'] and len(res['pixels']):<3d} "
            f"{res['mean_top_share']:7.1%} {res['mean_sigma_over_null']:6.1f}σ "
            f"{res['min_margin_over_runner_up']:7.1%} {res['constant_readout_baseline']:>10s}"
        )
        if args.verbose:
            for p in res["pixels"]:
                t = "divzero" if p["divzero"] else str(p["true"])
                print(
                    f"      pixel {p['pixel']} true={t:>7s} hist={p['histogram']} "
                    f"argmax={p['argmax']} ({p['sigma_over_null']}σ)"
                )

    print(
        "\n  null share = 1/2^q per bin; 'sigma' is how far the winning bin sits\n"
        "  above it. 'min gap' is the smallest argmax-to-runner-up margin in the\n"
        "  run: a per-pixel recovery claim needs that gap to be large.\n"
        "  'const null' is the match a constant-emitting device still scores."
    )
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
