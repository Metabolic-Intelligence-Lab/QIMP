"""Is the high quotient bit really the most error-exposed? (§7.5, from the archive)

§7.5 observes that Fura-2, whose correct quotients are R = 2, floors on two
backends at the same gate count where the Laurdan target (R in {0,1})
clears, and reads this as the high quotient bit being the most
error-exposed -- fixed in the first division iteration and therefore
upstream of the whole remaining cascade. The subsection is careful to say
this is "consistent with the data but not established by it", because the
comparison is between two different targets.

It can be established from the same archive without a QPU, by not
collapsing the histogram. The decoder's argmax throws away which *bit*
failed; marginalising the quotient histogram bit by bit keeps it. For each
run and each bit position j, this reports

    P(bit j correct) = sum of the histogram over quotient values whose
                       j-th bit matches the classical reference,

with its binomial sigma over the 0.5 chance level. If the mechanism is the
one §7.5 proposes, the low bit should carry signal while the high bit sits
at chance -- and crucially this can be read *within a single run on a
single target*, so it no longer depends on comparing Fura-2 against
Laurdan.

Usage:
    python scripts/analyse_quotient_bits.py
    python scripts/analyse_quotient_bits.py --run j16 --json out.json
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
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from analyse_hw_signal import analyse, load_images  # noqa: E402

HW = REPO / "data" / "output" / "ibm_hw"


def bit_accuracy(res: dict, q: int) -> list[dict]:
    """P(bit j correct) per bit position, pooled over the run's pixels.

    A bit whose true value is the *same* at every pixel of the target
    measures nothing about discrimination: the device scores on it by
    having a read-out bias that happens to point the right way, exactly
    the constant-emitter failure of §7.2 one level further down. On the
    balanced target R = [[1,0],[0,1]] the high bit is 0 everywhere, so a
    device that simply never emits a quotient >= 2 scores ~1.0 on it. Such
    bits are marked `varies: False` and must not be read as recovery.
    """
    out = []
    for j in range(q):
        probs, true_bits = [], []
        for px in res["pixels"]:
            if px["divzero"]:
                continue  # the flag, not the quotient, carries these
            true_bit = (int(px["true"]) >> j) & 1
            true_bits.append(true_bit)
            h = np.asarray(px["histogram"], dtype=float)
            probs.append(float(sum(h[v] for v in range(len(h))
                                   if ((v >> j) & 1) == true_bit)))
        if not probs:
            continue
        out.append({
            "bit": j,
            "p_correct": round(float(np.mean(probs)), 4),
            "varies": len(set(true_bits)) > 1,
            "n_pixels": len(probs),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="analyse_quotient_bits", description=__doc__)
    ap.add_argument("--run", type=str, default=None, help="substring filter on label")
    ap.add_argument("--json", type=Path,
                    default=REPO / "paper" / "data_autonomous" / "quotient_bits.json")
    args = ap.parse_args(argv)

    # Group by (dataset, mitigation, divider, n, q) and pool the repeats:
    # a single 4-pixel run is too thin to read a per-bit probability from.
    groups: dict[tuple, list[list[dict]]] = collections.defaultdict(list)
    shots_by_group: dict[tuple, int] = {}

    # Walk the per-run directories rather than summary.json: a driver that
    # dies mid-loop (a dropped network connection, in our case) leaves every
    # completed run persisted but never writes the summary, so a
    # summary-driven scan silently under-counts the archive.
    for run_dir in sorted(glob.glob(str(HW / "*" / "runs" / "*_hw"))):
        rd = Path(run_dir)
        try:
            meta = json.load(open(rd / "metadata.json"))
            counts = json.load(open(rd / "counts.json"))
        except OSError:
            continue
        label = meta.get("label", rd.name)
        # Older runs predate some metadata fields; skip what we cannot key.
        needed = ("dataset", "mitigation", "divider", "n", "q", "shots")
        if any(meta.get(k) is None for k in needed) or "job_id" not in meta:
            continue
        if args.run and args.run not in label:
            continue
        key = (meta["dataset"], meta["mitigation"], meta["divider"],
               meta["n"], meta["q"])
        try:
            images = load_images(meta["dataset"], meta["n"], meta["q"])
            res = analyse(counts, images, meta["divider"], meta["n"], meta["q"])
        except Exception:
            continue
        groups[key].append(bit_accuracy(res, meta["q"]))
        shots_by_group[key] = meta["shots"]

    rows = []
    for key, per_run in sorted(groups.items(), key=lambda kv: str(kv[0])):
        dataset, mitigation, divider, n, q = key
        n_runs = len(per_run)
        bits: dict[int, list[float]] = collections.defaultdict(list)
        for run in per_run:
            for b in run:
                bits[b["bit"]].append((b["p_correct"], b["varies"]))
        if not bits:
            continue
        # Effective shot count behind each pooled bit estimate: shots split
        # across position branches, times the runs pooled.
        n_px = (1 << n) ** 2
        eff = shots_by_group[key] / n_px * n_runs
        entry = {"dataset": dataset, "mitigation": mitigation, "divider": divider,
                 "n": n, "q": q, "n_runs": n_runs, "bits": []}
        for j in sorted(bits):
            p = float(np.mean([v for v, _ in bits[j]]))
            varies = any(w for _, w in bits[j])
            sigma = float(np.sqrt(0.25 / eff))
            entry["bits"].append({
                "bit": j,
                "p_correct": round(p, 4),
                "varies": varies,
                "sigma_over_chance": round((p - 0.5) / sigma, 2) if varies else None,
            })
        rows.append(entry)

    hdr = f"{'dataset':16s} {'mit':9s} {'div':13s} {'n,q':>5s} {'runs':>5s}  per-bit P(correct) [sigma over chance]"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        bits = "  ".join(
            (f"b{b['bit']}={b['p_correct']:.3f} [{b['sigma_over_chance']:+.1f}]"
             if b["varies"] else f"b{b['bit']}={b['p_correct']:.3f} (const)")
            for b in r["bits"]
        )
        print(f"{r['dataset'][:16]:16s} {str(r['mitigation'])[:9]:9s} "
              f"{str(r['divider'])[:13]:13s} {r['n']},{r['q']:<3d} {r['n_runs']:5d}  {bits}")
    print(
        "\n  b0 is the low quotient bit, b_{q-1} the high one fixed in the first\n"
        "  division iteration. Chance is 0.5. '(const)' marks a bit whose true\n"
        "  value is identical at every pixel of that target: it measures read-out\n"
        "  bias, not discrimination, and carries no sigma. Only the varying bits\n"
        "  bear on whether a given bit position is recoverable."
    )

    args.json.write_text(json.dumps(rows, indent=2))
    print(f"\nWritten to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
