"""
Full class_b_ratio with the non-restoring divider on IBM Heron r2,
parametric over dataset / n / q. Strengthens the key J7 result (the
algorithmic CX compression brings the full pipeline above the FT horizon)
by fanning out across the ratiometric-microscopy use-case panel and
across qubit count.

The J-series labelling used in the paper:
  J7  : canonical (Laurdan) n=1 q=2  (679 CX, 4/4)  [the original]
  J8  : synthetic           n=1 q=2  use-case generality
  J9  : fura2               n=1 q=2  use-case generality (calcium)
  J11 : canonical           n=2 q=2  qubit scaling (26 qubits)
  J12 : canonical           n=1 q=3  intensity-width scaling (36 qubits)

Usage:
  python scripts/run_hardware_class_b_nonrestoring.py --dataset fura2 --n 1 --q 2 --label j9_fura2_n1q2_nonrestoring --backend ibm_fez
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from qimp.processing.ratiometric_circuit import (  # noqa: E402
    class_b_ratio, decode_class_b_ratio,
)
from qimp.runtime import ibm  # noqa: E402
from prepare_autonomous_datasets import (  # noqa: E402
    synthesise_fura2, synthesise_rogfp, quantise_to_q,
)
from run_autonomous_n3_laurdan import prepare_canonical  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

DEFAULT_OUTDIR = REPO / "data" / "output" / "ibm_hw"


def utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def load_dataset(dataset: str, n: int, q: int) -> tuple[np.ndarray, np.ndarray]:
    side = 1 << n
    if dataset == "canonical":
        npz = REPO / "paper" / "data_autonomous" / f"canonical_{side}x{side}.npz"
        if q == 2 and npz.exists():
            d = np.load(npz)
            return d["I_a"].astype(np.int64), d["I_b"].astype(np.int64)
        I_a, I_b, _, _ = prepare_canonical(target_n=n)
        return I_a.astype(np.int64), I_b.astype(np.int64)
    if dataset == "synthetic":
        base_a = np.array([[3, 2], [3, 1]], dtype=np.int64)
        base_b = np.array([[1, 1], [3, 1]], dtype=np.int64)
        return (np.tile(base_a, (side // 2, side // 2)),
                np.tile(base_b, (side // 2, side // 2)))
    if dataset == "canonical_shared":
        # Laurdan 2x2 patch quantised on a single SHARED photometric scale
        # (both emission bands are physically on one scale), so the
        # inter-channel contrast is not renormalised away as it is by the
        # per-channel quantisation of `canonical` -- see §5.3 and §6.5.2.
        # Selected by scripts/select_shared_scale_patch.py: balanced
        # R = [[1,0],[0,1]] with no divide-by-zero pixel, so every pixel is
        # a genuine quotient match and a constant readout caps at 50%.
        d = np.load(REPO / "paper" / "data_autonomous"
                    / f"canonical_shared_n{n}_q{q}.npz")
        return d["I_a"].astype(np.int64), d["I_b"].astype(np.int64)
    if dataset == "canonical_nd":
        # Non-degenerate Laurdan 2x2 patch (I_a != I_b, R spans {0,1,2});
        # offset (94,60) of the canonical frame, q=2. See §6.5.9.
        d = np.load(REPO / "paper" / "data_autonomous" / "canonical_2x2_nd.npz")
        return d["I_a"].astype(np.int64), d["I_b"].astype(np.int64)
    if dataset == "fourvalue":
        # Campaign-3 target (10 Sep 2026, paper/HW_CAMPAIGN_3_PROTOCOL.md):
        # every quotient value of the q=2 codomain appears exactly once,
        # R = [[0,1],[2,3]] from I_a=[[1,3],[2,3]], I_b=[[2,3],[1,1]], no
        # divide-by-zero pixel, constant-read-out null 1/4. Non-trivial
        # divisions at every pixel (1//2, 3//3, 2//1, 3//1).
        base_a = np.array([[1, 3], [2, 3]], dtype=np.int64)
        base_b = np.array([[2, 3], [1, 1]], dtype=np.int64)
        return (np.tile(base_a, (side // 2, side // 2)),
                np.tile(base_b, (side // 2, side // 2)))
    if dataset in ("fourvalue_p2", "fourvalue_p3", "fourvalue_p4"):
        # Campaign 6 cyclic-calibration targets (paper/HW_CAMPAIGN_6_PROTOCOL.md):
        # together with `fourvalue` (R = [[0,1],[2,3]]) every pixel takes each
        # q=2 quotient value exactly once across the four targets.
        pats = {"fourvalue_p2": ([[2, 2], [3, 1]], [[2, 1], [1, 2]]),   # R = [[1,2],[3,0]]
                "fourvalue_p3": ([[2, 3], [1, 3]], [[1, 1], [3, 2]]),   # R = [[2,3],[0,1]]
                "fourvalue_p4": ([[3, 1], [3, 2]], [[1, 3], [2, 1]])}   # R = [[3,0],[1,2]]
        base_a, base_b = (np.array(x, dtype=np.int64) for x in pats[dataset])
        return (np.tile(base_a, (side // 2, side // 2)),
                np.tile(base_b, (side // 2, side // 2)))
    if dataset == "random4":
        # Campaign 7: non-periodic content, seeded. I_a in [0,3], I_b in [1,3]
        # (no divide-by-zero pixel, every pixel a genuine quotient match).
        rng = np.random.default_rng(1)
        return (rng.integers(0, 2**q, (side, side)).astype(np.int64),
                rng.integers(1, 2**q, (side, side)).astype(np.int64))
    if dataset == "blocks":
        # Campaign 18: block-constant content on 2x2 blocks, with the quotient
        # values balanced over the frame (one quarter of the blocks per value) and
        # the operand pair of each block drawn among those realising its value.
        # Balanced values keep the flat-field decoder's assumption and put the
        # constant-read-out null at 1/4; the block structure keeps the load's angle
        # spectrum sparse, which is what brings 256 pixels into the gate budget.
        rng = np.random.default_rng(18)
        nb = (side // 2) ** 2
        by_value: dict[int, list[tuple[int, int]]] = {}
        for a_val in range(2**q):
            for b_val in range(1, 2**q):
                by_value.setdefault(a_val // b_val, []).append((a_val, b_val))
        values = np.repeat(np.arange(2**q), nb // (2**q))
        rng.shuffle(values)
        blk_a = np.zeros(nb, dtype=np.int64); blk_b = np.zeros(nb, dtype=np.int64)
        for k, v in enumerate(values):
            options = by_value[int(v)]
            blk_a[k], blk_b[k] = options[rng.integers(len(options))]
        half = side // 2
        I_a = np.kron(blk_a.reshape(half, half), np.ones((2, 2), dtype=np.int64))
        I_b = np.kron(blk_b.reshape(half, half), np.ones((2, 2), dtype=np.int64))
        return I_a.astype(np.int64), I_b.astype(np.int64)
    if dataset == "gp_balanced":
        # Projection target for the Class-A circuit (calibrated noise model, not run on the
        # device): Laurdan GP at q = 2, q_frac = 2 balanced over the four values -1, -1/4, +1/4,
        # +1 (a quarter of the pixels each, seeded), so that both signs and two magnitudes are
        # present, no divide-by-zero pixel, and the constant-read-out null is 1/4. The Laurdan
        # patch of campaign 20 has every informative pixel on one sign, which a read-out biased
        # toward the relaxed state cannot be told apart from.
        rng = np.random.default_rng(20)
        options = {0: [(0, 1), (0, 2), (0, 3)], 1: [(1, 2)], 2: [(2, 1)], 3: [(1, 0), (2, 0), (3, 0)]}
        npx = side * side
        vals = np.repeat(np.arange(4), npx // 4); rng.shuffle(vals)
        I_a = np.zeros(npx, dtype=np.int64); I_b = np.zeros(npx, dtype=np.int64)
        for k, v in enumerate(vals):
            a_v, b_v = options[int(v)][rng.integers(len(options[int(v)]))]
            I_a[k], I_b[k] = a_v, b_v
        return I_a.reshape(side, side), I_b.reshape(side, side)
    if dataset == "tiled":
        # Campaign 18 control: the four-value 2x2 pattern replicated to the full
        # frame. Four distinct operand pairs at any size, the cheapest possible
        # load; it bounds what image size alone buys.
        base_a = np.array([[1, 3], [2, 3]], dtype=np.int64)
        base_b = np.array([[2, 3], [1, 1]], dtype=np.int64)
        r = side // 2
        return np.tile(base_a, (r, r)), np.tile(base_b, (r, r))
    if dataset == "balanced8":
        # Campaign 12: 8x8 with each quotient value on exactly 16 pixels (seeded
        # permutation), no divide-by-zero pixel; constant null 16/64. Built so
        # that the flat-field decoder's balance assumption holds at 64 pixels.
        rng = np.random.default_rng(12)
        vals = np.repeat(np.arange(4), (side * side) // 4); rng.shuffle(vals)
        R = vals.reshape(side, side)
        pairs = {0: (1, 2), 1: (3, 3), 2: (2, 1), 3: (3, 1)}   # (I_a, I_b) with I_a // I_b = R
        I_a = np.vectorize(lambda v: pairs[v][0])(R).astype(np.int64)
        I_b = np.vectorize(lambda v: pairs[v][1])(R).astype(np.int64)
        return I_a, I_b
    if dataset == "fourvalue_q3":
        # Campaign 6: q = 3, four distinct 3-bit quotients R = [[0,3],[5,7]]
        # from I_a=[[2,7],[5,7]], I_b=[[3,2],[1,1]]; null 1/4, chance 1/8.
        base_a = np.array([[2, 7], [5, 7]], dtype=np.int64)
        base_b = np.array([[3, 2], [1, 1]], dtype=np.int64)
        return (np.tile(base_a, (side // 2, side // 2)),
                np.tile(base_b, (side // 2, side // 2)))
    if dataset == "fura2":
        f340, f380, _ = synthesise_fura2(side=side, seed=1)
        return (quantise_to_q(f340, q).astype(np.int64),
                quantise_to_q(f380, q).astype(np.int64))
    if dataset == "rogfp2":
        f405, f488, _ = synthesise_rogfp(side=side, seed=2)
        return (quantise_to_q(f405, q).astype(np.int64),
                quantise_to_q(f488, q).astype(np.int64))
    raise ValueError(f"unknown dataset {dataset}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="run_hardware_class_b_nonrestoring",
                                 description=__doc__)
    ap.add_argument("--dataset", type=str, default="canonical",
                    choices=["canonical", "canonical_shared", "canonical_nd",
                             "synthetic", "fura2", "rogfp2", "fourvalue",
                             "fourvalue_p2", "fourvalue_p3", "fourvalue_p4", "fourvalue_q3", "blocks", "tiled", "gp_balanced", "random4", "balanced8"])
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--q", type=int, default=2)
    ap.add_argument("--divider", type=str, default="nonrestoring",
                    choices=["restoring", "nonrestoring", "lookup"])
    ap.add_argument("--load", type=str, default="mcx", choices=["mcx", "ucry"],
                    help="NEQR load: per-pixel multi-controlled X (mcx) or one uniformly "
                         "controlled Ry per bit-plane (ucry, 2^(2n) CNOTs per plane)")
    ap.add_argument("--label", type=str, default=None)
    ap.add_argument("--backend", type=str, default=None)
    ap.add_argument("--shots", type=int, default=None,
                    help="If omitted: max(4096, 1024*n_pixels).")
    ap.add_argument("--mitigation", type=str, default="trex+dd",
                    choices=["trex+dd", "trex", "dd", "none"])
    ap.add_argument("--dd-sequence", type=str, default="XY4",
                    choices=["XY4", "XpXm", "XX"],
                    help="Decoupling sequence when mitigation includes DD; these "
                         "are the three SamplerV2 exposes. XpXm and XX are the "
                         "matched-pulse-count pair that separates a coherent "
                         "pulse-error mechanism from an incoherent one: both "
                         "insert two pulses per window with identical timing, but "
                         "XpXm's +/- alternation cancels pulse-amplitude error to "
                         "first order while XX accumulates it (paper section 7.4).")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--repeat", type=int, default=1,
                    help="Submit the same circuit N times and report the match-rate "
                         "distribution plus the per-pixel recovery frequency. Every "
                         "hardware number in the paper is otherwise a single job on "
                         "4 pixels.")
    args = ap.parse_args(argv)
    if args.repeat < 1:
        ap.error("--repeat must be >= 1")

    n, q = args.n, args.q
    side = 1 << n
    n_pixels = side * side
    label = args.label or f"clb_{args.dataset}_n{n}q{q}_nonrestoring"
    shots = args.shots if args.shots is not None else max(4096, 1024 * n_pixels)

    outdir = args.outdir / utc_stamp()
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Outdir: {outdir}")

    service = ibm.get_service()
    backend = ibm.pick_backend(service, min_qubits=24, name=args.backend)
    print(f"Backend: {backend.name} ({backend.num_qubits} qubits, "
          f"pending {backend.status().pending_jobs})")

    I_a, I_b = load_dataset(args.dataset, n, q)
    R_classical = np.where(I_b > 0, I_a // np.maximum(I_b, 1), 0)
    divzero_classical = (I_b == 0)

    qc, layout = class_b_ratio(I_a, I_b, q=q, divider=args.divider, load=args.load)
    print(f"\n--- {label} ---")
    print(f"  dataset={args.dataset} n={n} q={q} pixels={n_pixels} shots={shots}")
    print(f"  logical qubits = {qc.num_qubits}, logical size = {qc.size()}")
    print(f"  divider = {args.divider}, mitigation = {args.mitigation}"
          + (f", dd = {args.dd_sequence}" if "dd" in args.mitigation else ""))
    runs = []
    for rep in range(1, args.repeat + 1):
        run_label = label if args.repeat == 1 else f"{label}_r{rep}"
        if args.repeat > 1:
            print(f"\n  [run {rep}/{args.repeat}] {run_label}")
        t0 = time.time()
        counts, transpiled, job_id, summary = ibm.hw_run(
            qc, backend=backend, shots=shots, mitigation=args.mitigation,
            dd_sequence=args.dd_sequence,
        )
        elapsed = time.time() - t0
        print(f"  job_id = {job_id}")
        print(f"  transpiled depth = {summary['depth']}, "
              f"two-q gates = {summary['two_q_gate_count']}, "
              f"qubits = {summary['num_qubits']}")
        print(f"  wallclock = {elapsed:.1f} s")
        print(f"  unique bitstrings = {len(counts)}")

        quotient, divzero = decode_class_b_ratio(
            counts, n=n, q=q, layout=layout, total_qubits=qc.num_qubits,
        )
        per_pixel = np.zeros((side, side), dtype=bool)
        for rr in range(side):
            for cc in range(side):
                if divzero_classical[rr, cc]:
                    per_pixel[rr, cc] = bool(divzero[rr, cc])
                else:
                    per_pixel[rr, cc] = (not divzero[rr, cc]) and \
                        int(quotient[rr, cc]) == int(R_classical[rr, cc])
        match_count = int(per_pixel.sum())
        pct = 100.0 * match_count / n_pixels
        print(f"  quantum quotient = {quotient.tolist()}")
        print(f"  classical R      = {R_classical.tolist()}")
        print(f"  pixelwise match  = {match_count} / {n_pixels} ({pct:.1f}%)")

        meta = {
            "label": run_label,
            "repeat_index": rep,
            "repeat_total": args.repeat,
            "dataset": args.dataset,
            "n": n,
            "q": q,
            "n_pixels": n_pixels,
            "backend": backend.name,
            "shots": shots,
            "mitigation": args.mitigation,
            "dd_sequence": args.dd_sequence,
            "job_id": job_id,
            "depth": summary["depth"],
            "two_q_gate_count": summary["two_q_gate_count"],
            "num_qubits": summary["num_qubits"],
            "logical_qubits": qc.num_qubits,
            "logical_size": qc.size(),
            "wallclock_seconds": round(elapsed, 1),
            "n_unique_bitstrings": len(counts),
            "divider": args.divider,
            "load": args.load,
            "quantum_quotient": quotient.tolist(),
            "quantum_divzero": divzero.tolist(),
            "classical_R": R_classical.tolist(),
            "classical_divzero": divzero_classical.tolist(),
            "per_pixel_match": per_pixel.tolist(),
            "match_count": match_count,
            "match_pct": round(pct, 1),
        }
        ibm.persist_run(outdir, label=run_label, pass_name="hw",
                        circuit=qc, transpiled=transpiled,
                        counts=counts, metadata=meta)
        runs.append(meta)

    payload = {r["label"]: r for r in runs}
    if args.repeat > 1:
        # A single job on 4 pixels is an anecdote; the distribution over
        # repeats is the measurement. Per-pixel frequency additionally shows
        # *which* quotient values survive, which is the §7.2 mechanism claim.
        counts_arr = np.array([r["match_count"] for r in runs], dtype=float)
        freq = np.mean([np.array(r["per_pixel_match"], dtype=float) for r in runs], axis=0)
        agg = {
            "label": label,
            "repeat_total": args.repeat,
            "backend": backend.name,
            "divider": args.divider,
            "dataset": args.dataset,
            "match_counts": [int(c) for c in counts_arr],
            "match_mean": round(float(counts_arr.mean()), 3),
            "match_std": round(float(counts_arr.std(ddof=1)) if args.repeat > 1 else 0.0, 3),
            "match_min": int(counts_arr.min()),
            "match_max": int(counts_arr.max()),
            "per_pixel_recovery_frequency": freq.round(3).tolist(),
            "classical_R": R_classical.tolist(),
            "classical_divzero": divzero_classical.tolist(),
            "job_ids": [r["job_id"] for r in runs],
        }
        payload["_aggregate"] = agg
        print(f"\n  === {label}: {args.repeat} runs on {backend.name} ===")
        print(f"  match counts     = {agg['match_counts']} / {n_pixels}")
        print(f"  mean +/- sd      = {agg['match_mean']:.2f} +/- {agg['match_std']:.2f}")
        print(f"  per-pixel recovery frequency = {agg['per_pixel_recovery_frequency']}")

    summary_file = outdir / "summary.json"
    summary_file.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSummary written to {summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
