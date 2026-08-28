"""How much depth is left on the table? (§8.3 steps (a) and (b), measured)

§8.3 lists transpiler-level compression as the next lever after the
non-restoring divider and estimates that much of the ~2.4x hardware/basis
CX ratio is routing overhead, but does not measure what is actually
recoverable. This does, on the real backend Target and without a QPU:

  * **optimisation level** -- the paper runs `optimization_level=3`
    throughout; levels 0-3 bound what the preset pipelines give.
  * **routing seed** -- SABRE is stochastic, so a single transpile is one
    draw from a distribution. Best-of-N seeds is free depth, and the
    spread also says how much of the reported CX count is luck.
  * **fractional gates** -- Heron exposes a native RZZ(theta). Arithmetic
    circuits are Toffoli-dominated and decompose to CZ, so the gain is
    not obvious a priori and has to be measured rather than assumed.

The output is the CX budget each lever buys, against the ~670 CX floor at
which §7.5 finds the high-order quotient bit still lost.

Usage:
    python scripts/compression_study.py
    python scripts/compression_study.py --seeds 30 --json out.json
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

from qiskit import transpile  # noqa: E402

from qimp.processing.ratiometric_circuit import class_b_ratio  # noqa: E402
from qimp.runtime import ibm  # noqa: E402
from qimp.testing import _ensure_measured  # noqa: E402

CONFIGS = [
    ("canonical_shared", 1, 2),
    ("fura2", 1, 2),
    ("canonical", 2, 2),
    ("canonical", 1, 3),
]


def two_q_count(circ) -> int:
    return sum(n for name, n in circ.count_ops().items()
               if name in ("cz", "cx", "ecr", "rzz"))


def build(dataset: str, n: int, q: int):
    from run_hardware_class_b_nonrestoring import load_dataset

    I_a, I_b = load_dataset(dataset, n, q)
    qc, _ = class_b_ratio(I_a, I_b, q=q, divider="nonrestoring")
    return _ensure_measured(qc)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="compression_study", description=__doc__)
    ap.add_argument("--backend", type=str, default="ibm_marrakesh")
    ap.add_argument("--seeds", type=int, default=20,
                    help="SABRE routing seeds to sample at optimisation level 3.")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    service = ibm.get_service()
    plain = ibm.pick_backend(service, min_qubits=24, name=args.backend)

    frac = None
    try:
        frac = service.backend(args.backend, use_fractional_gates=True)
        has_rzz = "rzz" in frac.target
        print(f"Backend: {plain.name}  (fractional gates available: {has_rzz})")
        if not has_rzz:
            frac = None
    except Exception as exc:
        print(f"Backend: {plain.name}  (fractional gates unavailable: {exc})")

    rows = []
    for dataset, n, q in CONFIGS:
        measured = build(dataset, n, q)
        cfg = f"{dataset} n={n} q={q}"
        print(f"\n=== {cfg} ===")
        # Basis translation *without* a coupling map: every Toffoli is
        # decomposed but nothing is routed. The gap to the coupled
        # transpile is the SWAP overhead, which is the quantity §8.3(a)
        # proposes to attack -- the raw logical count is not, since its
        # Toffolis have not been decomposed yet.
        basis_only = transpile(measured, basis_gates=["id", "u", "cx"],
                               optimization_level=3, seed_transpiler=1)
        rec: dict = {"dataset": dataset, "n": n, "q": q,
                     "basis_only_two_q": two_q_count(basis_only)}
        print(f"  basis only (no routing) CX = {rec['basis_only_two_q']:5d}")

        # (1) preset optimisation levels, default seed
        levels = {}
        for lvl in (0, 1, 2, 3):
            t = transpile(measured, target=plain.target, optimization_level=lvl,
                          seed_transpiler=1)
            levels[lvl] = two_q_count(t)
            print(f"  opt_level={lvl:d}   CX = {levels[lvl]:5d}")
        rec["levels"] = levels

        # (2) SABRE seed sweep at level 3 -- free depth, and the spread says
        #     how much of a single reported CX count is routing luck.
        counts = []
        for seed in range(args.seeds):
            t = transpile(measured, target=plain.target, optimization_level=3,
                          seed_transpiler=seed)
            counts.append(two_q_count(t))
        counts_arr = np.array(counts)
        rec["seed_sweep"] = {
            "n_seeds": args.seeds,
            "min": int(counts_arr.min()),
            "median": float(np.median(counts_arr)),
            "max": int(counts_arr.max()),
            "mean": float(counts_arr.mean()),
            "std": float(counts_arr.std(ddof=1)),
        }
        best = counts_arr.min()
        med = float(np.median(counts_arr))
        print(f"  seed sweep ({args.seeds} seeds): min {best}, median {med:.0f}, "
              f"max {counts_arr.max()}  -> best-of-N saves "
              f"{100 * (1 - best / med):.1f}% vs the median draw")

        # (3) fractional gates, same seed sweep
        if frac is not None:
            fcounts = []
            for seed in range(args.seeds):
                t = transpile(measured, target=frac.target, optimization_level=3,
                              seed_transpiler=seed)
                fcounts.append(two_q_count(t))
            farr = np.array(fcounts)
            rec["fractional"] = {
                "min": int(farr.min()), "median": float(np.median(farr)),
                "max": int(farr.max()),
            }
            print(f"  fractional gates:            min {farr.min()}, "
                  f"median {np.median(farr):.0f}  -> "
                  f"{100 * (1 - farr.min() / best):+.1f}% vs best plain")
        rows.append(rec)

    print("\n" + "=" * 78)
    print(f"{'config':22s} {'basis':>7s} {'paper (opt3)':>13s} {'routing':>8s} "
          f"{'best':>7s} {'gain':>7s}")
    print("-" * 78)
    for r in rows:
        cfg = f"{r['dataset']} n={r['n']} q={r['q']}"
        base = r["levels"][3]
        best = min(r["seed_sweep"]["min"], min(r["levels"].values()))
        fbest = r.get("fractional", {}).get("min")
        overall = min(best, fbest) if fbest else best
        bo = r["basis_only_two_q"]
        print(f"{cfg:22s} {bo:7d} {base:13d} {base / bo:7.2f}x "
              f"{overall:7d} {100 * (1 - overall / base):+6.1f}%")
    print("\n  'basis' is the Toffoli-decomposed circuit with no coupling map;\n"
          "  'routing' is how much the heavy-hex connectivity adds on top of it,\n"
          "  and 'gain' is what the best transpiler configuration recovers of\n"
          "  that overhead. The §7.5 floor for the high-order quotient bit\n"
          "  sits at ~670 CX.")

    if args.json:
        args.json.write_text(json.dumps({"backend": plain.name, "rows": rows}, indent=2))
        print(f"\nWritten to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
