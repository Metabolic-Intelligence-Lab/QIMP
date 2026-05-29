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
    if dataset == "canonical_nd":
        # Non-degenerate Laurdan 2x2 patch (I_a != I_b, R spans {0,1,2});
        # offset (94,60) of the canonical frame, q=2. See §6.5.9.
        d = np.load(REPO / "paper" / "data_autonomous" / "canonical_2x2_nd.npz")
        return d["I_a"].astype(np.int64), d["I_b"].astype(np.int64)
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
                    choices=["canonical", "canonical_nd", "synthetic", "fura2", "rogfp2"])
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--q", type=int, default=2)
    ap.add_argument("--divider", type=str, default="nonrestoring",
                    choices=["restoring", "nonrestoring"])
    ap.add_argument("--label", type=str, default=None)
    ap.add_argument("--backend", type=str, default=None)
    ap.add_argument("--shots", type=int, default=None,
                    help="If omitted: max(4096, 1024*n_pixels).")
    ap.add_argument("--mitigation", type=str, default="trex+dd",
                    choices=["trex+dd", "trex", "dd", "none"])
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args(argv)

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

    qc, layout = class_b_ratio(I_a, I_b, q=q, divider=args.divider)
    print(f"\n--- {label} ---")
    print(f"  dataset={args.dataset} n={n} q={q} pixels={n_pixels} shots={shots}")
    print(f"  logical qubits = {qc.num_qubits}, logical size = {qc.size()}")
    print(f"  divider = {args.divider}, mitigation = {args.mitigation}")
    t0 = time.time()
    counts, transpiled, job_id, summary = ibm.hw_run(
        qc, backend=backend, shots=shots, mitigation=args.mitigation,
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
    match_count = 0
    for rr in range(side):
        for cc in range(side):
            if divzero_classical[rr, cc]:
                if divzero[rr, cc]:
                    match_count += 1
            else:
                if quotient[rr, cc] == R_classical[rr, cc]:
                    match_count += 1
    pct = 100.0 * match_count / n_pixels
    print(f"  quantum quotient = {quotient.tolist()}")
    print(f"  classical R      = {R_classical.tolist()}")
    print(f"  pixelwise match  = {match_count} / {n_pixels} ({pct:.1f}%)")

    meta = {
        "label": label,
        "dataset": args.dataset,
        "n": n,
        "q": q,
        "n_pixels": n_pixels,
        "backend": backend.name,
        "shots": shots,
        "mitigation": args.mitigation,
        "job_id": job_id,
        "depth": summary["depth"],
        "two_q_gate_count": summary["two_q_gate_count"],
        "num_qubits": summary["num_qubits"],
        "logical_qubits": qc.num_qubits,
        "logical_size": qc.size(),
        "wallclock_seconds": round(elapsed, 1),
        "n_unique_bitstrings": len(counts),
        "divider": args.divider,
        "quantum_quotient": quotient.tolist(),
        "quantum_divzero": divzero.tolist(),
        "classical_R": R_classical.tolist(),
        "classical_divzero": divzero_classical.tolist(),
        "match_count": match_count,
        "match_pct": round(pct, 1),
    }
    ibm.persist_run(outdir, label=label, pass_name="hw",
                    circuit=qc, transpiled=transpiled,
                    counts=counts, metadata=meta)

    summary_file = outdir / "summary.json"
    summary_file.write_text(json.dumps({label: meta}, indent=2, default=str))
    print(f"\nSummary written to {summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
