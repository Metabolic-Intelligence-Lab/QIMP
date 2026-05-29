"""
Counts-side Zero-Noise Extrapolation (ZNE) for the autonomous Class B
pipeline on IBM Heron r2.

Motivation (§6.5.9): with the non-restoring divider the n=1,q=2 pipeline
sits right at the hardware survival ceiling. Low-order targets (Laurdan,
R in {0,1}) survive; the high quotient bit (Fura-2, R=2) is lost. ZNE is
the next lever (§7.3 step c): run the circuit at increasing noise scales
by global unitary folding U -> U (U^dag U)^k, then extrapolate each
per-pixel candidate-value probability back to zero noise.

Folding is done at the LOGICAL level with barriers between copies so the
opt-1 transpile cannot cancel U^dag U; the actual transpiled two-qubit
gate count per fold is recorded and used as the extrapolation abscissa
(more rigorous than the nominal 1x/3x/5x).

Decode: for each pixel and each candidate quotient value v, fit
p_v(N_2q) with a linear (or Richardson) model and evaluate at N_2q = 0;
argmax_v p_v(0) is the ZNE-decoded quotient.

Modes:
  --verify-sim   : run folds on noiseless MPS; every fold must be bit-exact
                   (sanity check that folding preserves the unitary).
  (default)      : submit folds to hardware and ZNE-extrapolate.

Usage:
  python scripts/run_hardware_zne.py --dataset fura2 --n 1 --q 2 --verify-sim
  python scripts/run_hardware_zne.py --dataset fura2 --n 1 --q 2 --folds 1 3 5 --backend ibm_marrakesh
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile

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


def load_dataset(dataset: str, n: int, q: int):
    side = 1 << n
    if dataset == "canonical":
        # Match run_hardware_class_b_nonrestoring.py exactly: prefer the
        # persisted npz patch (the one J7 used) so ZNE is a like-for-like
        # control rather than a different, divzero-heavy patch.
        npz = REPO / "paper" / "data_autonomous" / f"canonical_{side}x{side}.npz"
        if q == 2 and npz.exists():
            d = np.load(npz)
            return d["I_a"].astype(np.int64), d["I_b"].astype(np.int64)
        I_a, I_b, _, _ = prepare_canonical(target_n=n)
        return I_a.astype(np.int64), I_b.astype(np.int64)
    if dataset == "synthetic":
        ba = np.array([[3, 2], [3, 1]], dtype=np.int64)
        bb = np.array([[1, 1], [3, 1]], dtype=np.int64)
        return (np.tile(ba, (side // 2, side // 2)),
                np.tile(bb, (side // 2, side // 2)))
    if dataset == "fura2":
        f340, f380, _ = synthesise_fura2(side=side, seed=1)
        return (quantise_to_q(f340, q).astype(np.int64),
                quantise_to_q(f380, q).astype(np.int64))
    if dataset == "rogfp2":
        f405, f488, _ = synthesise_rogfp(side=side, seed=2)
        return (quantise_to_q(f405, q).astype(np.int64),
                quantise_to_q(f488, q).astype(np.int64))
    raise ValueError(dataset)


def build_folded(qc: QuantumCircuit, n_fold: int) -> QuantumCircuit:
    """Global unitary folding to odd noise scale ``n_fold`` (1, 3, 5, ...).

    Returns U (U^dag U)^k where k = (n_fold - 1) / 2, with a barrier
    between every copy so a downstream transpile cannot cancel U^dag U.
    The input circuit must be unmeasured.
    """
    if n_fold % 2 == 0 or n_fold < 1:
        raise ValueError(f"n_fold must be a positive odd integer, got {n_fold}")
    folded = qc.copy()
    inv = qc.inverse()
    k = (n_fold - 1) // 2
    for _ in range(k):
        folded.barrier()
        folded.compose(inv, inplace=True)
        folded.barrier()
        folded.compose(qc, inplace=True)
    return folded


def candidate_distributions(counts: dict, n: int, q: int, layout: dict,
                            total_qubits: int) -> dict:
    """Per-pixel probability distribution over quotient values from counts.

    Returns {pixel(row,col): {quotient_value: probability}}.
    """
    side = 1 << n
    quo_idx = layout["quotient"]
    pos_idx = layout["position"]
    shots_total = sum(counts.values())
    per_pixel: dict = {}
    pixel_shots: dict = {}
    for bits, c in counts.items():
        flat = bits.replace(" ", "")
        pos = sum(int(flat[total_qubits - 1 - pos_idx[i]]) << i
                  for i in range(2 * n))
        quo = sum(int(flat[total_qubits - 1 - quo_idx[i]]) << i
                  for i in range(q))
        row, col = pos // side, pos % side
        d = per_pixel.setdefault((row, col), {})
        d[quo] = d.get(quo, 0) + c
        pixel_shots[(row, col)] = pixel_shots.get((row, col), 0) + c
    # normalise to probabilities
    out = {}
    for px, dd in per_pixel.items():
        tot = pixel_shots[px]
        out[px] = {v: cnt / tot for v, cnt in dd.items()}
    return out


def zne_extrapolate(cx_scales: list[float], dists: list[dict],
                    n: int, q: int) -> np.ndarray:
    """Linear least-squares extrapolation of each pixel's candidate-value
    probability to zero gate count; argmax gives the decoded quotient."""
    side = 1 << n
    out = np.zeros((side, side), dtype=np.int64)
    xs = np.array(cx_scales, dtype=float)
    for row in range(side):
        for col in range(side):
            best_v, best_p0 = 0, -1.0
            for v in range(1 << q):
                ys = np.array([d.get((row, col), {}).get(v, 0.0)
                               for d in dists], dtype=float)
                # linear fit p(x) = a*x + b; extrapolate to x=0 -> b
                if len(xs) >= 2:
                    a, b = np.polyfit(xs, ys, 1)
                    p0 = b
                else:
                    p0 = ys[0]
                if p0 > best_p0:
                    best_p0, best_v = p0, v
            out[row, col] = best_v
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="run_hardware_zne", description=__doc__)
    ap.add_argument("--dataset", default="fura2",
                    choices=["canonical", "synthetic", "fura2", "rogfp2"])
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--q", type=int, default=2)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--backend", default=None)
    ap.add_argument("--shots", type=int, default=8192)
    ap.add_argument("--mitigation", default="trex+dd")
    ap.add_argument("--verify-sim", action="store_true")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args(argv)

    n, q = args.n, args.q
    side = 1 << n
    I_a, I_b = load_dataset(args.dataset, n, q)
    R_classical = np.where(I_b > 0, I_a // np.maximum(I_b, 1), 0)
    divzero = (I_b == 0)
    qc, layout = class_b_ratio(I_a, I_b, q=q, divider="nonrestoring")
    total_qubits = qc.num_qubits
    print(f"dataset={args.dataset} n={n} q={q}  logical qubits={total_qubits}")
    print(f"classical R = {R_classical.tolist()}")

    if args.verify_sim:
        from qiskit import ClassicalRegister
        from qiskit_aer import AerSimulator
        sim = AerSimulator(method="matrix_product_state")
        for nf in args.folds:
            folded = build_folded(qc, nf)
            creg = ClassicalRegister(total_qubits, "meas")
            folded.add_register(creg)
            folded.measure(range(total_qubits), range(total_qubits))
            t = transpile(folded, sim, basis_gates=["id", "u", "cx"],
                          optimization_level=1)
            cx = t.count_ops().get("cx", 0)
            counts = sim.run(t, shots=4096 * side * side).result().get_counts()
            quo, dz = decode_class_b_ratio(counts, n=n, q=q, layout=layout,
                                           total_qubits=total_qubits)
            Rcv = np.where(divzero, -1, R_classical)
            Rqv = np.where(dz, -1, quo)
            match = int((Rcv == Rqv).sum())
            print(f"  fold {nf}x: {cx} basisCX, match {match}/{side*side} "
                  f"({'OK' if match == side*side else 'MISMATCH'})")
        return 0

    # Hardware path
    outdir = args.outdir / utc_stamp()
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Outdir: {outdir}")
    service = ibm.get_service()
    backend = ibm.pick_backend(service, min_qubits=total_qubits, name=args.backend)
    print(f"Backend: {backend.name} (pending {backend.status().pending_jobs})")

    results = []
    dists = []
    cx_scales = []
    for nf in args.folds:
        folded = build_folded(qc, nf)
        label = f"zne_{args.dataset}_n{n}q{q}_fold{nf}"
        print(f"\n--- {label} ---")
        t0 = time.time()
        counts, transpiled, job_id, summary = ibm.hw_run(
            folded, backend=backend, shots=args.shots,
            mitigation=args.mitigation, optimization_level=1,
        )
        elapsed = time.time() - t0
        cx = summary["two_q_gate_count"]
        print(f"  job_id={job_id}  CX={cx}  depth={summary['depth']}  "
              f"wall={elapsed:.0f}s")
        dist = candidate_distributions(counts, n, q, layout, total_qubits)
        dists.append(dist)
        cx_scales.append(float(cx))
        meta = {
            "label": label, "dataset": args.dataset, "n": n, "q": q,
            "fold": nf, "backend": backend.name, "shots": args.shots,
            "job_id": job_id, "two_q_gate_count": cx,
            "depth": summary["depth"], "wallclock_seconds": round(elapsed, 1),
        }
        ibm.persist_run(outdir, label=label, pass_name="hw",
                        circuit=folded, transpiled=transpiled,
                        counts=counts, metadata=meta)
        results.append(meta)

    # ZNE extrapolation. Report genuine quotient matches on the
    # non-divzero pixels ONLY (ZNE here extrapolates the quotient register;
    # divzero pixels are excluded from the score rather than counted free).
    R_zne = zne_extrapolate(cx_scales, dists, n, q)
    n_quotient_pixels = int((~divzero).sum())
    match_quotient = 0
    for r in range(side):
        for c in range(side):
            if not divzero[r, c] and R_zne[r, c] == R_classical[r, c]:
                match_quotient += 1
    print(f"\n=== ZNE extrapolated R = {R_zne.tolist()} ===")
    print(f"classical R = {R_classical.tolist()}")
    print(f"divzero     = {divzero.tolist()}")
    print(f"ZNE quotient match (non-divzero pixels only) = "
          f"{match_quotient}/{n_quotient_pixels}")

    summary_file = outdir / "zne_summary.json"
    summary_file.write_text(json.dumps({
        "dataset": args.dataset, "n": n, "q": q, "folds": args.folds,
        "cx_scales": cx_scales, "R_classical": R_classical.tolist(),
        "divzero": divzero.tolist(),
        "R_zne": R_zne.tolist(),
        "match_quotient": match_quotient,
        "n_quotient_pixels": n_quotient_pixels,
        "jobs": results,
    }, indent=2, default=str))
    print(f"Summary written to {summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
