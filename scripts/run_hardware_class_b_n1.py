"""Submit the autonomous Class-B integer-ratio pipeline to IBM Quantum
Heron r2 hardware in four progressively-larger jobs.

Smallest first:
  J1 (q_div_restoring standalone, ~14 qubits): smoke-test the divider
     primitive against a known classical input on hardware.
  J2 (dual_neqr_load standalone, ~8 qubits): smoke-test the NEQR
     encoder in isolation.
  J3 (full class_b_ratio on synthetic 2×2, 24 qubits): headline result.
  J4 (full class_b_ratio on Laurdan 2×2, 24 qubits): same on real data.

All jobs go through `qimp.runtime.ibm.hw_run` with TREX + XY4 DD by
default, and persist their results under
``data/output/ibm_hw/<UTC-timestamp>/runs/<label>_hw/``.

Usage:
  python scripts/run_hardware_class_b_n1.py --job j1
  python scripts/run_hardware_class_b_n1.py --job all
  python scripts/run_hardware_class_b_n1.py --job j1 --backend ibm_marrakesh
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qimp.processing.arithmetic import q_div_restoring  # noqa: E402
from qimp.processing.ratiometric_circuit import (  # noqa: E402
    class_b_ratio,
    decode_class_b_ratio,
    dual_neqr_load,
)
from qimp.runtime import ibm  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = REPO / "data" / "output" / "ibm_hw"


# ---------------------------------------------------------------------------
# Circuit builders
# ---------------------------------------------------------------------------


def build_j1_divider_only(dividend: int = 3, divisor: int = 1, q: int = 2):
    """Build a circuit that runs q_div_restoring on basis-state-initialised
    dividend / divisor (no NEQR superposition).

    Layout (~14 qubits at q=2):
      dividend (q) | divisor (q) | quotient (q) | work (q) | pad (1) | c (12) | flag (1)
    """
    needed_c = (q + 1) * (q + 2)
    div_q = QuantumRegister(q, "div")
    ds_q = QuantumRegister(q, "ds")
    quo_q = QuantumRegister(q, "quo")
    work_q = QuantumRegister(q, "work")
    pad_q = QuantumRegister(1, "pad")
    c_q = QuantumRegister(needed_c, "c")
    flag_q = QuantumRegister(1, "flag")
    qc = QuantumCircuit(div_q, ds_q, quo_q, work_q, pad_q, c_q, flag_q)

    div_idx = list(range(q))
    ds_idx = list(range(q, 2 * q))
    quo_idx = list(range(2 * q, 3 * q))
    work_idx = list(range(3 * q, 4 * q))
    pad_idx = 4 * q
    c_idx = list(range(4 * q + 1, 4 * q + 1 + needed_c))
    flag_idx = 4 * q + 1 + needed_c

    # X-init the dividend and divisor to the requested integer values.
    for i in range(q):
        if (dividend >> i) & 1:
            qc.x(div_idx[i])
        if (divisor >> i) & 1:
            qc.x(ds_idx[i])

    q_div_restoring(qc, div_idx, ds_idx, quo_idx, work_idx,
                    pad_idx, c_idx, flag_idx)
    return qc, {"quo": quo_idx, "div": div_idx, "flag": flag_idx,
                "n_qubits": qc.num_qubits}


def build_j2_neqr_only(image_a: np.ndarray, image_b: np.ndarray, q: int = 2):
    """Build a circuit that does dual_neqr_load only — no division."""
    pos = QuantumRegister(2, "pos")
    ia = QuantumRegister(q, "ia")
    ib = QuantumRegister(q, "ib")
    qc = QuantumCircuit(pos, ia, ib)
    pos_idx = list(range(2))
    ia_idx = list(range(2, 2 + q))
    ib_idx = list(range(2 + q, 2 + 2 * q))
    dual_neqr_load(qc, image_a, image_b, q,
                   position_qubits=pos_idx,
                   intensity_a_qubits=ia_idx,
                   intensity_b_qubits=ib_idx)
    return qc, {"pos": pos_idx, "ia": ia_idx, "ib": ib_idx,
                "n_qubits": qc.num_qubits}


def build_j3_or_j4_full(image_a: np.ndarray, image_b: np.ndarray, q: int = 2):
    """Full class_b_ratio at n=1, q=2 (24 qubits)."""
    qc, layout = class_b_ratio(image_a, image_b, q=q)
    layout["n_qubits"] = qc.num_qubits
    return qc, layout


# ---------------------------------------------------------------------------
# Dataset references
# ---------------------------------------------------------------------------


def synthetic_2x2() -> tuple[np.ndarray, np.ndarray]:
    return (np.array([[3, 2], [3, 1]], dtype=np.int64),
            np.array([[1, 1], [3, 1]], dtype=np.int64))


def laurdan_2x2() -> tuple[np.ndarray, np.ndarray]:
    d = np.load(REPO / "paper" / "data_autonomous" / "canonical_2x2.npz")
    return d["I_a"].astype(np.int64), d["I_b"].astype(np.int64)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def submit_one(qc: QuantumCircuit, *, label: str, backend, outdir: Path,
               shots: int, mitigation: str) -> dict:
    """Submit one circuit via ibm.hw_run, persist, return a small dict."""
    print(f"\n--- {label} ---")
    print(f"  logical qubits = {qc.num_qubits}, logical size = {qc.size()}")
    t0 = time.time()
    counts, transpiled, job_id, summary = ibm.hw_run(
        qc, backend=backend, shots=shots, mitigation=mitigation,
    )
    elapsed = time.time() - t0
    print(f"  job_id = {job_id}")
    print(f"  transpiled depth = {summary['depth']}, "
          f"two-q gates = {summary['two_q_gate_count']}, "
          f"qubits = {summary['num_qubits']}")
    print(f"  wallclock = {elapsed:.1f} s")
    print(f"  unique bitstrings = {len(counts)}")

    meta = {
        "label": label,
        "backend": backend.name,
        "shots": shots,
        "mitigation": mitigation,
        "job_id": job_id,
        "depth": summary["depth"],
        "two_q_gate_count": summary["two_q_gate_count"],
        "num_qubits": summary["num_qubits"],
        "logical_qubits": qc.num_qubits,
        "logical_size": qc.size(),
        "wallclock_seconds": round(elapsed, 1),
        "n_unique_bitstrings": len(counts),
    }
    ibm.persist_run(outdir, label=label, pass_name="hw",
                    circuit=qc, transpiled=transpiled,
                    counts=counts, metadata=meta)
    return {"counts": counts, "meta": meta, "transpiled": transpiled}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="run_hardware_class_b_n1", description=__doc__)
    ap.add_argument("--job", choices=["j1", "j2", "j3", "j4", "all"], default="j1")
    ap.add_argument("--backend", type=str, default=None,
                    help="If omitted, picks least-busy ≥ 24-qubit backend.")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--mitigation", type=str, default="trex+dd",
                    choices=["trex+dd", "trex", "dd", "none"])
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args(argv)

    outdir = args.outdir / utc_stamp()
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Outdir: {outdir}")

    service = ibm.get_service()
    backend = ibm.pick_backend(service, min_qubits=24, name=args.backend)
    print(f"Backend: {backend.name} ({backend.num_qubits} qubits, "
          f"pending {backend.status().pending_jobs})")

    jobs_to_run: list[str] = ["j1", "j2", "j3", "j4"] if args.job == "all" else [args.job]
    results = {}

    if "j1" in jobs_to_run:
        qc, layout = build_j1_divider_only(dividend=3, divisor=1, q=2)
        r = submit_one(qc, label="j1_div_only", backend=backend, outdir=outdir,
                       shots=args.shots, mitigation=args.mitigation)
        # Decode: read just the quotient register from the counts.
        # At n=0 there's only one branch — majority vote on quotient bits.
        quo_idx = layout["quo"]
        flag_idx = layout["flag"]
        total_q = layout["n_qubits"]
        # qiskit's count strings are MSB-first across all qubits in the
        # circuit; after transpile measure() places classical bits in
        # register order. Since hw_run uses _ensure_measured which adds
        # measure_all in big-endian convention, qubit i is at position
        # total_q - 1 - i from the LEFT of the bitstring.
        votes: dict[int, int] = {}
        flag_votes: dict[int, int] = {}
        for bits, n in r["counts"].items():
            flat = bits.replace(" ", "")
            quo = sum(int(flat[total_q - 1 - q_idx]) << i
                      for i, q_idx in enumerate(quo_idx))
            flag = int(flat[total_q - 1 - flag_idx])
            votes[quo] = votes.get(quo, 0) + n
            flag_votes[flag] = flag_votes.get(flag, 0) + n
        majority_q = max(votes, key=votes.__getitem__)
        majority_flag = max(flag_votes, key=flag_votes.__getitem__)
        # Classical reference: 3 // 1 = 3, divzero flag should be 0.
        print(f"  decoded quotient distribution (top 5): "
              f"{sorted(votes.items(), key=lambda x: -x[1])[:5]}")
        print(f"  majority quotient = {majority_q}  (classical truth: 3)")
        print(f"  majority flag     = {majority_flag} (classical truth: 0)")
        results["j1"] = {
            **r["meta"],
            "decoded_majority_quotient": majority_q,
            "classical_truth": 3,
            "match_majority": majority_q == 3,
        }

    if "j2" in jobs_to_run:
        I_a, I_b = synthetic_2x2()
        qc, layout = build_j2_neqr_only(I_a, I_b, q=2)
        r = submit_one(qc, label="j2_neqr_only", backend=backend, outdir=outdir,
                       shots=args.shots, mitigation=args.mitigation)
        # Decode: for each (row, col) position branch, read the most-common
        # (i_a, i_b) pair and compare against the input.
        pos_idx = layout["pos"]
        ia_idx = layout["ia"]
        ib_idx = layout["ib"]
        total_q = layout["n_qubits"]
        per_pixel: dict[tuple[int, int], dict] = {}
        for bits, n in r["counts"].items():
            flat = bits.replace(" ", "")
            col = sum(int(flat[total_q - 1 - pos_idx[i]]) << i for i in range(1))
            row = sum(int(flat[total_q - 1 - pos_idx[1 + i]]) << i for i in range(1))
            ia = sum(int(flat[total_q - 1 - q_idx]) << i
                     for i, q_idx in enumerate(ia_idx))
            ib = sum(int(flat[total_q - 1 - q_idx]) << i
                     for i, q_idx in enumerate(ib_idx))
            d = per_pixel.setdefault((row, col), {})
            d[(ia, ib)] = d.get((ia, ib), 0) + n
        match_a = 0
        match_b = 0
        for (r2, c2), votes in per_pixel.items():
            best_ia, best_ib = max(votes, key=votes.__getitem__)
            if best_ia == int(I_a[r2, c2]):
                match_a += 1
            if best_ib == int(I_b[r2, c2]):
                match_b += 1
        print(f"  per-pixel I_a match (majority): {match_a} / 4")
        print(f"  per-pixel I_b match (majority): {match_b} / 4")
        results["j2"] = {
            **r["meta"],
            "match_i_a_4": match_a,
            "match_i_b_4": match_b,
        }

    for jkey, label, dataset in (
        ("j3", "j3_class_b_synthetic", synthetic_2x2()),
        ("j4", "j4_class_b_laurdan",  laurdan_2x2()),
    ):
        if jkey not in jobs_to_run:
            continue
        I_a, I_b = dataset
        qc, layout = build_j3_or_j4_full(I_a, I_b, q=2)
        r = submit_one(qc, label=label, backend=backend, outdir=outdir,
                       shots=args.shots, mitigation=args.mitigation)
        quotient, divzero = decode_class_b_ratio(
            r["counts"], n=1, q=2, layout=layout, total_qubits=qc.num_qubits,
        )
        R_classical = np.where(I_b > 0, I_a // np.maximum(I_b, 1), 0)
        divzero_classical = (I_b == 0)
        match_count = 0
        for rr in range(2):
            for cc in range(2):
                if divzero_classical[rr, cc]:
                    if divzero[rr, cc]:
                        match_count += 1
                else:
                    if quotient[rr, cc] == R_classical[rr, cc]:
                        match_count += 1
        print(f"  quantum quotient = {quotient.tolist()}")
        print(f"  quantum divzero  = {divzero.tolist()}")
        print(f"  classical R      = {R_classical.tolist()}")
        print(f"  classical divz   = {divzero_classical.tolist()}")
        print(f"  pixelwise match  = {match_count} / 4")
        results[jkey] = {
            **r["meta"],
            "quantum_quotient": quotient.tolist(),
            "quantum_divzero": divzero.tolist(),
            "classical_R": R_classical.tolist(),
            "classical_divzero": divzero_classical.tolist(),
            "match_count": match_count,
        }

    # Aggregate summary
    summary_file = outdir / "summary.json"
    summary_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSummary written to {summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
