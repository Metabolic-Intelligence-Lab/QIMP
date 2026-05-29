"""
Map the NEQR-only primitive's gate-budget ceiling on IBM Heron r2.

The Class B pipeline already showed (J1, J3, J4) that the autonomous
divider sits on the noise floor at present-generation gate error. The
encoder primitive (J2 at n=1) survived 146 transpiled CX gates with
4/4 per-pixel match — but where does it stop surviving?

This script submits two follow-up jobs:

  J5: dual_neqr_load at n=2, q=2 (4x4 image, ~600-800 CX expected)
  J6: dual_neqr_load at n=3, q=2 (8x8 image, ~2k CX expected)

Both use Laurdan canonical data so the result is reproducible. Persists
under data/output/ibm_hw/<timestamp>/runs/<label>_hw/.

Usage:
  python scripts/run_hardware_neqr_scaling.py --job j5
  python scripts/run_hardware_neqr_scaling.py --job j6 --backend ibm_kingston
  python scripts/run_hardware_neqr_scaling.py --job all
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from qimp.processing.ratiometric_circuit import dual_neqr_load  # noqa: E402
from qimp.runtime import ibm  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

DEFAULT_OUTDIR = REPO / "data" / "output" / "ibm_hw"


def build_neqr_circuit(image_a: np.ndarray, image_b: np.ndarray,
                       n: int, q: int) -> tuple[QuantumCircuit, dict]:
    """Build a dual_neqr_load circuit at arbitrary n, q on the given images.

    Layout: position(2n) | I_a(q) | I_b(q). Total = 2n + 2q logical qubits.
    """
    pos = QuantumRegister(2 * n, "pos")
    ia = QuantumRegister(q, "ia")
    ib = QuantumRegister(q, "ib")
    qc = QuantumCircuit(pos, ia, ib)
    pos_idx = list(range(2 * n))
    ia_idx = list(range(2 * n, 2 * n + q))
    ib_idx = list(range(2 * n + q, 2 * n + 2 * q))
    dual_neqr_load(qc, image_a, image_b, q,
                   position_qubits=pos_idx,
                   intensity_a_qubits=ia_idx,
                   intensity_b_qubits=ib_idx)
    return qc, {"pos": pos_idx, "ia": ia_idx, "ib": ib_idx,
                "n_qubits": qc.num_qubits, "n": n, "q": q}


def load_canonical(n: int) -> tuple[np.ndarray, np.ndarray]:
    side = 1 << n
    npz_path = REPO / "paper" / "data_autonomous" / f"canonical_{side}x{side}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"missing canonical_{side}x{side}.npz; "
            f"run scripts/run_autonomous_n3_laurdan.py to generate it")
    d = np.load(npz_path)
    return d["I_a"].astype(np.int64), d["I_b"].astype(np.int64)


def decode_neqr(counts: dict, layout: dict, image_a: np.ndarray,
                image_b: np.ndarray) -> dict:
    """Per-pixel majority vote over (I_a, I_b) and pixel match count."""
    pos_idx = layout["pos"]
    ia_idx = layout["ia"]
    ib_idx = layout["ib"]
    total_q = layout["n_qubits"]
    n = layout["n"]
    n_pix = 1 << (2 * n)
    side = 1 << n

    per_pixel: dict[int, dict] = {}
    for bits, count in counts.items():
        flat = bits.replace(" ", "")
        pos = sum(int(flat[total_q - 1 - pos_idx[i]]) << i
                  for i in range(2 * n))
        ia = sum(int(flat[total_q - 1 - q_idx]) << i
                 for i, q_idx in enumerate(ia_idx))
        ib = sum(int(flat[total_q - 1 - q_idx]) << i
                 for i, q_idx in enumerate(ib_idx))
        d = per_pixel.setdefault(pos, {})
        d[(ia, ib)] = d.get((ia, ib), 0) + count

    match_a = 0
    match_b = 0
    decoded_ia = np.zeros((side, side), dtype=np.int64)
    decoded_ib = np.zeros((side, side), dtype=np.int64)
    truth_present_for = 0
    for pos in range(n_pix):
        row = pos // side
        col = pos % side
        votes = per_pixel.get(pos, {})
        if not votes:
            continue
        truth_present_for += 1
        best_ia, best_ib = max(votes, key=votes.__getitem__)
        decoded_ia[row, col] = best_ia
        decoded_ib[row, col] = best_ib
        if best_ia == int(image_a[row, col]):
            match_a += 1
        if best_ib == int(image_b[row, col]):
            match_b += 1
    return {
        "match_i_a": match_a,
        "match_i_b": match_b,
        "n_pixels": n_pix,
        "pixels_with_any_shot": truth_present_for,
        "decoded_i_a": decoded_ia.tolist(),
        "decoded_i_b": decoded_ib.tolist(),
        "classical_i_a": image_a.astype(int).tolist(),
        "classical_i_b": image_b.astype(int).tolist(),
    }


def utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def submit_one(qc: QuantumCircuit, *, label: str, backend, outdir: Path,
               shots: int, mitigation: str) -> dict:
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
    ap = argparse.ArgumentParser(prog="run_hardware_neqr_scaling",
                                 description=__doc__)
    ap.add_argument("--job", choices=["j5", "j6", "all"], default="j5")
    ap.add_argument("--backend", type=str, default=None)
    ap.add_argument("--shots", type=int, default=None,
                    help="If omitted: 16 shots/pixel by default.")
    ap.add_argument("--mitigation", type=str, default="trex+dd",
                    choices=["trex+dd", "trex", "dd", "none"])
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--min-qubits", type=int, default=12,
                    help="pick_backend min_num_qubits constraint.")
    args = ap.parse_args(argv)

    outdir = args.outdir / utc_stamp()
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Outdir: {outdir}")

    service = ibm.get_service()
    backend = ibm.pick_backend(service, min_qubits=args.min_qubits,
                               name=args.backend)
    print(f"Backend: {backend.name} ({backend.num_qubits} qubits, "
          f"pending {backend.status().pending_jobs})")

    jobs_to_run = ["j5", "j6"] if args.job == "all" else [args.job]
    results: dict[str, dict] = {}

    job_specs = {
        "j5": ("j5_neqr_n2", 2),
        "j6": ("j6_neqr_n3", 3),
    }
    for jkey in jobs_to_run:
        label, n = job_specs[jkey]
        I_a, I_b = load_canonical(n)
        qc, layout = build_neqr_circuit(I_a, I_b, n=n, q=2)
        side = 1 << n
        shots = args.shots if args.shots is not None else 16 * side * side
        r = submit_one(qc, label=label, backend=backend, outdir=outdir,
                       shots=shots, mitigation=args.mitigation)
        decoded = decode_neqr(r["counts"], layout, I_a, I_b)
        n_pix = decoded["n_pixels"]
        print(f"  per-pixel I_a match (majority): "
              f"{decoded['match_i_a']} / {n_pix}")
        print(f"  per-pixel I_b match (majority): "
              f"{decoded['match_i_b']} / {n_pix}")
        results[jkey] = {**r["meta"], "n": n, **decoded}

    summary_file = outdir / "summary.json"
    summary_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSummary written to {summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
