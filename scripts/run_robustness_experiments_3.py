#!/usr/bin/env python
"""Robustness experiments — round 3 (H + I + J) for the paper revision.

Three follow-ups that close the dangling ends of the v3 draft:

- H : Push n=3 — submit gp@n=3 (8 qubits, transpiled depth ~80k+, ~30k
      two-qubit gates) to ibm_marrakesh with TREX+DD. Establishes
      whether the recovered signal at gp@n=2 (~7 dB) is at the noise
      floor of this regime or whether there is still room to fall at
      larger n.

- I : Shot scaling — gp@n=2 on canonical frame with shot counts
      {1024, 16384, 65536} (the 4096-shot point is the existing baseline).
      A genuine recovered signal should scale roughly as +0.5 * log10(S)
      until shot noise stops being the dominant uncertainty; a flat
      curve would indicate gate-noise saturation.

- J : ZNE recovery — gp@n=2 on canonical frame with mitigation="zne"
      (resilience_level=2, zero-noise extrapolation on top of TREX+DD).
      Answers the direct question of Section 7.4: does more aggressive
      mitigation recover any of the 94 dB algorithmic advantage that
      collapsed to sub-sigma on TREX+DD?

Outputs under data/output/ibm/robustness3_<UTC-timestamp>/.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image as PilImage

from qimp.metrics import mse as _mse
from qimp.metrics import psnr as _psnr
from qimp.runtime import ibm
from qimp.runtime.circuits import build_recipes

CANONICAL_FRAME = "membraneStack_Sample011_L_UV_DC_001rbc3DM2.tif"


def _utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def _max_intensity_for_metric(ref: np.ndarray) -> float:
    if ref.min() < 0:
        return 2.0
    return float(max(ref.max(), 1.0))


def _load_image(repo_root: Path, frame_name: str) -> np.ndarray:
    return np.asarray(PilImage.open(
        repo_root / "data" / "immagini" / "trainQML" / frame_name
    ))


def _write_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _append(path: Path, row: dict, all_rows: list[dict]) -> None:
    all_rows.append(row)
    _write_summary(path, all_rows)


def _submit_and_persist(*, key, experiment, qc, decoder, reference,
                       backend, shots, mitigation, extra, outdir):
    print(f"[{key}] submitting (shots={shots}, mitigation={mitigation})...",
          flush=True)
    try:
        counts, transpiled, job_id, tsummary = ibm.hw_run(
            qc, backend=backend, shots=shots, mitigation=mitigation,
        )
    except Exception as exc:
        print(f"[{key}] FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        return {"key": key, "experiment": experiment, "status": "failed",
                "error": str(exc), "shots": shots, "mitigation": mitigation,
                **extra}
    decoded = decoder(counts)
    ref = reference.astype(np.float64)
    dec_f = np.asarray(decoded, dtype=np.float64)
    max_i = _max_intensity_for_metric(ref)
    row = {
        "key": key, "experiment": experiment, "status": "completed",
        "backend": backend.name, "mitigation": mitigation, "shots": shots,
        "job_id": job_id,
        "mse": float(_mse(ref, dec_f)),
        "psnr": float(_psnr(ref, dec_f, max_intensity=max_i)),
        "depth_transpiled": tsummary["depth"],
        "two_q_gate_count": tsummary["two_q_gate_count"],
        "num_qubits": tsummary["num_qubits"],
        **extra,
    }
    ibm.persist_run(outdir, label=key, pass_name="hw",
                    circuit=qc, transpiled=transpiled,
                    counts=counts, metadata={**row})
    print(f"[{key}] done. PSNR={row['psnr']:.2f} dB  "
          f"depth={tsummary['depth']}  2q={tsummary['two_q_gate_count']}  "
          f"job_id={job_id}")
    return row


# ------------------ H: push n=3 ---------------------------------------------


def run_experiment_h(service, repo_root, outdir, summary_path, all_rows,
                     backend_name, shots):
    print("\n=== Experiment H: push n=3 (gp@n=3, 8 qubits) ===\n")
    img = _load_image(repo_root, CANONICAL_FRAME)
    rec = next(r for r in build_recipes(img, n=3, alpha=0.5) if r.encoder == "gp")
    backend = ibm.pick_backend(service, min_qubits=8, name=backend_name)
    key = "H_push_gp_n3"
    if ibm.is_run_complete(outdir, key, "hw"):
        print(f"[{key}] already on disk, skipping")
        return
    row = _submit_and_persist(
        key=key, experiment="H_push_n3",
        qc=rec.qc, decoder=rec.decoder, reference=rec.reference,
        backend=backend, shots=shots, mitigation="trex+dd",
        extra={"frame": CANONICAL_FRAME, "n": 3, "alpha": 0.5},
        outdir=outdir,
    )
    _append(summary_path, row, all_rows)


# ------------------ I: shot scaling -----------------------------------------


def run_experiment_i(service, repo_root, outdir, summary_path, all_rows,
                     backend_name):
    print("\n=== Experiment I: shot scaling (gp@n=2 at 1k/16k/64k) ===\n")
    img = _load_image(repo_root, CANONICAL_FRAME)
    rec = next(r for r in build_recipes(img, n=2, alpha=0.5) if r.encoder == "gp")
    backend = ibm.pick_backend(service, min_qubits=6, name=backend_name)
    for shots in (1024, 16384, 65536):
        key = f"I_shots{shots}_gp_n2"
        if ibm.is_run_complete(outdir, key, "hw"):
            print(f"[{key}] already on disk, skipping")
            continue
        row = _submit_and_persist(
            key=key, experiment="I_shot_scaling",
            qc=rec.qc, decoder=rec.decoder, reference=rec.reference,
            backend=backend, shots=shots, mitigation="trex+dd",
            extra={"frame": CANONICAL_FRAME, "n": 2, "alpha": 0.5},
            outdir=outdir,
        )
        _append(summary_path, row, all_rows)


# ------------------ J: ZNE recovery -----------------------------------------


def run_experiment_j(service, repo_root, outdir, summary_path, all_rows,
                     backend_name, shots):
    print("\n=== Experiment J: ZNE recovery (gp@n=2, resilience_level=2) ===\n")
    img = _load_image(repo_root, CANONICAL_FRAME)
    rec = next(r for r in build_recipes(img, n=2, alpha=0.5) if r.encoder == "gp")
    backend = ibm.pick_backend(service, min_qubits=6, name=backend_name)
    key = "J_zne_gp_n2"
    if ibm.is_run_complete(outdir, key, "hw"):
        print(f"[{key}] already on disk, skipping")
        return
    row = _submit_and_persist(
        key=key, experiment="J_zne_recovery",
        qc=rec.qc, decoder=rec.decoder, reference=rec.reference,
        backend=backend, shots=shots, mitigation="zne",
        extra={"frame": CANONICAL_FRAME, "n": 2, "alpha": 0.5},
        outdir=outdir,
    )
    _append(summary_path, row, all_rows)


# ------------------ driver --------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qimp-robustness-3", description=__doc__)
    p.add_argument("--experiments", nargs="+", default=["H", "I", "J"],
                   choices=["H", "I", "J"])
    p.add_argument("--shots", type=int, default=4096,
                   help="shot count for H and J (I has its own scan)")
    p.add_argument("--backend", type=str, default="ibm_marrakesh")
    p.add_argument("--outdir", type=Path, default=Path("data/output/ibm"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    stamp = _utc_stamp()
    outdir = args.outdir / f"robustness3_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.csv"
    all_rows: list[dict] = []

    print(f"Output: {outdir}")
    print(f"Experiments: {args.experiments}")

    service = ibm.get_service()

    if "H" in args.experiments:
        run_experiment_h(service, repo_root, outdir, summary_path, all_rows,
                         backend_name=args.backend, shots=args.shots)
    if "I" in args.experiments:
        run_experiment_i(service, repo_root, outdir, summary_path, all_rows,
                         backend_name=args.backend)
    if "J" in args.experiments:
        run_experiment_j(service, repo_root, outdir, summary_path, all_rows,
                         backend_name=args.backend, shots=args.shots)

    _write_summary(summary_path, all_rows)
    completed = sum(1 for r in all_rows if r.get("status") == "completed")
    failed = sum(1 for r in all_rows if r.get("status") == "failed")
    print(f"\nDone. {completed} completed, {failed} failed.")
    print(json.dumps({"experiments_run": args.experiments,
                      "completed": completed, "failed": failed,
                      "outdir": str(outdir)}, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
