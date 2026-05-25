#!/usr/bin/env python
"""Robustness experiments for the hardware GP validation (paper revision).

Four experiments designed for the IBM Quantum Open plan budget:

- A : Multi-frame sweep — gp@n=2 on 10 microscopy frames, single backend,
      TREX+DD, 4096 shots. Establishes PSNR distribution across the
      dataset (not just the single representative frame of Section 6.2).

- B : Multi-backend sweep — gp@n=2 on one canonical frame, run on the
      three Heron r3 backends currently visible to the Open plan
      (ibm_marrakesh, ibm_kingston, ibm_fez). Shows the result is not
      tied to a single device's calibration of the day.

- C : Mitigation ablation — gp@n=2 on one canonical frame, run with each
      of {"none", "trex", "dd", "trex+dd"} mitigation modes. Decomposes
      the contribution of TREX vs DD to the recovered signal.

- D : Repeated sessions — gp@n=2 on one canonical frame, single backend,
      TREX+DD, repeated 5 times. Yields an error bar on the PSNR.

Outputs land under data/output/ibm/robustness_<UTC-timestamp>/, with
- summary.csv : one row per (experiment, frame, backend, mitigation, repeat)
- runs/<key>/{circuit.qpy, transpiled.qpy, counts.json, metadata.json}
- figures/    : per-experiment summary plots

Hardware jobs are submitted SEQUENTIALLY to respect the Open plan's
3-active-job cap. Each row is persisted as soon as it completes so an
interruption mid-sweep leaves recoverable job IDs on disk.
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

# ----------------------------- experiment design ---------------------------

# Frames to use for experiment A — 10 frames stratified across the two
# membrane samples present in the lab dataset.
FRAMES_A = [
    "membraneStack_Sample011_L_UV_DC_001rbc3DM2.tif",
    "membraneStack_Sample011_L_UV_DC_001rbc4DM2.tif",
    "membraneStack_Sample011_L_UV_DC_001rbc5DM2.tif",
    "membraneStack_Sample011_L_UV_DC_001rbc6DM2.tif",
    "membraneStack_Sample016_L_UV_DC_003rbc10DM2.tif",
    "membraneStack_Sample016_L_UV_DC_003rbc11DM2.tif",
    "membraneStack_Sample016_L_UV_DC_003rbc12DM2.tif",
    "membraneStack_Sample016_L_UV_DC_003rbc13DM2.tif",
    "membraneStack_Sample016_L_UV_DC_003rbc14DM2.tif",
    "membraneStack_Sample016_L_UV_DC_003rbc15DM2.tif",
]

# Canonical frame shared by experiments B/C/D
CANONICAL_FRAME = "membraneStack_Sample011_L_UV_DC_001rbc3DM2.tif"

# Heron r3 backends visible to the Open plan (verified 2026-05-25)
BACKENDS_B = ["ibm_marrakesh", "ibm_kingston", "ibm_fez"]

MITIGATION_MODES_C = ["none", "trex", "dd", "trex+dd"]

REPEATS_D = 5


# ----------------------------- helpers -------------------------------------


def _utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def _max_intensity_for_metric(ref: np.ndarray) -> float:
    if ref.min() < 0:
        return 2.0  # GP range [-1, 1] peak-to-peak
    return float(max(ref.max(), 1.0))


def _load_image(repo_root: Path, frame_name: str) -> np.ndarray:
    img_path = repo_root / "data" / "immagini" / "trainQML" / frame_name
    return np.asarray(PilImage.open(img_path))


def _build_gp_recipe(image: np.ndarray, n: int = 2, alpha: float = 0.5):
    """Return only the GP recipe for the given (image, n)."""
    recipes = build_recipes(image, n=n, alpha=alpha)
    return next(r for r in recipes if r.encoder == "gp")


def _run_one(
    *,
    key: str,
    experiment: str,
    rec,
    backend,
    shots: int,
    mitigation: str,
    extra_metadata: dict,
    outdir: Path,
) -> dict:
    """Submit one circuit, decode, persist, return summary row."""
    print(f"[{key}] submitting on {backend.name} (mitigation={mitigation})...",
          flush=True)
    try:
        counts, transpiled, job_id, tsummary = ibm.hw_run(
            rec.qc, backend=backend, shots=shots, mitigation=mitigation,
        )
    except Exception as exc:
        print(f"[{key}] FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        return {
            "key": key, "experiment": experiment, "status": "failed",
            "error": str(exc), **extra_metadata,
        }

    decoded = rec.decoder(counts)
    ref = rec.reference.astype(np.float64)
    dec_f = np.asarray(decoded, dtype=np.float64)
    max_i = _max_intensity_for_metric(ref)

    row = {
        "key": key,
        "experiment": experiment,
        "status": "completed",
        "backend": backend.name,
        "mitigation": mitigation,
        "shots": shots,
        "job_id": job_id,
        "mse": float(_mse(ref, dec_f)),
        "psnr": float(_psnr(ref, dec_f, max_intensity=max_i)),
        "depth_transpiled": tsummary["depth"],
        "two_q_gate_count": tsummary["two_q_gate_count"],
        "num_qubits": tsummary["num_qubits"],
        **extra_metadata,
    }

    ibm.persist_run(
        outdir,
        label=key,
        pass_name="hw",
        circuit=rec.qc,
        transpiled=transpiled,
        counts=counts,
        metadata={**row},
    )
    print(f"[{key}] done. PSNR={row['psnr']:.2f} dB  job_id={job_id}")
    return row


def _write_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _append_summary(path: Path, row: dict, all_rows: list[dict]) -> None:
    """Re-write the summary after each row so the file is current on disk."""
    all_rows.append(row)
    _write_summary(path, all_rows)


# ----------------------------- experiments ---------------------------------


def run_experiment_a(service, repo_root: Path, outdir: Path,
                     summary_path: Path, all_rows: list[dict],
                     backend_name: str, shots: int) -> None:
    """A — Multi-frame sweep."""
    print("\n=== Experiment A: multi-frame (gp@n=2, 10 frames) ===\n")
    backend = ibm.pick_backend(service, min_qubits=6, name=backend_name)
    for i, frame_name in enumerate(FRAMES_A, start=1):
        img = _load_image(repo_root, frame_name)
        rec = _build_gp_recipe(img, n=2)
        key = f"A_frame{i:02d}_gp_n2"
        if ibm.is_run_complete(outdir, key, "hw"):
            print(f"[{key}] already on disk, skipping")
            continue
        row = _run_one(
            key=key, experiment="A_multi_frame", rec=rec, backend=backend,
            shots=shots, mitigation="trex+dd",
            extra_metadata={"frame": frame_name, "n": 2, "alpha": 0.5},
            outdir=outdir,
        )
        _append_summary(summary_path, row, all_rows)


def run_experiment_b(service, repo_root: Path, outdir: Path,
                     summary_path: Path, all_rows: list[dict],
                     shots: int) -> None:
    """B — Multi-backend on canonical frame.

    Skip ibm_marrakesh (already covered in the original session) when the
    user opts to save 1 job; we keep it here for self-contained reproducibility.
    """
    print("\n=== Experiment B: multi-backend (gp@n=2 on 3 Heron r3) ===\n")
    img = _load_image(repo_root, CANONICAL_FRAME)
    rec = _build_gp_recipe(img, n=2)
    for backend_name in BACKENDS_B:
        try:
            backend = ibm.pick_backend(service, min_qubits=6, name=backend_name)
        except Exception as exc:
            print(f"[B/{backend_name}] backend unavailable: {exc}", file=sys.stderr)
            continue
        key = f"B_{backend_name}_gp_n2"
        if ibm.is_run_complete(outdir, key, "hw"):
            print(f"[{key}] already on disk, skipping")
            continue
        row = _run_one(
            key=key, experiment="B_multi_backend", rec=rec, backend=backend,
            shots=shots, mitigation="trex+dd",
            extra_metadata={"frame": CANONICAL_FRAME, "n": 2, "alpha": 0.5},
            outdir=outdir,
        )
        _append_summary(summary_path, row, all_rows)


def run_experiment_c(service, repo_root: Path, outdir: Path,
                     summary_path: Path, all_rows: list[dict],
                     backend_name: str, shots: int) -> None:
    """C — Mitigation ablation on canonical frame."""
    print("\n=== Experiment C: mitigation ablation (gp@n=2, 4 modes) ===\n")
    backend = ibm.pick_backend(service, min_qubits=6, name=backend_name)
    img = _load_image(repo_root, CANONICAL_FRAME)
    rec = _build_gp_recipe(img, n=2)
    for mode in MITIGATION_MODES_C:
        key = f"C_{mode.replace('+','_')}_gp_n2"
        if ibm.is_run_complete(outdir, key, "hw"):
            print(f"[{key}] already on disk, skipping")
            continue
        row = _run_one(
            key=key, experiment="C_mitigation_ablation", rec=rec, backend=backend,
            shots=shots, mitigation=mode,
            extra_metadata={"frame": CANONICAL_FRAME, "n": 2, "alpha": 0.5},
            outdir=outdir,
        )
        _append_summary(summary_path, row, all_rows)


def run_experiment_d(service, repo_root: Path, outdir: Path,
                     summary_path: Path, all_rows: list[dict],
                     backend_name: str, shots: int, repeats: int) -> None:
    """D — Repeated sessions on canonical frame."""
    print(f"\n=== Experiment D: repeated sessions (gp@n=2 x {repeats}) ===\n")
    backend = ibm.pick_backend(service, min_qubits=6, name=backend_name)
    img = _load_image(repo_root, CANONICAL_FRAME)
    rec = _build_gp_recipe(img, n=2)
    for i in range(1, repeats + 1):
        key = f"D_repeat{i:02d}_gp_n2"
        if ibm.is_run_complete(outdir, key, "hw"):
            print(f"[{key}] already on disk, skipping")
            continue
        row = _run_one(
            key=key, experiment="D_repeated_sessions", rec=rec, backend=backend,
            shots=shots, mitigation="trex+dd",
            extra_metadata={
                "frame": CANONICAL_FRAME, "n": 2, "alpha": 0.5, "repeat": i,
            },
            outdir=outdir,
        )
        _append_summary(summary_path, row, all_rows)


# ----------------------------- driver --------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qimp-robustness", description=__doc__)
    p.add_argument(
        "--experiments", nargs="+", default=["A", "B", "C", "D"],
        choices=["A", "B", "C", "D"],
        help="Which experiments to run (default: all four)",
    )
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument(
        "--backend", type=str, default="ibm_marrakesh",
        help="Canonical backend for experiments A/C/D (default: ibm_marrakesh)",
    )
    p.add_argument("--repeats", type=int, default=REPEATS_D,
                   help="Number of repeated sessions in experiment D")
    p.add_argument(
        "--outdir", type=Path, default=Path("data/output/ibm"),
        help="Parent output directory (timestamped subdir created)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]

    stamp = _utc_stamp()
    outdir = args.outdir / f"robustness_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.csv"
    all_rows: list[dict] = []

    print(f"Output: {outdir}")
    print(f"Experiments: {args.experiments}")

    service = ibm.get_service()

    if "A" in args.experiments:
        run_experiment_a(service, repo_root, outdir, summary_path, all_rows,
                         backend_name=args.backend, shots=args.shots)
    if "B" in args.experiments:
        run_experiment_b(service, repo_root, outdir, summary_path, all_rows,
                         shots=args.shots)
    if "C" in args.experiments:
        run_experiment_c(service, repo_root, outdir, summary_path, all_rows,
                         backend_name=args.backend, shots=args.shots)
    if "D" in args.experiments:
        run_experiment_d(service, repo_root, outdir, summary_path, all_rows,
                         backend_name=args.backend, shots=args.shots,
                         repeats=args.repeats)

    _write_summary(summary_path, all_rows)
    completed = sum(1 for r in all_rows if r.get("status") == "completed")
    failed = sum(1 for r in all_rows if r.get("status") == "failed")
    print(f"\nDone. {completed} completed, {failed} failed. "
          f"Summary: {summary_path}")
    print(json.dumps({
        "experiments_run": args.experiments,
        "completed": completed,
        "failed": failed,
        "outdir": str(outdir),
    }, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
