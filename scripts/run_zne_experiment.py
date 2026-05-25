#!/usr/bin/env python
"""Experiment K — manual counts-side ZNE for gp@n=2.

qiskit-ibm-runtime >= 0.47 SamplerV2 does not expose resilience_level
(ZNE is an EstimatorV2 / expectation-value technique). This script
implements zero-noise extrapolation manually for the counts-side GP
reconstruction:

  1. Transpile gp@n=2 once at optimization_level=3 against ibm_marrakesh.
  2. For each noise scale s in {1, 3, 5}, build a folded circuit by
     composing the transpiled circuit with its inverse k = (s-1)/2 times:
        s=1 -> qc
        s=3 -> qc · qc⁻¹ · qc
        s=5 -> qc · qc⁻¹ · qc · qc⁻¹ · qc
     The folded circuits are NOT re-transpiled — they execute exactly the
     same physical gates so the noise scales as ~s.
  3. Submit each folded circuit via SamplerV2 with TREX + XY4 DD,
     4096 shots, sequential submission.
  4. Decode counts to per-pixel GP image at each noise scale.
  5. For each pixel, fit a polynomial of degree 2 to the three (s,
     decoded(s)) points and read off the s=0 intercept. The
     pixel-wise s=0 extrapolation is the ZNE-mitigated GP image.
  6. Compute PSNR of the ZNE image vs the classical reference and
     compare to the s=1 baseline.

If ZNE recovers a non-trivial fraction of the algorithmic-to-noise-floor
gap, the recovered PSNR will exceed the s=1 baseline. If not, the
extrapolation will be noise-amplified and PSNR may degrade.
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
from qiskit import QuantumCircuit

from qimp.metrics import mse as _mse
from qimp.metrics import psnr as _psnr
from qimp.runtime import ibm
from qimp.runtime.circuits import build_recipes

CANONICAL_FRAME = "membraneStack_Sample011_L_UV_DC_001rbc3DM2.tif"
NOISE_SCALES = (1, 3, 5)


def _utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def _load_image(repo_root: Path, frame_name: str) -> np.ndarray:
    return np.asarray(PilImage.open(
        repo_root / "data" / "immagini" / "trainQML" / frame_name
    ))


def _fold_global(qc: QuantumCircuit, scale_factor: int) -> QuantumCircuit:
    """Global gate folding on the LOGICAL circuit (pre-transpile).

    Replaces qc with qc · (qc⁻¹ · qc)^k where k = (scale_factor - 1) / 2,
    inserting barriers between segments to prevent the transpiler from
    optimising U·U⁻¹·U → U.
    """
    if scale_factor < 1 or scale_factor % 2 == 0:
        raise ValueError(
            f"scale_factor must be a positive odd integer, got {scale_factor}"
        )
    if scale_factor == 1:
        return qc.copy()
    inv = qc.inverse()
    folded = qc.copy()
    k = (scale_factor - 1) // 2
    for _ in range(k):
        folded.barrier()
        folded = folded.compose(inv)
        folded.barrier()
        folded = folded.compose(qc)
    return folded


def _extrapolate_per_pixel(
    decodes_by_scale: dict[int, np.ndarray], scales: tuple[int, ...]
) -> np.ndarray:
    """Per-pixel polynomial fit of degree min(2, len-1), evaluate at s=0."""
    arrays = [decodes_by_scale[s] for s in scales]
    stacked = np.stack(arrays, axis=0)  # (n_scales, H, W)
    s_arr = np.asarray(scales, dtype=np.float64)
    H, W = stacked.shape[1], stacked.shape[2]
    flat = stacked.reshape(len(scales), -1)  # (n_scales, H*W)
    deg = min(2, len(scales) - 1)
    # polyfit accepts (n_scales,) and (n_scales, N) -> returns (deg+1, N)
    coefs = np.polyfit(s_arr, flat, deg=deg)
    # Evaluate at x=0: that's just the constant term, the LAST coefficient
    # (polyfit returns high-to-low).
    extrapolated_flat = coefs[-1]
    return extrapolated_flat.reshape(H, W)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="qimp-zne", description=__doc__)
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--backend", type=str, default="ibm_marrakesh")
    p.add_argument("--outdir", type=Path, default=Path("data/output/ibm"))
    args = p.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    stamp = _utc_stamp()
    outdir = args.outdir / f"zne_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    print(f"Output: {outdir}")
    print(f"Backend: {args.backend}")
    print(f"Noise scales: {NOISE_SCALES}")

    # Build the gp@n=2 recipe (unmeasured)
    img = _load_image(repo_root, CANONICAL_FRAME)
    rec = next(r for r in build_recipes(img, n=2, alpha=0.5) if r.encoder == "gp")
    print(f"Recipe: {rec.label}  qubits={rec.qc.num_qubits}")

    service = ibm.get_service()
    backend = ibm.pick_backend(service, min_qubits=6, name=args.backend)

    # Submit each noise scale by folding the LOGICAL circuit, then letting
    # hw_run do measure + transpile + submit + decode. Barriers in the fold
    # prevent the transpiler from optimising U·U⁻¹·U → U.
    decodes: dict[int, np.ndarray] = {}
    for s in NOISE_SCALES:
        key = f"K_zne_scale{s}_gp_n2"
        if ibm.is_run_complete(outdir, key, "hw"):
            print(f"[{key}] already on disk, skipping submission")
            continue

        folded_logical = _fold_global(rec.qc, s)
        print(f"\n[scale={s}x] logical depth={folded_logical.depth()}, "
              f"2q={sum(1 for instr in folded_logical.data if instr.operation.num_qubits >= 2)}", flush=True)
        try:
            counts, transpiled, job_id, summary = ibm.hw_run(
                folded_logical, backend=backend, shots=args.shots,
                mitigation="trex+dd",
            )
        except Exception as exc:
            print(f"[{key}] FAILED: {exc}", file=sys.stderr)
            traceback.print_exc()
            rows.append({"key": key, "status": "failed", "error": str(exc),
                         "scale_factor": s})
            continue
        decoded = rec.decoder(counts)
        decodes[s] = decoded
        ref = rec.reference.astype(np.float64)
        max_i = 2.0 if ref.min() < 0 else max(float(ref.max()), 1.0)
        row = {
            "key": key, "status": "completed",
            "backend": backend.name, "scale_factor": s, "shots": args.shots,
            "job_id": job_id,
            "mse": float(_mse(ref, decoded)),
            "psnr": float(_psnr(ref, decoded, max_intensity=max_i)),
            "depth_transpiled": summary["depth"],
            "two_q_gate_count": summary["two_q_gate_count"],
            "num_qubits": summary["num_qubits"],
        }
        rows.append(row)
        ibm.persist_run(outdir, label=key, pass_name="hw",
                        circuit=folded_logical, transpiled=transpiled,
                        counts=counts, metadata={**row})
        print(f"  -> PSNR(scale={s}x) = {row['psnr']:.2f} dB  "
              f"job_id={job_id}  transpiled depth={summary['depth']}")

    # Per-pixel extrapolation
    if len(decodes) == len(NOISE_SCALES):
        print("\n--- Per-pixel ZNE extrapolation to s=0 ---")
        extrapolated = _extrapolate_per_pixel(decodes, NOISE_SCALES)
        # Clip to valid GP range
        extrapolated_clipped = np.clip(extrapolated, -1.0, 1.0)
        ref = rec.reference.astype(np.float64)
        max_i = 2.0 if ref.min() < 0 else max(float(ref.max()), 1.0)
        psnr_raw = float(_psnr(ref, extrapolated, max_intensity=max_i))
        psnr_clipped = float(_psnr(ref, extrapolated_clipped, max_intensity=max_i))
        mse_raw = float(_mse(ref, extrapolated))
        mse_clipped = float(_mse(ref, extrapolated_clipped))
        print(f"  extrapolated PSNR (raw)     = {psnr_raw:.2f} dB  (MSE={mse_raw:.4e})")
        print(f"  extrapolated PSNR (clipped) = {psnr_clipped:.2f} dB  (MSE={mse_clipped:.4e})")
        print(f"  baseline    PSNR (scale=1x) = {next(r for r in rows if r.get('scale_factor')==1)['psnr']:.2f} dB")
        rows.append({
            "key": "K_zne_extrapolated", "status": "completed",
            "scale_factor": "extrapolated_to_0",
            "psnr": psnr_clipped, "mse": mse_clipped,
            "psnr_unclipped": psnr_raw, "mse_unclipped": mse_raw,
        })
        np.save(outdir / "decoded_per_scale.npy",
                {s: decodes[s] for s in NOISE_SCALES}, allow_pickle=True)
        np.save(outdir / "extrapolated.npy", extrapolated_clipped)

    # Write summary
    if rows:
        fields = sorted({k for r in rows for k in r})
        with open(outdir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    print(json.dumps({"outdir": str(outdir),
                      "n_scales_submitted": len(decodes),
                      "n_rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
