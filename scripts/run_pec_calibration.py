#!/usr/bin/env python
"""Experiment N — per-pixel response calibration (PEC-style) for gp@n=2.

Tests whether the hardware-decoded image carries a measurable systematic
response to the target u that can be inverted to recover information
beyond the noise floor.

Pipeline:
  1. Use the canonical microscopy frame's (Ia, Ib) channels as the FRQI
     encoding. The encoder is identical across all submissions; only the
     parametric block changes between calibration frames.
  2. Submit 5 CALIBRATION frames with uniform targets u_k = -1, -0.5,
     0, 0.5, +1 (16 pixels at each value), all via
     analytical_gp_params(target=u_k * ones).
  3. Submit 1 TEST frame with target = classical GP image of the
     canonical (Ia, Ib) — same as the §6.2 reference image.
  4. For each pixel p, fit a linear model dec_k[p] = a_p · u_k + b_p
     on the 5 calibration points. Threshold |a_p| > 0.1 to avoid
     ill-conditioned inversion.
  5. Calibrated estimate of the test target:
        u_est[p] = (dec_test[p] - b_p) / a_p  if |a_p| > threshold
                 = dec_test[p] otherwise.
  6. Compare PSNR(u_est, u_true) to PSNR(dec_test_raw, u_true).

If the algorithm is at total noise floor (a_p ≈ 0 everywhere),
calibration cannot help — and we expect PSNR(u_est) ≤ PSNR(raw).
If there is residual per-pixel response, calibration may recover
signal — PSNR(u_est) > PSNR(raw).

6 HW submissions (5 calibration + 1 test), ~6 s QPU.
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

from qimp.encoding.frqi import frqi_circuit
from qimp.metrics import mse as _mse
from qimp.metrics import psnr as _psnr
from qimp.processing.gp_ratio import (
    analytical_gp_params,
    apply_gp_function,
    classical_gp_image,
    decode_gp_counts,
)
from qimp.runtime import ibm
from qimp.runtime.circuits import _downsample_to_n

CANONICAL_FRAME = "membraneStack_Sample011_L_UV_DC_001rbc3DM2.tif"
CALIBRATION_TARGETS = (-1.0, -0.5, 0.0, 0.5, 1.0)
N = 2  # spatial qubits
SLOPE_THRESHOLD = 0.1  # min |a_p| to apply inverse calibration


def _utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def _max_intensity_for_metric(ref: np.ndarray) -> float:
    if ref.min() < 0:
        return 2.0
    return float(max(ref.max(), 1.0))


def _build_circuit_for_target(g_chan, r_chan, target, norm: float, n: int):
    """Construct the GP corrected circuit with a given per-pixel target."""
    rg_stack = np.stack([r_chan, g_chan], axis=0).astype(np.float64)
    qc = frqi_circuit(rg_stack, normalization=norm)
    params = analytical_gp_params(g_chan, r_chan, target=target, normalization=norm)
    apply_gp_function(qc, n=n, m=1, params=list(params))
    return qc


def _submit_and_record(*, key, qc, target_known, backend, shots, outdir, extra):
    """Submit one frame, decode, return (decoded, row)."""
    print(f"[{key}] submitting...", flush=True)
    try:
        counts, transpiled, job_id, summary = ibm.hw_run(
            qc, backend=backend, shots=shots, mitigation="trex+dd",
        )
    except Exception as exc:
        print(f"[{key}] FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        return None, {"key": key, "status": "failed", "error": str(exc), **extra}
    decoded = decode_gp_counts(counts, n=N, m=1)
    ref = target_known.astype(np.float64)
    dec_f = np.asarray(decoded, dtype=np.float64)
    max_i = _max_intensity_for_metric(ref)
    row = {
        "key": key, "status": "completed",
        "backend": backend.name, "shots": shots, "job_id": job_id,
        "psnr_raw": float(_psnr(ref, dec_f, max_intensity=max_i)),
        "mse_raw": float(_mse(ref, dec_f)),
        "target_mean": float(ref.mean()),
        "target_std": float(ref.std()),
        "depth_transpiled": summary["depth"],
        **extra,
    }
    ibm.persist_run(outdir, label=key, pass_name="hw",
                    circuit=qc, transpiled=transpiled,
                    counts=counts, metadata={**row})
    print(f"  -> raw PSNR={row['psnr_raw']:.2f} dB  job_id={job_id}")
    return dec_f, row


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="qimp-pec", description=__doc__)
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--backend", type=str, default="ibm_marrakesh")
    p.add_argument("--outdir", type=Path, default=Path("data/output/ibm"))
    args = p.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    stamp = _utc_stamp()
    outdir = args.outdir / f"pec_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    img = np.asarray(PilImage.open(
        repo_root / "data" / "immagini" / "trainQML" / CANONICAL_FRAME
    ))
    rgb = _downsample_to_n(img, n=N)
    g_chan = rgb[..., 1].astype(np.float64)
    r_chan = rgb[..., 0].astype(np.float64)
    rg_stack = np.stack([r_chan, g_chan], axis=0)
    norm = float(rg_stack.max()) or 1.0
    print(f"Canonical channels: g range=[{g_chan.min()},{g_chan.max()}], r range=[{r_chan.min()},{r_chan.max()}], norm={norm}")

    side = 1 << N
    service = ibm.get_service()
    backend = ibm.pick_backend(service, min_qubits=2 * N + 2, name=args.backend)

    # --- 5 calibration frames ---
    decodes_cal: dict[float, np.ndarray] = {}
    for uk in CALIBRATION_TARGETS:
        target_uniform = np.full((side, side), uk, dtype=np.float64)
        # Avoid pure-uniform-1.0 edge case where arcsin(-1) might saturate
        # the parametric block; analytical_gp_params handles clipping internally.
        key = f"N_cal_u{uk:+.1f}"
        if ibm.is_run_complete(outdir, key, "hw"):
            print(f"[{key}] already on disk, skipping")
            continue
        qc = _build_circuit_for_target(g_chan, r_chan, target_uniform, norm, N)
        dec, row = _submit_and_record(
            key=key, qc=qc, target_known=target_uniform,
            backend=backend, shots=args.shots, outdir=outdir,
            extra={"experiment": "N_pec_calibration", "phase": "calibration",
                   "u_uniform": uk},
        )
        rows.append(row)
        if dec is not None:
            decodes_cal[uk] = dec

    # --- 1 test frame: real classical GP target ---
    test_target = classical_gp_image(g_chan, r_chan, alpha=0.5)
    print(f"\nTest target: range=[{test_target.min():.3f}, {test_target.max():.3f}], "
          f"mean={test_target.mean():.3f}, std={test_target.std():.3f}")
    qc_test = _build_circuit_for_target(g_chan, r_chan, test_target, norm, N)
    key_test = "N_test_classical_gp"
    if not ibm.is_run_complete(outdir, key_test, "hw"):
        dec_test, row_test = _submit_and_record(
            key=key_test, qc=qc_test, target_known=test_target,
            backend=backend, shots=args.shots, outdir=outdir,
            extra={"experiment": "N_pec_calibration", "phase": "test",
                   "frame": CANONICAL_FRAME},
        )
        rows.append(row_test)
    else:
        dec_test = None

    # --- Per-pixel calibration ---
    if len(decodes_cal) == len(CALIBRATION_TARGETS) and dec_test is not None:
        print("\n--- Per-pixel calibration ---")
        u_arr = np.asarray(CALIBRATION_TARGETS, dtype=np.float64)
        # Build the (5, 16) stack of calibration decodeds
        cal_stack = np.stack([decodes_cal[uk] for uk in CALIBRATION_TARGETS], axis=0)
        cal_flat = cal_stack.reshape(len(CALIBRATION_TARGETS), -1)  # (5, side²)
        # Linear fit per pixel: cal_flat[:, p] = a_p · u_arr + b_p
        # numpy.polyfit returns (deg+1, n_pixels) coeffs high-to-low
        coefs = np.polyfit(u_arr, cal_flat, deg=1)
        a_flat, b_flat = coefs[0], coefs[1]  # (side²,) each
        a = a_flat.reshape(side, side)
        b = b_flat.reshape(side, side)
        print(f"Slope a: range [{a.min():.3f}, {a.max():.3f}], mean {a.mean():.3f}, std {a.std():.3f}")
        print(f"Intercept b: range [{b.min():.3f}, {b.max():.3f}]")
        print(f"# pixels with |a| > {SLOPE_THRESHOLD}: {int((np.abs(a) > SLOPE_THRESHOLD).sum())} / {side*side}")

        # Goodness-of-fit (R²) per pixel
        fitted_cal = a_flat * u_arr[:, None] + b_flat
        ss_res = ((cal_flat - fitted_cal) ** 2).sum(axis=0)
        ss_tot = ((cal_flat - cal_flat.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        print(f"R²: range [{r2.min():.3f}, {r2.max():.3f}], mean {r2.mean():.3f}")

        # Apply inverse calibration to dec_test
        mask = np.abs(a) > SLOPE_THRESHOLD
        u_est = np.where(mask, (dec_test - b) / a, dec_test)
        u_est_clipped = np.clip(u_est, -1.0, 1.0)
        ref = test_target.astype(np.float64)
        psnr_calibrated = float(_psnr(ref, u_est_clipped, max_intensity=2.0))
        psnr_raw_test = float(_psnr(ref, dec_test, max_intensity=2.0))
        print(f"\nPSNR(raw test)       = {psnr_raw_test:.2f} dB")
        print(f"PSNR(calibrated)     = {psnr_calibrated:.2f} dB   "
              f"(delta = {psnr_calibrated - psnr_raw_test:+.2f} dB)")
        rows.append({
            "key": "N_calibrated_result", "status": "completed",
            "phase": "calibration_result",
            "n_pixels_calibrated": int(mask.sum()),
            "n_pixels_total": int(side * side),
            "slope_mean": float(a.mean()), "slope_std": float(a.std()),
            "intercept_mean": float(b.mean()),
            "r2_mean": float(r2.mean()),
            "psnr_raw": psnr_raw_test,
            "psnr_calibrated": psnr_calibrated,
            "psnr_gain": psnr_calibrated - psnr_raw_test,
        })
        np.savez(outdir / "calibration.npz",
                 calibration_decodes=cal_stack, calibration_targets=u_arr,
                 a=a, b=b, r2=r2.reshape(side, side),
                 dec_test=dec_test, u_est=u_est, u_est_clipped=u_est_clipped,
                 test_target=test_target)

    # Write summary
    if rows:
        fields = sorted({k for r in rows for k in r})
        with open(outdir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    print(json.dumps({"outdir": str(outdir), "n_rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
