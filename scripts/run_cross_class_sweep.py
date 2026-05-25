#!/usr/bin/env python
"""Experiment O — extended cross-class HW validation (Lemmas 1, 2).

Round-2 Experiment F validated Classes B (Fura-2 calcium) and C (roGFP2
redox) on hardware at N=1 each. This script sweeps each class over 5
synthetic frames with controlled target dynamic ranges, so the resulting
PSNR distribution can be plotted against target range and the
Lemma-1/Lemma-2 transfer is established statistically rather than from
a single point.

Pipeline per frame:
  1. Synthesise (Ia, Ib) intensity pair at side=2^n with a frame-
     specific parameter (Ca peak for B, OxD slope for C).
  2. Compute the bounded surrogate u (Lemma 1 for B, Lemma 2 for C).
  3. Submit the corrected ansatz with analytical_gp_params(target=u)
     via ibm.hw_run (TREX + XY4 DD, 4096 shots).
  4. Decode and record PSNR vs u.

10 sequential HW submissions on ibm_marrakesh, ~10 s QPU, well within
the Open plan budget.
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

from qimp.encoding.frqi import frqi_circuit
from qimp.metrics import mse as _mse
from qimp.metrics import psnr as _psnr
from qimp.processing.gp_ratio import (
    analytical_gp_params,
    apply_gp_function,
    decode_gp_counts,
)
from qimp.runtime import ibm

# ---------------- Class B (Fura-2 calcium) synthesis ------------------------


def synthesise_fura2(side: int, ca_peak_nM: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fura-2 dual-excitation channels (F340, F380) for the given peak
    free [Ca²⁺] (in nM). Grynkiewicz (1985) calibration.

    The spatial profile is a radial gaussian peak with mild shot noise;
    the peak Ca controls the target's dynamic range.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float64)
    rr = np.sqrt(
        (yy - (side - 1) / 2) ** 2 + (xx - (side - 1) / 2) ** 2
    ) / max(side / 2, 1)
    ca_baseline = 80.0
    ca = ca_baseline + (ca_peak_nM - ca_baseline) * np.exp(-(rr ** 2) * 1.5)
    ca += rng.normal(0.0, 2.0, ca.shape)
    ca = np.clip(ca, 10.0, 5000.0)

    K_d, Sf340, Sb340, Sf380, Sb380 = 224.0, 50.0, 250.0, 250.0, 50.0
    f340 = Sf340 + (Sb340 - Sf340) * ca / (K_d + ca)
    f380 = Sf380 + (Sb380 - Sf380) * ca / (K_d + ca)
    return f340, f380


# ---------------- Class C (roGFP2 redox) synthesis --------------------------


def synthesise_rogfp(side: int, slope: float, axis: int = 0,
                     seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """roGFP2 dual-excitation channels (F405, F488) for a linear OxD
    gradient of given `slope` along `axis` (0 = vertical, 1 = horizontal,
    2 = diagonal). `slope` ∈ [0, 1] controls the OxD dynamic range.
    """
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float64)
    if axis == 0:
        grad = yy / max(side - 1, 1)
    elif axis == 1:
        grad = xx / max(side - 1, 1)
    else:
        grad = ((yy + xx) / max(2 * (side - 1), 1))
    oxd = 0.5 + slope * (grad - 0.5) * 2  # span 0.5 ± slope, then clipped
    oxd = np.clip(oxd, 0.0, 1.0)
    R_red, R_ox = 0.20, 4.50
    R = R_red + (R_ox - R_red) * oxd
    iso = 100.0
    f405 = iso * R / (1.0 + R)
    f488 = iso * 1.0 / (1.0 + R)
    return f405, f488


# ---------------- shared helpers --------------------------------------------


def _utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def _max_intensity_for_metric(ref: np.ndarray) -> float:
    if ref.min() < 0:
        return 2.0
    return float(max(ref.max(), 1.0))


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


def _submit_class_frame(
    *, key, experiment, channel_a, channel_b, target_u,
    n, backend, shots, outdir, extra,
):
    """Build the corrected circuit with analytical_gp_params(target=u),
    submit to HW, decode, append row."""
    rg_stack = np.stack([channel_b, channel_a], axis=0).astype(np.float64)
    norm = float(rg_stack.max()) or 1.0
    qc = frqi_circuit(rg_stack, normalization=norm)
    params = analytical_gp_params(
        channel_a, channel_b, target=target_u, normalization=norm,
    )
    apply_gp_function(qc, n=n, m=1, params=list(params))

    print(f"[{key}] target range=[{target_u.min():.3f}, {target_u.max():.3f}], "
          f"target std={target_u.std():.3f}", flush=True)
    try:
        counts, transpiled, job_id, summary = ibm.hw_run(
            qc, backend=backend, shots=shots, mitigation="trex+dd",
        )
    except Exception as exc:
        print(f"[{key}] FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        return {"key": key, "experiment": experiment, "status": "failed",
                "error": str(exc), **extra}

    decoded = decode_gp_counts(counts, n=n, m=1)
    ref = target_u.astype(np.float64)
    dec_f = np.asarray(decoded, dtype=np.float64)
    max_i = _max_intensity_for_metric(ref)
    row = {
        "key": key, "experiment": experiment, "status": "completed",
        "backend": backend.name, "shots": shots, "job_id": job_id,
        "target_min": float(ref.min()),
        "target_max": float(ref.max()),
        "target_range": float(ref.max() - ref.min()),
        "target_std": float(ref.std()),
        "mse": float(_mse(ref, dec_f)),
        "psnr": float(_psnr(ref, dec_f, max_intensity=max_i)),
        "depth_transpiled": summary["depth"],
        "two_q_gate_count": summary["two_q_gate_count"],
        "num_qubits": summary["num_qubits"],
        "n": n, **extra,
    }
    ibm.persist_run(outdir, label=key, pass_name="hw",
                    circuit=qc, transpiled=transpiled,
                    counts=counts, metadata={**row})
    print(f"  -> PSNR = {row['psnr']:.2f} dB  job_id={job_id}")
    return row


# ---------------- experiments -----------------------------------------------


def run_class_b_sweep(service, outdir, summary_path, all_rows,
                     backend_name, shots, n=2):
    print(f"\n=== Class B (Fura-2) sweep — 5 frames at n={n} ===\n")
    backend = ibm.pick_backend(service, min_qubits=2 * n + 2, name=backend_name)
    side = 1 << n
    # Sweep [Ca²⁺] peak from low (narrow target range) to high (wide range)
    ca_peaks = [100.0, 250.0, 500.0, 1000.0, 2000.0]
    for i, ca_peak in enumerate(ca_peaks, start=1):
        f340, f380 = synthesise_fura2(side, ca_peak_nM=ca_peak, seed=i)
        R = f340 / (f380 + 1e-12)
        u = (R - 1.0) / (R + 1.0)
        key = f"O_classB_fura2_ca{int(ca_peak):04d}_n{n}"
        if ibm.is_run_complete(outdir, key, "hw"):
            print(f"[{key}] already on disk, skipping")
            continue
        row = _submit_class_frame(
            key=key, experiment="O_class_B_fura2",
            channel_a=f340, channel_b=f380, target_u=u,
            n=n, backend=backend, shots=shots, outdir=outdir,
            extra={"class": "B", "ca_peak_nM": ca_peak, "frame_index": i},
        )
        _append(summary_path, row, all_rows)


def run_class_c_sweep(service, outdir, summary_path, all_rows,
                     backend_name, shots, n=2):
    print(f"\n=== Class C (roGFP2) sweep — 5 frames at n={n} ===\n")
    backend = ibm.pick_backend(service, min_qubits=2 * n + 2, name=backend_name)
    side = 1 << n
    # Sweep OxD gradient slope; the 5th frame uses a diagonal gradient
    configs = [
        ("v", 0, 0.20),    # vertical, very mild slope
        ("v", 0, 0.50),
        ("v", 0, 0.80),
        ("h", 1, 0.80),    # horizontal, full slope
        ("d", 2, 0.80),    # diagonal
    ]
    for i, (label, axis, slope) in enumerate(configs, start=1):
        f405, f488 = synthesise_rogfp(side, slope=slope, axis=axis, seed=i)
        R = f405 / (f488 + 1e-12)
        R_red, R_ox = 0.20, 4.50
        R_C = np.clip((R - R_red) / (R_ox - R_red), 0.0, 1.0)
        u = 2.0 * R_C - 1.0
        key = f"O_classC_rogfp_{label}_slope{int(slope*100):02d}_n{n}"
        if ibm.is_run_complete(outdir, key, "hw"):
            print(f"[{key}] already on disk, skipping")
            continue
        row = _submit_class_frame(
            key=key, experiment="O_class_C_rogfp",
            channel_a=f405, channel_b=f488, target_u=u,
            n=n, backend=backend, shots=shots, outdir=outdir,
            extra={"class": "C", "axis": label, "slope": slope,
                   "frame_index": i, "R_red": R_red, "R_ox": R_ox},
        )
        _append(summary_path, row, all_rows)


# ---------------- driver ----------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qimp-cross-class-sweep", description=__doc__)
    p.add_argument("--classes", nargs="+", default=["B", "C"], choices=["B", "C"])
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--backend", type=str, default="ibm_marrakesh")
    p.add_argument("--outdir", type=Path, default=Path("data/output/ibm"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    stamp = _utc_stamp()
    outdir = args.outdir / f"crossclass_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.csv"
    all_rows: list[dict] = []

    print(f"Output: {outdir}")
    print(f"Classes: {args.classes}")

    service = ibm.get_service()

    if "B" in args.classes:
        run_class_b_sweep(service, outdir, summary_path, all_rows,
                          backend_name=args.backend, shots=args.shots)
    if "C" in args.classes:
        run_class_c_sweep(service, outdir, summary_path, all_rows,
                          backend_name=args.backend, shots=args.shots)

    _write_summary(summary_path, all_rows)
    completed = sum(1 for r in all_rows if r.get("status") == "completed")
    failed = sum(1 for r in all_rows if r.get("status") == "failed")
    print(f"\nDone. {completed} completed, {failed} failed.")
    print(json.dumps({"classes": args.classes, "completed": completed,
                      "failed": failed, "outdir": str(outdir)}, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
