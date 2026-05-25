#!/usr/bin/env python
"""Robustness experiments — round 2 (E+F+G) for the paper revision.

Three additional experiments designed to (i) generalise the C-finding of
round 1 (no-mitigation > TREX+DD), (ii) extend the hardware validation
to the other two ratiometric classes (Lemmas 1, 2), and (iii) supply a
killer baseline that pins the recovered signal on the per-pixel locality
of the corrected ansatz rather than on any other property of the run.

- E : Mitigation ablation at n=1 — gp@n=1 under {none, trex, dd,
      trex+dd}. Verifies that the round-1 "mitigation hurts" result
      generalises to the shallower circuit.

- F : Cross-class hardware validation — Class B (Fura-2 calcium ratio)
      and Class C (roGFP2 redox) on synthetic targets at n=2 via the
      `analytical_gp_params(target=u_B|u_C)` entry point. Confirms that
      the hardware survival is a property of the bounded-target class,
      not of the specific GP calibration.

- G : Naive vs corrected ansatz on hardware — at n=2, the textbook
      naive block of §4.1 (CRY controlled only by the selection qubit,
      H inside the per-pixel loop) is applied to the same FRQI encoding
      with the same closed-form parameter vector that drives the
      corrected ansatz. By failure modes 1 and 2 of §4 the naive output
      collapses to a single effective rotation and an index-parity sign
      flip — so the decoded image cannot represent a non-constant
      target. Observing this collapse on hardware (alongside the
      corrected-ansatz survival) is the cleanest test that the recovered
      signal is the work of the architectural correction.

Outputs land under data/output/ibm/robustness2_<UTC-timestamp>/. All
hardware jobs are submitted sequentially, persisted as soon as they
complete, and resumable by re-launching with the same --outdir.
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
from qiskit.circuit.library import HGate, RYGate

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
from qimp.runtime.circuits import build_recipes

CANONICAL_FRAME = "membraneStack_Sample011_L_UV_DC_001rbc3DM2.tif"
MITIGATION_MODES = ["none", "trex", "dd", "trex+dd"]


# ------------------ utilities -----------------------------------------------


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


def _submit_and_persist(
    *, key: str, experiment: str, qc, decoder, reference,
    backend, shots: int, mitigation: str, extra: dict, outdir: Path,
) -> dict:
    print(f"[{key}] submitting on {backend.name} (mitigation={mitigation})...",
          flush=True)
    try:
        counts, transpiled, job_id, tsummary = ibm.hw_run(
            qc, backend=backend, shots=shots, mitigation=mitigation,
        )
    except Exception as exc:
        print(f"[{key}] FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        return {
            "key": key, "experiment": experiment, "status": "failed",
            "error": str(exc), "mitigation": mitigation, **extra,
        }
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
    print(f"[{key}] done. PSNR={row['psnr']:.2f} dB  job_id={job_id}")
    return row


# ------------------ Experiment E: n=1 ablation ------------------------------


def run_experiment_e(service, repo_root, outdir, summary_path, all_rows,
                     backend_name, shots):
    print("\n=== Experiment E: mitigation ablation at n=1 ===\n")
    img = _load_image(repo_root, CANONICAL_FRAME)
    rec = next(r for r in build_recipes(img, n=1, alpha=0.5) if r.encoder == "gp")
    backend = ibm.pick_backend(service, min_qubits=4, name=backend_name)
    for m in MITIGATION_MODES:
        key = f"E_{m.replace('+','_')}_gp_n1"
        if ibm.is_run_complete(outdir, key, "hw"):
            print(f"[{key}] already on disk, skipping")
            continue
        row = _submit_and_persist(
            key=key, experiment="E_mitigation_ablation_n1",
            qc=rec.qc, decoder=rec.decoder, reference=rec.reference,
            backend=backend, shots=shots, mitigation=m,
            extra={"frame": CANONICAL_FRAME, "n": 1, "alpha": 0.5},
            outdir=outdir,
        )
        _append(summary_path, row, all_rows)


# ------------------ Experiment F: cross-class HW ----------------------------


def _synthesise_fura2(side: int) -> tuple[np.ndarray, np.ndarray]:
    """Simplified Fura-2 synthesis for small `side` (n=2 means side=4).

    Two co-registered intensity arrays from a smooth calcium gradient with
    Grynkiewicz calibration. No puncta (would be off-grid at side=4).
    """
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float64)
    rr = np.sqrt((yy - (side - 1) / 2) ** 2 + (xx - (side - 1) / 2) ** 2) / (side / 2)
    ca = 80.0 + 600.0 * np.exp(-(rr ** 2) * 1.5)
    ca += rng.normal(0.0, 5.0, ca.shape)
    ca = np.clip(ca, 10.0, 2000.0)
    K_d, Sf340, Sb340, Sf380, Sb380 = 224.0, 50.0, 250.0, 250.0, 50.0
    f340 = Sf340 + (Sb340 - Sf340) * ca / (K_d + ca)
    f380 = Sf380 + (Sb380 - Sf380) * ca / (K_d + ca)
    return f340, f380


def _synthesise_rogfp(side: int) -> tuple[np.ndarray, np.ndarray]:
    """Simplified roGFP2 synthesis for small `side`.

    Linear top-to-bottom oxidation-degree gradient. Same Schwarzländer-Hanson
    spectral coefficients as Section 6.7's gen_cross_class_validation.py.
    """
    yy, _ = np.mgrid[0:side, 0:side].astype(np.float64)
    oxd = yy / (side - 1)  # 0 at top -> 1 at bottom
    R_red, R_ox = 0.20, 4.50
    R = R_red + (R_ox - R_red) * oxd
    # spectral coefficients normalised so the brightest channel is ~256
    iso = 100.0  # isosbestic baseline brightness
    f405 = iso * R / (1.0 + R)
    f488 = iso * 1.0 / (1.0 + R)
    return f405, f488


def run_experiment_f(service, repo_root, outdir, summary_path, all_rows,
                     backend_name, shots, n=2):
    print(f"\n=== Experiment F: cross-class HW (Fura-2 + roGFP2 at n={n}) ===\n")
    side = 1 << n
    backend = ibm.pick_backend(service, min_qubits=2 * n + 2, name=backend_name)

    # --- Class B: Fura-2 (Lemma 1: u_B = (R-1)/(R+1)) ---
    f340, f380 = _synthesise_fura2(side)
    R = f340 / (f380 + 1e-12)
    u_B = (R - 1.0) / (R + 1.0)
    # Use F340 as "green" and F380 as "red" placeholders for the encoder.
    rg_stack = np.stack([f380, f340], axis=0).astype(np.float64)
    norm_B = float(rg_stack.max()) or 1.0
    qc_B = frqi_circuit(rg_stack, normalization=norm_B)
    params_B = analytical_gp_params(f340, f380, target=u_B,
                                    normalization=norm_B)
    apply_gp_function(qc_B, n=n, m=1, params=list(params_B))

    def _decode_b(counts: dict[str, int]) -> np.ndarray:
        return decode_gp_counts(counts, n=n, m=1)

    key_B = f"F_classB_fura2_n{n}"
    if not ibm.is_run_complete(outdir, key_B, "hw"):
        row = _submit_and_persist(
            key=key_B, experiment="F_class_B_fura2",
            qc=qc_B, decoder=_decode_b, reference=u_B,
            backend=backend, shots=shots, mitigation="trex+dd",
            extra={"class": "B", "synthesis": "fura2", "n": n, "side": side,
                   "Kd_nM": 224.0},
            outdir=outdir,
        )
        _append(summary_path, row, all_rows)
    else:
        print(f"[{key_B}] already on disk, skipping")

    # --- Class C: roGFP2 (Lemma 2: u_C = 2*R_C - 1) ---
    f405, f488 = _synthesise_rogfp(side)
    R = f405 / (f488 + 1e-12)
    R_red, R_ox = 0.20, 4.50
    R_C = np.clip((R - R_red) / (R_ox - R_red), 0.0, 1.0)
    u_C = 2.0 * R_C - 1.0
    rg_stack = np.stack([f488, f405], axis=0).astype(np.float64)
    norm_C = float(rg_stack.max()) or 1.0
    qc_C = frqi_circuit(rg_stack, normalization=norm_C)
    params_C = analytical_gp_params(f405, f488, target=u_C,
                                    normalization=norm_C)
    apply_gp_function(qc_C, n=n, m=1, params=list(params_C))

    def _decode_c(counts: dict[str, int]) -> np.ndarray:
        return decode_gp_counts(counts, n=n, m=1)

    key_C = f"F_classC_rogfp_n{n}"
    if not ibm.is_run_complete(outdir, key_C, "hw"):
        row = _submit_and_persist(
            key=key_C, experiment="F_class_C_rogfp",
            qc=qc_C, decoder=_decode_c, reference=u_C,
            backend=backend, shots=shots, mitigation="trex+dd",
            extra={"class": "C", "synthesis": "rogfp", "n": n, "side": side,
                   "R_red": R_red, "R_ox": R_ox},
            outdir=outdir,
        )
        _append(summary_path, row, all_rows)
    else:
        print(f"[{key_C}] already on disk, skipping")


# ------------------ Experiment G: naive vs corrected ------------------------


def _apply_gp_function_naive(
    qc: QuantumCircuit, n: int, m: int, params: list[float],
) -> QuantumCircuit:
    """Textbook naive ansatz (paper §4.1).

    For each pixel, two CRY gates controlled ONLY by the selection qubit
    (failure mode 1 — position register ignored) and a Hadamard INSIDE the
    per-pixel loop (failure mode 2 — index-parity sign flip across pixels).
    Same parameter layout as apply_gp_function so the closed-form vector
    can be reused for the comparison.
    """
    expected = 2 * (1 << (2 * n))
    if len(params) != expected:
        raise ValueError(f"expected {expected} parameters, got {len(params)}")
    if m < 1:
        raise ValueError("m must be >= 1")
    sel = 2 * n
    color = 2 * n + m
    idx = 0
    for pixel_idx in range(1 << (2 * n)):
        pos_bits = format(pixel_idx, f"0{2 * n}b")[::-1]
        flips = [q for q, bit in enumerate(pos_bits) if bit == "0"]
        for q in flips:
            qc.x(q)
        # Failure mode 1: CRY conditioned ONLY on the selection qubit.
        qc.append(RYGate(params[idx]).control(1), [sel, color])
        idx += 1
        qc.x(sel)
        qc.append(RYGate(params[idx]).control(1), [sel, color])
        idx += 1
        qc.x(sel)
        for q in flips:
            qc.x(q)
        # Failure mode 2: Hadamard INSIDE the per-pixel loop.
        qc.append(HGate(), [color])
        qc.barrier()
    return qc


def run_experiment_g(service, repo_root, outdir, summary_path, all_rows,
                     backend_name, shots, n=2):
    print(f"\n=== Experiment G: naive vs corrected ansatz at n={n} ===\n")
    backend = ibm.pick_backend(service, min_qubits=2 * n + 2, name=backend_name)
    img = _load_image(repo_root, CANONICAL_FRAME)

    # The downsampled (R, G) channels — same as build_recipes("gp"@n).
    from qimp.runtime.circuits import _downsample_to_n
    rgb = _downsample_to_n(img, n=n)
    g_chan = rgb[..., 1].astype(np.float64)
    r_chan = rgb[..., 0].astype(np.float64)
    rg_stack = np.stack([r_chan, g_chan], axis=0)
    norm = float(rg_stack.max()) or 1.0

    params = analytical_gp_params(g_chan, r_chan, alpha=0.5, normalization=norm)
    target = classical_gp_image(g_chan, r_chan, alpha=0.5)

    # --- corrected (control: the textbook locality-fixed ansatz) ---
    qc_corr = frqi_circuit(rg_stack, normalization=norm)
    apply_gp_function(qc_corr, n=n, m=1, params=list(params))

    def _decode(counts: dict[str, int]) -> np.ndarray:
        return decode_gp_counts(counts, n=n, m=1)

    key_corr = f"G_corrected_gp_n{n}"
    if not ibm.is_run_complete(outdir, key_corr, "hw"):
        row = _submit_and_persist(
            key=key_corr, experiment="G_naive_vs_corrected",
            qc=qc_corr, decoder=_decode, reference=target,
            backend=backend, shots=shots, mitigation="trex+dd",
            extra={"ansatz": "corrected", "frame": CANONICAL_FRAME, "n": n,
                   "alpha": 0.5},
            outdir=outdir,
        )
        _append(summary_path, row, all_rows)
    else:
        print(f"[{key_corr}] already on disk, skipping")

    # --- naive (textbook §4.1, same parameters) ---
    qc_naive = frqi_circuit(rg_stack, normalization=norm)
    _apply_gp_function_naive(qc_naive, n=n, m=1, params=list(params))

    key_naive = f"G_naive_gp_n{n}"
    if not ibm.is_run_complete(outdir, key_naive, "hw"):
        row = _submit_and_persist(
            key=key_naive, experiment="G_naive_vs_corrected",
            qc=qc_naive, decoder=_decode, reference=target,
            backend=backend, shots=shots, mitigation="trex+dd",
            extra={"ansatz": "naive", "frame": CANONICAL_FRAME, "n": n,
                   "alpha": 0.5},
            outdir=outdir,
        )
        _append(summary_path, row, all_rows)
    else:
        print(f"[{key_naive}] already on disk, skipping")


# ------------------ driver --------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qimp-robustness-2", description=__doc__)
    p.add_argument("--experiments", nargs="+", default=["E", "F", "G"],
                   choices=["E", "F", "G"])
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--backend", type=str, default="ibm_marrakesh")
    p.add_argument("--outdir", type=Path, default=Path("data/output/ibm"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    stamp = _utc_stamp()
    outdir = args.outdir / f"robustness2_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.csv"
    all_rows: list[dict] = []

    print(f"Output: {outdir}")
    print(f"Experiments: {args.experiments}")

    service = ibm.get_service()

    if "E" in args.experiments:
        run_experiment_e(service, repo_root, outdir, summary_path, all_rows,
                         backend_name=args.backend, shots=args.shots)
    if "F" in args.experiments:
        run_experiment_f(service, repo_root, outdir, summary_path, all_rows,
                         backend_name=args.backend, shots=args.shots, n=2)
    if "G" in args.experiments:
        run_experiment_g(service, repo_root, outdir, summary_path, all_rows,
                         backend_name=args.backend, shots=args.shots, n=2)

    _write_summary(summary_path, all_rows)
    completed = sum(1 for r in all_rows if r.get("status") == "completed")
    failed = sum(1 for r in all_rows if r.get("status") == "failed")
    print(f"\nDone. {completed} completed, {failed} failed.")
    print(json.dumps({
        "experiments_run": args.experiments,
        "completed": completed, "failed": failed,
        "outdir": str(outdir),
    }, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
