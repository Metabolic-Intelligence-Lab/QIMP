#!/usr/bin/env python
"""Run the qimp-mi encoder + GP suite on Aer (ideal + noisy) and optionally
on IBM Quantum hardware, with persisted artifacts and per-circuit metrics.

Designed for the IBM Quantum Open (free) plan: hardware execution is
restricted to a whitelist of high-value circuits (default: gp@1, gp@2,
frqi_multi@1) to stay well within the ~10 min/month QPU budget.

Tasks 13-16 of the implementation plan fill in the three-pass sweep loop;
this scaffold provides argparse + the --list-backends diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np


def _parse_hw_whitelist(raw: list[str]) -> set[tuple[str, int]]:
    """Parse '<encoder>@<n>' tokens, e.g. 'gp@2 frqi_multi@1'."""
    out: set[tuple[str, int]] = set()
    for tok in raw:
        if "@" not in tok:
            raise SystemExit(f"--hw-circuits expects '<encoder>@<n>', got {tok!r}")
        enc, n_str = tok.split("@", 1)
        try:
            out.add((enc.strip(), int(n_str)))
        except ValueError as exc:
            raise SystemExit(
                f"--hw-circuits: cannot parse n from {tok!r}: {exc}"
            ) from exc
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qimp-hw-sweep",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--image", type=Path, help="Path to source image (TIFF/PNG)")
    p.add_argument(
        "--sizes", type=int, nargs="+", default=[1, 2],
        help="Spatial qubit counts n to sweep (default: 1 2)",
    )
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument(
        "--q", type=int, default=2,
        help="NEQR/NCQI intensity qubits (default 2)",
    )
    p.add_argument(
        "--alpha", type=float, default=0.5,
        help="GP G-factor (default 0.5)",
    )
    p.add_argument(
        "--backend", type=str, default=None,
        help="Backend name (default: least-busy on the saved account)",
    )
    p.add_argument(
        "--hw-circuits", nargs="*",
        default=["gp@1", "gp@2", "frqi_multi@1"],
        help="Whitelist of <encoder>@<n> for hardware execution",
    )
    p.add_argument("--outdir", type=Path, default=Path("data/output/ibm"))
    p.add_argument(
        "--skip-hw", action="store_true",
        help="Skip the hardware pass entirely (Aer ideal + Aer noisy only)",
    )
    p.add_argument(
        "--list-backends", action="store_true",
        help="Print available backends and exit",
    )
    return p


def _utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def _write_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _max_intensity_for_metric(ref: np.ndarray) -> float:
    """Pick a sensible max-intensity for PSNR computation.

    - Range looks like [-1, 1] (GP) -> use 2.0 (peak-to-peak).
    - Anything else -> max(ref.max(), 1.0).
    """
    if ref.min() < 0:
        return 2.0
    return float(max(ref.max(), 1.0))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.list_backends:
        from qimp.runtime import ibm

        ibm.get_service()
        rows = ibm.list_backends()
        print(json.dumps(rows, indent=2, default=str))
        return 0

    if args.image is None:
        print("--image required (or use --list-backends)", file=sys.stderr)
        return 2

    # Validate --sizes are powers of two >= 2 (i.e. n >= 1).
    for n in args.sizes:
        if n < 1:
            print(f"--sizes: n must be >= 1, got {n}", file=sys.stderr)
            return 2

    # Validate --hw-circuits tokens up front (even though Pass 3 isn't here yet)
    # so a malformed flag fails fast.
    _parse_hw_whitelist(args.hw_circuits)

    from PIL import Image as PilImage

    from qimp.metrics import mse as _mse
    from qimp.metrics import psnr as _psnr
    from qimp.runtime import ibm
    from qimp.runtime.circuits import build_recipes
    from qimp.testing import exact_counts

    raw = np.asarray(PilImage.open(args.image))

    timestamp = _utc_stamp()
    outdir = args.outdir / timestamp
    outdir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    # ---- Pass 1: Aer ideal (noise-free statevector) ----
    for n in args.sizes:
        recipes = build_recipes(raw, n=n, q=args.q, alpha=args.alpha)
        for rec in recipes:
            label = rec.label
            if ibm.is_run_complete(outdir, label, "aer-ideal"):
                print(f"[skip] {label} aer-ideal already on disk")
                continue

            counts = exact_counts(rec.qc)
            decoded = rec.decoder(counts)

            ref = rec.reference.astype(np.float64)
            dec_f = np.asarray(decoded, dtype=np.float64)
            max_i = _max_intensity_for_metric(ref)

            row = {
                "label": label,
                "encoder": rec.encoder,
                "n": rec.n,
                "q": rec.q,
                "m": rec.m,
                "pass": "aer-ideal",
                "shots": "exact",
                "mse": float(_mse(ref, dec_f)),
                "psnr": float(_psnr(ref, dec_f, max_intensity=max_i)),
                "depth": int(rec.qc.depth()),
                "num_qubits": int(rec.qc.num_qubits),
            }
            summary_rows.append(row)
            ibm.persist_run(
                outdir,
                label=label,
                pass_name="aer-ideal",
                circuit=rec.qc,
                transpiled=None,
                counts=counts,
                metadata={**row, "status": "completed"},
            )

    _write_summary(outdir / "summary.csv", summary_rows)
    print(
        f"Pass 1 (Aer ideal) done. {len(summary_rows)} rows in "
        f"{outdir / 'summary.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
