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


def _estimate_qpu_seconds(qc_depth: int, shots: int, avg_gate_ns: float = 100.0) -> float:
    """Order-of-magnitude QPU time estimate per circuit.

    Average single-/two-qubit gate time on Heron-class hardware is ~50-100 ns;
    we use 100 ns as a conservative upper bound. Result is in seconds.
    """
    return qc_depth * shots * avg_gate_ns * 1e-9


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


def _to_displayable(arr: np.ndarray) -> np.ndarray:
    """Reduce an array to 2D for imshow.

    RGB stacks (h, w, 3) pass through; everything else is squeezed.
    """
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return arr
    return np.asarray(arr).squeeze()


def _normalise_rgb_or_pass(arr: np.ndarray) -> np.ndarray:
    """RGB displays need uint-style [0,1] floats or uint8. Pass through 2D
    arrays unchanged."""
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        a = arr.astype(np.float64)
        a_max = a.max() if a.max() > 0 else 1.0
        return np.clip(a / a_max, 0.0, 1.0)
    return arr


def _render_figure(
    outdir: Path,
    label: str,
    ref: np.ndarray,
    panels: dict[str, np.ndarray],
) -> None:
    """Render a side-by-side comparison figure for one (encoder, n).

    Layout: [classical | decoded(pass_1) | |diff|(pass_1) | decoded(pass_2) |
             |diff|(pass_2) | ...] horizontally.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_pass = len(panels)
    if n_pass == 0:
        return

    ref_d = _to_displayable(ref.astype(np.float64))
    is_rgb = ref_d.ndim == 3
    cmap = None if is_rgb else "gray"
    diff_cmap = "magma"

    n_cols = 1 + 2 * n_pass
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3))
    if n_cols == 1:
        axes = [axes]

    # Normalisation: use ref's range for the decoded panels so the
    # comparison is fair (GP is [-1,1], intensities are [0, max]).
    vmin, vmax = float(ref_d.min()), float(ref_d.max())
    if vmin == vmax:
        vmax = vmin + 1.0  # avoid zero-range edge case

    axes[0].imshow(_normalise_rgb_or_pass(ref_d), cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("classical")
    axes[0].axis("off")

    for i, (pass_name, arr) in enumerate(panels.items()):
        arr_d = _to_displayable(np.asarray(arr, dtype=np.float64))
        axes[1 + 2 * i].imshow(
            _normalise_rgb_or_pass(arr_d), cmap=cmap, vmin=vmin, vmax=vmax
        )
        axes[1 + 2 * i].set_title(pass_name)
        axes[1 + 2 * i].axis("off")

        diff = np.abs(arr_d - ref_d)
        # For RGB diffs, sum across channels to get a single-channel magnitude.
        if diff.ndim == 3:
            diff = diff.sum(axis=-1) / diff.shape[-1]
        axes[2 + 2 * i].imshow(diff, cmap=diff_cmap)
        axes[2 + 2 * i].set_title(f"|diff| {pass_name}")
        axes[2 + 2 * i].axis("off")

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figdir / f"{label}.png", dpi=120)
    plt.close(fig)


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

    try:
        raw = np.asarray(PilImage.open(args.image))
    except (FileNotFoundError, PilImage.UnidentifiedImageError) as exc:
        print(f"--image: cannot open {args.image}: {exc}", file=sys.stderr)
        return 2

    timestamp = _utc_stamp()
    outdir = args.outdir / timestamp
    outdir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    panels_by_label: dict[str, dict[str, np.ndarray]] = {}

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
            panels_by_label.setdefault(label, {})["aer-ideal"] = decoded

    pass1_count = len(summary_rows)
    print(
        f"Pass 1 (Aer ideal) done. {pass1_count} rows"
    )

    # ---- Pass 2: Aer with backend's noise model ----
    # Pass 2 needs a backend (real or fake) for NoiseModel.from_backend.
    # Skip entirely when --skip-hw AND no explicit --backend was given —
    # the user is signalling "local statevector only".
    backend = None
    if (not args.skip_hw) or args.backend is not None:
        service = ibm.get_service()
        # Pick the backend ONCE. Use the largest circuit's qubit count as
        # the min_qubits filter so the same backend can host every recipe
        # in the sweep (both as noise source and as the Pass 3 target).
        max_qubits = max(
            rec.qc.num_qubits
            for n in args.sizes
            for rec in build_recipes(raw, n=n, q=args.q, alpha=args.alpha)
        )
        backend = ibm.pick_backend(service, min_qubits=max_qubits, name=args.backend)
        backend_info = {
            "name": backend.name,
            "num_qubits": backend.num_qubits,
            "basis_gates": list(getattr(backend, "basis_gates", []) or []),
        }
        (outdir / "backend_info.json").write_text(
            json.dumps(backend_info, indent=2, default=str)
        )
        print(f"Backend selected: {backend.name} ({backend.num_qubits} qubits)")

        for n in args.sizes:
            for rec in build_recipes(raw, n=n, q=args.q, alpha=args.alpha):
                label = rec.label
                if ibm.is_run_complete(outdir, label, "aer-noisy"):
                    print(f"[skip] {label} aer-noisy already on disk")
                    continue

                try:
                    counts = ibm.aer_noisy_run(
                        rec.qc, backend=backend, shots=args.shots
                    )
                except Exception as exc:
                    print(
                        f"[warn] aer-noisy failed for {label}: {exc}",
                        file=sys.stderr,
                    )
                    continue

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
                    "pass": "aer-noisy",
                    "shots": args.shots,
                    "backend": backend.name,
                    "mse": float(_mse(ref, dec_f)),
                    "psnr": float(_psnr(ref, dec_f, max_intensity=max_i)),
                    "depth": int(rec.qc.depth()),
                    "num_qubits": int(rec.qc.num_qubits),
                }
                summary_rows.append(row)
                ibm.persist_run(
                    outdir,
                    label=label,
                    pass_name="aer-noisy",
                    circuit=rec.qc,
                    transpiled=None,
                    counts=counts,
                    metadata={**row, "status": "completed"},
                )
                panels_by_label.setdefault(label, {})["aer-noisy"] = decoded
        print("Pass 2 (Aer noisy) done.")
    else:
        print("Pass 2 skipped (--skip-hw and no --backend)")

    # ---- Pass 3: hardware (whitelist) ----
    if backend is not None and not args.skip_hw:
        whitelist = _parse_hw_whitelist(args.hw_circuits)
        hw_recipes = [
            rec
            for n in args.sizes
            for rec in build_recipes(raw, n=n, q=args.q, alpha=args.alpha)
            if (rec.encoder, rec.n) in whitelist
        ]
        if not hw_recipes:
            print("Pass 3: empty whitelist after filtering, nothing to do.")
        else:
            est = sum(
                _estimate_qpu_seconds(r.qc.depth(), args.shots) for r in hw_recipes
            )
            print(
                f"Pass 3 estimate: {est:.1f} s QPU across {len(hw_recipes)} "
                f"circuits on {backend.name} ({args.shots} shots each)."
            )
            if est > 120.0:
                try:
                    ans = input(
                        "Estimate exceeds 120 s. Proceed? [y/N] "
                    ).strip().lower()
                except EOFError:
                    ans = "n"  # no tty -> safe abort
                if ans != "y":
                    print("Aborted by user (or non-interactive stdin).")
                    _write_summary(outdir / "summary.csv", summary_rows)
                    return 0

            for rec in hw_recipes:
                label = rec.label
                if ibm.is_run_complete(outdir, label, "hw"):
                    print(f"[skip] {label} hw already on disk")
                    continue

                # Persist a stub metadata BEFORE submission so a crash
                # mid-wait leaves a job_id on disk.
                stub_dir = outdir / "runs" / f"{label}_hw"
                stub_dir.mkdir(parents=True, exist_ok=True)
                (stub_dir / "metadata.json").write_text(
                    json.dumps(
                        {"label": label, "status": "submitted"},
                        default=str,
                    )
                )

                try:
                    counts, transpiled, job_id, tsummary = ibm.hw_run(
                        rec.qc,
                        backend=backend,
                        shots=args.shots,
                        mitigation="trex+dd",
                    )
                except Exception as exc:
                    print(
                        f"[warn] hw submission failed for {label}: {exc}",
                        file=sys.stderr,
                    )
                    continue

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
                    "pass": "hw",
                    "shots": args.shots,
                    "backend": backend.name,
                    "job_id": job_id,
                    "mse": float(_mse(ref, dec_f)),
                    "psnr": float(_psnr(ref, dec_f, max_intensity=max_i)),
                    "depth": tsummary["depth"],
                    "two_q_gate_count": tsummary["two_q_gate_count"],
                    "num_qubits": tsummary["num_qubits"],
                }
                summary_rows.append(row)
                ibm.persist_run(
                    outdir,
                    label=label,
                    pass_name="hw",
                    circuit=rec.qc,
                    transpiled=transpiled,
                    counts=counts,
                    metadata={**row, "status": "completed"},
                )
                panels_by_label.setdefault(label, {})["hw"] = decoded
            print("Pass 3 (hardware) done.")

    # ---- Figures ----
    # Recompute references from a fresh `build_recipes` call rather than
    # re-storing them in panels_by_label — references are deterministic
    # given (raw, n, q, alpha).
    for n in args.sizes:
        for rec in build_recipes(raw, n=n, q=args.q, alpha=args.alpha):
            if rec.label in panels_by_label:
                _render_figure(
                    outdir,
                    rec.label,
                    rec.reference,
                    panels_by_label[rec.label],
                )
    print(f"Figures rendered under {outdir / 'figures'}")

    _write_summary(outdir / "summary.csv", summary_rows)
    print(
        f"Summary written: {len(summary_rows)} rows in "
        f"{outdir / 'summary.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
