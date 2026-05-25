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
import datetime as dt
import json
import sys
from pathlib import Path


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

    raise SystemExit("sweep loop not implemented yet -- see Tasks 13-16")


if __name__ == "__main__":
    raise SystemExit(main())
