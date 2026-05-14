"""qimp command-line interface."""

from __future__ import annotations

import argparse
import sys

from qimp import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qimp", description="Quantum Image Processing CLI")
    parser.add_argument("--version", action="version", version=f"qimp {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    # placeholders — populated during Fase 3/4
    subparsers.add_parser("encode", help="Encode an image into a quantum circuit")
    subparsers.add_parser("process", help="Apply a processing operation to an encoded circuit")
    subparsers.add_parser("measure", help="Measure a circuit (ideal/noisy/device)")
    subparsers.add_parser("benchmark", help="Benchmark a pipeline over varying n / q")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    print(f"`qimp {args.command}` not yet implemented (Fase 3+)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
