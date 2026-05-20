"""qimp command-line interface."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from qimp import __version__


def _run_ui(_args: argparse.Namespace) -> int:
    """Launch the Streamlit explorer app via `streamlit run`."""
    if shutil.which("streamlit") is None:
        print(
            'streamlit not found. Install the UI extras:\n    pip install "qimp-mi[ui]"',
            file=sys.stderr,
        )
        return 1

    # Locate apps/qimp_explorer/app.py — package install or editable check-out.
    # The package lives at <root>/src/qimp/, and the app at <root>/apps/qimp_explorer/.
    pkg_root = Path(__file__).resolve().parent
    repo_root = pkg_root.parents[1]  # …/repo
    app_path = repo_root / "apps" / "qimp_explorer" / "app.py"
    if not app_path.exists():
        print(
            f"Streamlit app not found at {app_path}. The `qimp ui` subcommand "
            "only works from a development checkout of the repository.",
            file=sys.stderr,
        )
        return 1

    result = subprocess.run(
        [
            "streamlit",
            "run",
            str(app_path),
            "--browser.gatherUsageStats=false",
        ],
        check=False,
    )
    return int(result.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qimp", description="Quantum Image Processing CLI")
    parser.add_argument("--version", action="version", version=f"qimp {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    ui_parser = subparsers.add_parser(
        "ui",
        help="Launch the Streamlit QIMP Explorer (requires the [ui] extra)",
    )
    ui_parser.set_defaults(func=_run_ui)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
