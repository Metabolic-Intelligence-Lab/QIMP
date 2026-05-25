"""End-to-end run of the hardware-sweep CLI in --skip-hw mode against a
tiny synthetic image. Asserts the summary structure and figure presence
without making any network call."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture()
def synthetic_image(tmp_path: Path) -> Path:
    arr = np.random.default_rng(0).integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    p = tmp_path / "src.png"
    Image.fromarray(arr).save(p)
    return p


@pytest.fixture()
def repo_root() -> Path:
    """The path to the repository root, derived from the test file location."""
    return Path(__file__).resolve().parents[2]


def test_sweep_aer_only_produces_summary_and_figures(
    synthetic_image: Path, tmp_path: Path, repo_root: Path
) -> None:
    """End-to-end Aer-ideal sweep should produce summary.csv with 7 rows
    (one per encoder at n=1) and 7 non-trivial PNG figures."""
    out = tmp_path / "out"
    script = repo_root / "scripts" / "run_hardware_sweep.py"
    cmd = [
        sys.executable,
        str(script),
        "--image",
        str(synthetic_image),
        "--sizes",
        "1",
        "--skip-hw",
        "--outdir",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert r.returncode == 0, f"sweep exited with {r.returncode}: stderr={r.stderr}"

    runs = list(out.iterdir())
    assert len(runs) == 1, f"expected one timestamped run dir, got {len(runs)}"
    timestamp_dir = runs[0]

    summary_csv = timestamp_dir / "summary.csv"
    assert summary_csv.exists(), "summary.csv missing"

    with open(summary_csv) as f:
        rows = list(csv.DictReader(f))

    aer_ideal = [r for r in rows if r["pass"] == "aer-ideal"]
    assert len(aer_ideal) == 7, (
        f"expected 7 aer-ideal rows (one per encoder), got {len(aer_ideal)}: "
        f"{[r['label'] for r in aer_ideal]}"
    )

    encoders = {r["encoder"] for r in aer_ideal}
    assert encoders == {
        "frqi", "frqi_multi", "neqr", "qpie", "mcrqi", "ncqi", "gp"
    }, f"unexpected encoder set: {encoders}"

    figs = list((timestamp_dir / "figures").glob("*.png"))
    assert len(figs) == 7, f"expected 7 figures, got {len(figs)}"
    for f in figs:
        assert f.stat().st_size > 1024, (
            f"{f.name} is suspiciously small ({f.stat().st_size} bytes)"
        )


def test_sweep_aer_only_invalid_image_exits_nonzero(
    tmp_path: Path, repo_root: Path
) -> None:
    """Passing a non-existent --image should produce a clean error and a
    non-zero exit code, not a Python traceback."""
    script = repo_root / "scripts" / "run_hardware_sweep.py"
    cmd = [
        sys.executable,
        str(script),
        "--image",
        str(tmp_path / "missing.png"),
        "--sizes",
        "1",
        "--skip-hw",
        "--outdir",
        str(tmp_path / "out"),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert r.returncode != 0
    assert "cannot open" in r.stderr.lower() or "no such file" in r.stderr.lower()
