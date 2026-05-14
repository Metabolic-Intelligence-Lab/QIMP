"""Smoke tests for the Streamlit app.

Page files call Streamlit widget APIs at import time (`st.title`,
`st.set_page_config`, etc.), which only work inside a real `streamlit run`
context. So the smoke tests here verify two things, without executing the
page bodies:

1. The Streamlit-free helpers (`_io`, `_viz`) import successfully.
2. Every page file is syntactically valid Python and resolves all imports
   that don't trigger Streamlit widget execution.

End-to-end UI testing (Playwright / Selenium) is intentionally deferred.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="install [ui] extra to run app tests")
pytest.importorskip("pandas", reason="install [ui] extra to run app tests")

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_repo_on_path() -> None:
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def test_io_module_imports() -> None:
    _ensure_repo_on_path()
    mod = importlib.import_module("apps.qimp_explorer._io")
    for name in (
        "DATASET_GRAYSCALE",
        "DATASET_RGB",
        "OUTPUT_ROOT",
        "discover_dataset_images",
        "infer_n_from_image",
        "is_power_of_two",
        "load_image",
        "new_output_dir",
        "save_named_panels",
        "save_tiff",
    ):
        assert hasattr(mod, name), f"_io missing {name}"


def test_viz_module_imports() -> None:
    _ensure_repo_on_path()
    mod = importlib.import_module("apps.qimp_explorer._viz")
    for name in ("image_figure", "panel_grid_figure", "bar_chart_figure", "safe_circuit_figure"):
        assert hasattr(mod, name), f"_viz missing {name}"


@pytest.mark.parametrize(
    "relative_path",
    [
        "apps/qimp_explorer/app.py",
        "apps/qimp_explorer/pages/1_Encoder_Explorer.py",
        "apps/qimp_explorer/pages/2_Processing_Playground.py",
        "apps/qimp_explorer/pages/3_Benchmark.py",
        "apps/qimp_explorer/pages/4_GP_Ratio.py",
    ],
)
def test_streamlit_page_compiles(relative_path: str) -> None:
    """Every page file must be syntactically valid Python.

    We compile, not exec, because top-level `st.title(...)` / `st.set_page_config(...)`
    requires a live Streamlit ScriptRunContext.
    """
    file_path = REPO_ROOT / relative_path
    assert file_path.exists(), f"{file_path} not found"
    source = file_path.read_text(encoding="utf-8")
    compile(source, str(file_path), "exec")


@pytest.mark.parametrize(
    "relative_path",
    [
        "apps/qimp_explorer/app.py",
        "apps/qimp_explorer/pages/1_Encoder_Explorer.py",
        "apps/qimp_explorer/pages/2_Processing_Playground.py",
        "apps/qimp_explorer/pages/3_Benchmark.py",
        "apps/qimp_explorer/pages/4_GP_Ratio.py",
    ],
)
def test_streamlit_page_runs(relative_path: str) -> None:
    """Run each page through Streamlit's AppTest harness.

    No real browser involved. Pages that depend on `session_state["image"]`
    will hit the early `st.stop()` branch (which is the correct happy path
    when no image has been picked) without raising.
    """
    from streamlit.testing.v1 import AppTest

    _ensure_repo_on_path()
    at = AppTest.from_file(str(REPO_ROOT / relative_path))
    at.run(timeout=30)
    assert not at.exception, f"Streamlit page raised: {at.exception}"


def test_helpers_save_named_panels(tmp_path: Path) -> None:
    """Round-trip the public helper used by every page's Save button."""
    _ensure_repo_on_path()
    import numpy as np
    from apps.qimp_explorer._io import save_named_panels

    panels = [
        ("alpha", np.zeros((4, 4), dtype=np.uint8)),
        ("beta", np.ones((4, 4), dtype=np.uint16)),
    ]
    written = save_named_panels(panels, tmp_path)
    assert len(written) == 2
    for path in written:
        assert path.exists()
        assert path.suffix == ".tif"
