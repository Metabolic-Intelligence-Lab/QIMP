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
    mod = importlib.import_module("apps.qimp_explorer.app_io")
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
    for name in (
        "image_figure",
        "panel_grid_figure",
        "bar_chart_figure",
        "safe_circuit_figure",
        "to_display_uint8",
    ):
        assert hasattr(mod, name), f"_viz missing {name}"


def test_to_display_uint8_handles_common_shapes() -> None:
    _ensure_repo_on_path()
    import numpy as np
    from apps.qimp_explorer._viz import to_display_uint8

    # uint8 grayscale: pass-through after normalization.
    out = to_display_uint8(np.arange(16, dtype=np.uint8).reshape(4, 4))
    assert out.shape == (4, 4) and out.dtype == np.uint8
    assert out.max() == 255

    # uint16 RGB: down-shifted to uint8.
    rgb16 = np.full((4, 4, 3), 30000, dtype=np.uint16)
    out_rgb = to_display_uint8(rgb16)
    assert out_rgb.shape == (4, 4, 3) and out_rgb.dtype == np.uint8

    # Float grayscale with arbitrary range: linear stretched.
    out_f = to_display_uint8(np.linspace(-3.0, 7.0, 16).reshape(4, 4))
    assert out_f.shape == (4, 4) and out_f.dtype == np.uint8
    assert out_f.min() == 0 and out_f.max() == 255

    # All-zero image: returns zeros.
    zeros = np.zeros((4, 4), dtype=np.float32)
    assert to_display_uint8(zeros).max() == 0


@pytest.mark.parametrize(
    "relative_path",
    [
        "apps/qimp_explorer/app.py",
        "apps/qimp_explorer/pages/1_Benchmark.py",
        "apps/qimp_explorer/pages/2_GP_Ratio.py",
        "apps/qimp_explorer/pages/3_System_Info.py",
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
        "apps/qimp_explorer/pages/1_Benchmark.py",
        "apps/qimp_explorer/pages/2_GP_Ratio.py",
        "apps/qimp_explorer/pages/3_System_Info.py",
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


@pytest.mark.parametrize("target", [2, 4, 8, 16, 32, 64])
def test_resize_to_square_preserves_shape_and_dtype_grayscale(target: int) -> None:
    _ensure_repo_on_path()
    import numpy as np
    from apps.qimp_explorer.app_io import resize_to_square

    img = np.arange(40 * 30, dtype=np.uint8).reshape(40, 30)
    out = resize_to_square(img, target)
    assert out.shape == (target, target)
    assert out.dtype == np.uint8


def test_resize_to_square_handles_rgb() -> None:
    _ensure_repo_on_path()
    import numpy as np
    from apps.qimp_explorer.app_io import resize_to_square

    img = np.zeros((110, 110, 3), dtype=np.uint8)
    img[:, :, 0] = 200  # solid red
    out = resize_to_square(img, 16)
    assert out.shape == (16, 16, 3)
    # Center pixels should still be red after LANCZOS down-sample.
    assert out[8, 8, 0] > 150


def test_resize_to_square_preserves_uint16() -> None:
    _ensure_repo_on_path()
    import numpy as np
    from apps.qimp_explorer.app_io import resize_to_square

    img = (np.arange(32 * 32, dtype=np.uint16) * 100).reshape(32, 32)
    out = resize_to_square(img, 8)
    assert out.shape == (8, 8)
    assert out.dtype == np.uint16


def test_resize_to_square_center_crops_non_square() -> None:
    _ensure_repo_on_path()
    import numpy as np
    from apps.qimp_explorer.app_io import resize_to_square

    img = np.zeros((100, 50), dtype=np.uint8)
    img[40:60, 15:35] = 255  # bright square at the geometric center
    out = resize_to_square(img, 8)
    assert out.shape == (8, 8)
    # After center-cropping a 100×50 to 50×50 then resizing to 8×8, the bright
    # square is roughly at the centre.
    assert out[3:5, 3:5].mean() > out[0:2, 0:2].mean()


def test_resize_to_square_rejects_non_pow2_target() -> None:
    _ensure_repo_on_path()
    import numpy as np
    from apps.qimp_explorer.app_io import resize_to_square

    with pytest.raises(ValueError, match="power of two"):
        resize_to_square(np.zeros((16, 16), dtype=np.uint8), target_side=10)


def test_to_grayscale_preserves_2d_input() -> None:
    _ensure_repo_on_path()
    import numpy as np
    from apps.qimp_explorer.app_io import to_grayscale

    img = np.arange(16, dtype=np.uint8).reshape(4, 4)
    out = to_grayscale(img)
    assert out is img or np.array_equal(out, img)


def test_to_grayscale_converts_rgb_via_luma() -> None:
    _ensure_repo_on_path()
    import numpy as np
    from apps.qimp_explorer.app_io import to_grayscale

    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[..., 1] = 100  # pure green
    out = to_grayscale(img)
    assert out.shape == (4, 4)
    assert out.dtype == np.uint8
    # BT.601: green coefficient is 0.587 → ≈ 58.
    assert 55 <= int(out[0, 0]) <= 62


def test_to_grayscale_rejects_unsupported_shape() -> None:
    _ensure_repo_on_path()
    import numpy as np
    import pytest as _pytest
    from apps.qimp_explorer.app_io import to_grayscale

    with _pytest.raises(ValueError, match="grayscale"):
        to_grayscale(np.zeros((2, 2, 5), dtype=np.uint8))


def test_ibm_module_qasm_export() -> None:
    """`circuit_to_qasm3` should produce a parsable string for a tiny circuit."""
    _ensure_repo_on_path()
    from apps.qimp_explorer._ibm import circuit_to_qasm3
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    qasm = circuit_to_qasm3(qc)
    assert isinstance(qasm, str)
    assert len(qasm) > 0
    # Either QASM 3 or QASM 2 header.
    assert "OPENQASM" in qasm.upper()


def test_ibm_module_have_runtime_returns_bool() -> None:
    _ensure_repo_on_path()
    from apps.qimp_explorer._ibm import have_ibm_runtime

    assert isinstance(have_ibm_runtime(), bool)


def test_helpers_save_named_panels(tmp_path: Path) -> None:
    """Round-trip the public helper used by every page's Save button."""
    _ensure_repo_on_path()
    import numpy as np
    from apps.qimp_explorer.app_io import save_named_panels

    panels = [
        ("alpha", np.zeros((4, 4), dtype=np.uint8)),
        ("beta", np.ones((4, 4), dtype=np.uint16)),
    ]
    written = save_named_panels(panels, tmp_path)
    assert len(written) == 2
    for path in written:
        assert path.exists()
        assert path.suffix == ".tif"
