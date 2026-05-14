"""Smoke tests verifying the package imports and exposes its version."""

from __future__ import annotations

import importlib

import qimp


def test_package_imports() -> None:
    assert qimp.__version__


def test_version_is_string() -> None:
    assert isinstance(qimp.__version__, str)
    assert len(qimp.__version__) > 0


def test_subpackages_importable() -> None:
    for name in ("qimp.encoding", "qimp.io", "qimp.processing", "qimp.qml", "qimp.runtime"):
        assert importlib.import_module(name) is not None


def test_config_dataclass() -> None:
    from qimp.config import ProcessingConfig

    cfg = ProcessingConfig(n_spatial_qubits=3, q_intensity_qubits=4)
    assert cfg.n_spatial_qubits == 3
    assert cfg.q_intensity_qubits == 4


def test_config_rejects_invalid_qubit_counts() -> None:
    import pytest

    from qimp.config import ProcessingConfig

    with pytest.raises(ValueError):
        ProcessingConfig(n_spatial_qubits=0)
    with pytest.raises(ValueError):
        ProcessingConfig(q_intensity_qubits=0)
