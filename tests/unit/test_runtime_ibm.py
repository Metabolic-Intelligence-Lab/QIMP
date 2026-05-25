"""Unit tests for qimp.runtime.ibm — all paths mocked, no network."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Each test starts with a fresh service singleton."""
    from qimp.runtime import ibm

    ibm._SERVICE = None
    yield
    ibm._SERVICE = None


def test_get_service_caches_singleton():
    from qimp.runtime import ibm

    fake_service = MagicMock()
    with patch.object(ibm, "QiskitRuntimeService", return_value=fake_service) as ctor:
        a = ibm.get_service()
        b = ibm.get_service()
    assert a is b is fake_service
    assert ctor.call_count == 1


def test_list_backends_returns_serialisable_rows():
    from qimp.runtime import ibm

    fake_backend = MagicMock()
    fake_backend.name = "ibm_brisbane"
    fake_backend.num_qubits = 127
    fake_backend.status.return_value.pending_jobs = 17
    fake_backend.status.return_value.operational = True
    fake_service = MagicMock()
    fake_service.backends.return_value = [fake_backend]

    with patch.object(ibm, "_SERVICE", fake_service):
        rows = ibm.list_backends()

    assert rows == [
        {"name": "ibm_brisbane", "num_qubits": 127, "pending_jobs": 17, "operational": True}
    ]
