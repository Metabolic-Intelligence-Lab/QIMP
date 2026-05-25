"""Unit tests for qimp.runtime.ibm — all paths mocked, no network."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from qiskit import QuantumCircuit


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


def test_pick_backend_by_name():
    from qimp.runtime import ibm

    fake_backend = MagicMock()
    fake_backend.name = "ibm_brisbane"
    fake_backend.num_qubits = 127
    fake_service = MagicMock()
    fake_service.backend.return_value = fake_backend

    chosen = ibm.pick_backend(fake_service, min_qubits=8, name="ibm_brisbane")

    fake_service.backend.assert_called_once_with("ibm_brisbane")
    assert chosen is fake_backend


def test_pick_backend_least_busy():
    from qimp.runtime import ibm

    fake_backend = MagicMock()
    fake_backend.name = "ibm_sherbrooke"
    fake_backend.num_qubits = 127
    fake_service = MagicMock()
    fake_service.least_busy.return_value = fake_backend

    chosen = ibm.pick_backend(fake_service, min_qubits=8)

    fake_service.least_busy.assert_called_once_with(
        operational=True, simulator=False, min_num_qubits=8
    )
    assert chosen is fake_backend


def test_pick_backend_under_qubits_raises():
    from qimp.runtime import ibm

    fake_backend = MagicMock()
    fake_backend.name = "ibm_brisbane"
    fake_backend.num_qubits = 5
    fake_service = MagicMock()
    fake_service.backend.return_value = fake_backend

    with pytest.raises(ValueError, match="needs >= 8"):
        ibm.pick_backend(fake_service, min_qubits=8, name="ibm_brisbane")


def _toy_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def test_persist_run_writes_four_files_and_roundtrips_qpy(tmp_path):
    from qiskit import qpy

    from qimp.runtime import ibm

    qc = _toy_circuit()
    counts = {"00": 512, "11": 512}
    metadata = {"encoder": "frqi", "n": 1, "job_id": "fake-1", "status": "completed"}

    run_dir = ibm.persist_run(
        outdir=tmp_path,
        label="frqi_n1",
        pass_name="aer-ideal",
        circuit=qc,
        transpiled=None,
        counts=counts,
        metadata=metadata,
    )

    assert run_dir == tmp_path / "runs" / "frqi_n1_aer-ideal"
    assert (run_dir / "circuit.qpy").exists()
    assert (run_dir / "counts.json").exists()
    assert (run_dir / "metadata.json").exists()
    assert not (run_dir / "transpiled.qpy").exists()  # transpiled=None

    with open(run_dir / "circuit.qpy", "rb") as f:
        loaded = qpy.load(f)[0]
    assert loaded.num_qubits == 2

    assert json.loads((run_dir / "counts.json").read_text()) == counts
    assert json.loads((run_dir / "metadata.json").read_text())["status"] == "completed"


def test_persist_run_writes_transpiled_when_provided(tmp_path):
    from qimp.runtime import ibm

    qc = _toy_circuit()
    transpiled = _toy_circuit()  # placeholder

    run_dir = ibm.persist_run(
        outdir=tmp_path, label="gp_n1", pass_name="hw",
        circuit=qc, transpiled=transpiled, counts={}, metadata={"job_id": "x"},
    )
    assert (run_dir / "transpiled.qpy").exists()


def test_is_run_complete_true_when_counts_nonempty(tmp_path):
    from qimp.runtime import ibm

    qc = _toy_circuit()
    ibm.persist_run(
        outdir=tmp_path, label="frqi_n1", pass_name="aer-ideal",
        circuit=qc, transpiled=None, counts={"00": 1}, metadata={"status": "completed"},
    )
    assert ibm.is_run_complete(tmp_path, "frqi_n1", "aer-ideal")


def test_is_run_complete_false_when_missing(tmp_path):
    from qimp.runtime import ibm

    assert not ibm.is_run_complete(tmp_path, "frqi_n1", "aer-ideal")


def test_is_run_complete_false_when_counts_empty(tmp_path):
    from qimp.runtime import ibm

    qc = _toy_circuit()
    ibm.persist_run(
        outdir=tmp_path, label="frqi_n1", pass_name="hw",
        circuit=qc, transpiled=None, counts={}, metadata={"status": "submitted"},
    )
    assert not ibm.is_run_complete(tmp_path, "frqi_n1", "hw")
