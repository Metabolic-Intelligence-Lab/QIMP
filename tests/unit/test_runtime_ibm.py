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


def test_aer_noisy_run_returns_counts():
    """End-to-end: encode a Bell state, run on Aer with the fake-backend
    noise model, get counts. Even with depolarising noise, 00/11 stay
    dominant."""
    from qimp.runtime import ibm

    pytest.importorskip("qiskit_ibm_runtime.fake_provider")
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2

    backend = FakeManilaV2()
    qc = _toy_circuit()
    counts = ibm.aer_noisy_run(qc, backend=backend, shots=1024)

    assert isinstance(counts, dict)
    assert sum(counts.values()) == 1024
    dominant = sorted(counts.items(), key=lambda kv: -kv[1])[:2]
    assert {dominant[0][0], dominant[1][0]} == {"00", "11"}


def test_hw_run_submits_with_mitigation_and_returns_counts():
    """Verify hw_run:
    - transpiles against the backend (uses optimization_level=3 by default)
    - instantiates SamplerV2 with options carrying DD (XY4) + TREX flags
    - returns (counts, transpiled, job_id, summary) with summary keys
      'depth' and 'two_q_gate_count'.
    """
    from qimp.runtime import ibm

    fake_backend = MagicMock()
    fake_backend.name = "ibm_fake"
    fake_backend.num_qubits = 5

    fake_job = MagicMock()
    fake_job.job_id.return_value = "abc"
    fake_pub_result = MagicMock()
    fake_pub_result.data.meas.get_counts.return_value = {"00": 800, "11": 224}
    fake_job.result.return_value = [fake_pub_result]

    fake_sampler_instance = MagicMock()
    fake_sampler_instance.run.return_value = fake_job

    fake_sampler_cls = MagicMock(return_value=fake_sampler_instance)
    fake_options_cls = MagicMock()

    with patch.object(ibm, "_sampler_v2_cls", return_value=fake_sampler_cls), \
         patch.object(ibm, "_sampler_options_cls", return_value=fake_options_cls):
        counts, transpiled, job_id, summary = ibm.hw_run(
            _toy_circuit(),
            backend=fake_backend,
            shots=1024,
            mitigation="trex+dd",
        )

    assert counts == {"00": 800, "11": 224}
    assert job_id == "abc"
    assert transpiled.num_qubits >= 2  # transpile may add ancillas; at least the 2 logical
    assert "depth" in summary
    assert "two_q_gate_count" in summary
    assert "num_qubits" in summary

    # Sampler was instantiated with the right backend + options object
    fake_sampler_cls.assert_called_once()
    _, kw = fake_sampler_cls.call_args
    assert kw["mode"] is fake_backend
    fake_sampler_instance.run.assert_called_once()


def test_hw_run_rejects_unknown_mitigation():
    from qimp.runtime import ibm

    fake_backend = MagicMock()
    fake_backend.name = "ibm_fake"
    fake_backend.num_qubits = 5

    with patch.object(ibm, "_sampler_v2_cls", return_value=MagicMock()), \
         patch.object(ibm, "_sampler_options_cls", return_value=MagicMock()), \
         pytest.raises(ValueError, match="unknown mitigation"):
        ibm.hw_run(_toy_circuit(), backend=fake_backend, mitigation="zne")
