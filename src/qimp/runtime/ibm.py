"""IBM Quantum Runtime helpers — service singleton, backend resolution,
hardware execution via SamplerV2 with TREX + DD mitigation, Aer
noise-model fallback, and per-run artifact persistence.

All IBM-Runtime imports are local to this module: the rest of the
library imports only the typed helpers exposed here.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit

__all__ = [
    "aer_noisy_run",
    "get_service",
    "is_run_complete",
    "list_backends",
    "persist_run",
    "pick_backend",
]

logger = logging.getLogger(__name__)

# Module-level cache for the QiskitRuntimeService singleton. Tests reset
# this to None via the autouse `_reset_singleton` fixture.
_SERVICE: Any = None

# Indirection so tests can patch `ibm.QiskitRuntimeService` without the
# qiskit_ibm_runtime import happening at module-load time (the test fixture
# patches this attribute before `get_service` is called).
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
except ImportError:  # pragma: no cover - depends on optional `[ibm]` extra
    QiskitRuntimeService = None


def get_service(instance: str | None = None) -> Any:
    """Return the cached `QiskitRuntimeService` instance.

    Reads `~/.qiskit/qiskit-ibm.json` on first call. Raises a clear error
    if no saved credentials are found or the optional `[ibm]` extra is
    not installed.
    """
    global _SERVICE
    if _SERVICE is not None:
        return _SERVICE

    if QiskitRuntimeService is None:
        raise ImportError(
            "qimp.runtime.ibm requires `pip install qimp-mi[ibm]` "
            "(qiskit-ibm-runtime)"
        )

    try:
        _SERVICE = (
            QiskitRuntimeService(instance=instance)
            if instance
            else QiskitRuntimeService()
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not initialise QiskitRuntimeService — is "
            "~/.qiskit/qiskit-ibm.json present and the token valid? "
            f"Underlying error: {exc}"
        ) from exc

    return _SERVICE


def list_backends() -> list[dict[str, Any]]:
    """Return one row per operational backend visible to the saved account.

    Each row: ``{name, num_qubits, pending_jobs, operational}``.
    """
    service = get_service() if _SERVICE is None else _SERVICE
    rows: list[dict[str, Any]] = []
    for backend in service.backends(operational=True):
        status = backend.status()
        rows.append(
            {
                "name": backend.name,
                "num_qubits": backend.num_qubits,
                "pending_jobs": status.pending_jobs,
                "operational": status.operational,
            }
        )
    return rows


def pick_backend(
    service: Any,
    *,
    min_qubits: int,
    name: str | None = None,
) -> Any:
    """Return a backend with at least `min_qubits` operational qubits.

    If `name` is given, fetches that specific backend; otherwise asks the
    service for ``least_busy(operational=True, simulator=False,
    min_num_qubits=min_qubits)``. Raises ``ValueError`` if the chosen
    backend doesn't meet the qubit budget.
    """
    if name is not None:
        backend = service.backend(name)
    else:
        backend = service.least_busy(
            operational=True, simulator=False, min_num_qubits=min_qubits
        )

    if backend.num_qubits < min_qubits:
        raise ValueError(
            f"Backend {backend.name} has {backend.num_qubits} qubits, "
            f"needs >= {min_qubits}"
        )
    logger.info("Picked backend %s (%d qubits)", backend.name, backend.num_qubits)
    return backend


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def persist_run(
    outdir: Path,
    *,
    label: str,
    pass_name: str,
    circuit: QuantumCircuit,
    transpiled: QuantumCircuit | None,
    counts: dict[str, int],
    metadata: dict[str, Any],
) -> Path:
    """Persist a single run's artifacts under ``outdir/runs/<label>_<pass>/``.

    Writes (atomically):
      - ``circuit.qpy`` — the un-transpiled circuit
      - ``transpiled.qpy`` — only when ``transpiled is not None`` (HW runs)
      - ``counts.json`` — ``dict[str, int]``
      - ``metadata.json`` — caller-supplied dict (job_id, status, depths, ...)
    """
    import io

    from qiskit import qpy

    run_dir = Path(outdir) / "runs" / f"{label}_{pass_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    buf = io.BytesIO()
    qpy.dump(circuit, buf)
    _atomic_write_bytes(run_dir / "circuit.qpy", buf.getvalue())

    if transpiled is not None:
        buf = io.BytesIO()
        qpy.dump(transpiled, buf)
        _atomic_write_bytes(run_dir / "transpiled.qpy", buf.getvalue())

    _atomic_write_text(run_dir / "counts.json", json.dumps(counts))
    _atomic_write_text(run_dir / "metadata.json", json.dumps(metadata, default=str))
    return run_dir


def is_run_complete(outdir: Path, label: str, pass_name: str) -> bool:
    """True iff the run dir has a non-empty ``counts.json`` on disk."""
    counts_path = Path(outdir) / "runs" / f"{label}_{pass_name}" / "counts.json"
    if not counts_path.exists():
        return False
    try:
        return bool(json.loads(counts_path.read_text()))
    except json.JSONDecodeError:
        return False


def aer_noisy_run(
    qc: QuantumCircuit,
    *,
    backend: Any,
    shots: int = 4096,
) -> dict[str, int]:
    """Run ``qc`` on an Aer simulator wired with ``backend``'s noise model.

    Uses :func:`AerSimulator.from_backend` which inherits the device's
    noise + coupling map + basis gates. Adds ``measure_all()`` if the
    circuit has no measurements (reuses ``qimp.testing._ensure_measured``).
    """
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    from qimp.testing import _ensure_measured

    sim = AerSimulator.from_backend(backend)
    measured = _ensure_measured(qc)
    transpiled = transpile(measured, sim)
    result = sim.run(transpiled, shots=shots).result()
    return dict(result.get_counts())
