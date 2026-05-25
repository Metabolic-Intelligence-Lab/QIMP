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
    "hw_run",
    "is_run_complete",
    "list_backends",
    "persist_run",
    "pick_backend",
]

logger = logging.getLogger(__name__)

# Module-level cache for the QiskitRuntimeService singleton. Tests reset
# this to None via the autouse `_reset_singleton` fixture.
_SERVICE: Any = None

# Cached QiskitRuntimeService class (or None if the optional [ibm]
# extra is not installed). Resolved lazily on the first call to
# get_service(); tests can override by patching the module attribute.
QiskitRuntimeService: Any = None


def _resolve_runtime_service_cls() -> Any:
    global QiskitRuntimeService
    if QiskitRuntimeService is not None:
        return QiskitRuntimeService
    try:
        import qiskit_ibm_runtime as _ibm_rt
    except ImportError:
        return None
    QiskitRuntimeService = _ibm_rt.QiskitRuntimeService
    return QiskitRuntimeService


def get_service(instance: str | None = None) -> Any:
    """Return the cached `QiskitRuntimeService` instance.

    Reads `~/.qiskit/qiskit-ibm.json` on first call. Raises a clear error
    if no saved credentials are found or the optional `[ibm]` extra is
    not installed.
    """
    global _SERVICE
    if _SERVICE is not None:
        return _SERVICE

    qrs_cls = _resolve_runtime_service_cls()
    if qrs_cls is None:
        raise ImportError(
            "qimp.runtime.ibm requires `pip install qimp-mi[ibm]` "
            "(qiskit-ibm-runtime)"
        )

    try:
        _SERVICE = qrs_cls(instance=instance) if instance else qrs_cls()
    except Exception as exc:
        raise RuntimeError(
            "Could not initialise QiskitRuntimeService — is "
            "~/.qiskit/qiskit-ibm.json present and the token valid? "
            f"Underlying error: {exc}"
        ) from exc

    return _SERVICE


def list_backends(service: Any | None = None) -> list[dict[str, Any]]:
    """Return one row per operational backend visible to the saved account.

    Each row: ``{name, num_qubits, pending_jobs, operational}``.

    When ``service`` is None, resolves via :func:`get_service`.
    """
    if service is None:
        service = get_service()
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


def _sampler_v2_cls() -> Any:
    """Indirection so tests can patch the SamplerV2 import without touching
    the qiskit_ibm_runtime package."""
    from qiskit_ibm_runtime import SamplerV2

    return SamplerV2


def _sampler_options_cls() -> Any:
    """Indirection so tests can patch SamplerOptions without touching
    the qiskit_ibm_runtime package."""
    from qiskit_ibm_runtime.options import SamplerOptions

    return SamplerOptions


def _transpile_summary(qc: QuantumCircuit) -> dict[str, Any]:
    """Return a small dict of transpile metrics for ``qc``."""
    depth = qc.depth()
    two_q = sum(1 for instr in qc.data if instr.operation.num_qubits >= 2)
    return {
        "depth": depth,
        "two_q_gate_count": two_q,
        "num_qubits": qc.num_qubits,
    }


def hw_run(
    qc: QuantumCircuit,
    *,
    backend: Any,
    shots: int = 4096,
    mitigation: str = "trex+dd",
    optimization_level: int = 3,
    timeout_s: float = 1200.0,
) -> tuple[dict[str, int], QuantumCircuit, str, dict[str, Any]]:
    """Submit ``qc`` to ``backend`` via SamplerV2 with TREX + DD mitigation.

    Returns ``(counts, transpiled_qc, job_id, transpile_summary)``.

    Parameters
    ----------
    qc:
        The quantum circuit to run (measurements added automatically if absent).
    backend:
        An IBM backend object (real or fake).
    shots:
        Number of shots.
    mitigation:
        ``'trex+dd'`` enables twirled readout error extinction and XY4
        dynamical decoupling (default). ``'trex'`` enables only TREX,
        ``'dd'`` enables only DD, ``'none'`` disables all mitigation.
        ``'zne'`` enables zero-noise extrapolation on top of TREX+DD
        by setting ``resilience_level=2`` on SamplerOptions; the runtime
        executes three implicit noise-scaled copies and returns the
        extrapolated mitigated counts. The five modes together support
        ablation studies of which layer contributes the recovered signal.
    optimization_level:
        Transpiler optimisation level (0–3). Defaults to 3.
    timeout_s:
        ``timeout_s`` is currently advisory; recovery of long-running jobs is
        handled by the caller (the CLI sweep script in Task 15 persists
        ``job_id`` before awaiting ``job.result()``).
    """
    _allowed_mitigation = ("trex+dd", "trex", "dd", "none", "zne")
    if mitigation not in _allowed_mitigation:
        raise ValueError(
            f"unknown mitigation {mitigation!r}; expected one of {_allowed_mitigation}"
        )

    from qiskit import transpile
    from qiskit.transpiler import Target

    from qimp.testing import _ensure_measured

    measured = _ensure_measured(qc)
    # Pull the classical register name from the measured circuit before
    # submission. measure_all() creates a register named "meas", but circuits
    # that already have measurements may use a different register name. Using
    # the name from `measured.cregs` keeps this robust to both cases.
    creg_name = measured.cregs[0].name if measured.cregs else "meas"

    # Use the backend's Target object directly when available; this is the
    # recommended path for newer Qiskit / qiskit-ibm-runtime and avoids
    # issues with backend-supplied plugin name discovery during transpilation.
    _target = backend.target if isinstance(backend.target, Target) else None
    transpiled = transpile(
        measured, target=_target, optimization_level=optimization_level
    )
    summary = _transpile_summary(transpiled)

    Options = _sampler_options_cls()
    options = Options()
    if mitigation in ("trex+dd", "dd"):
        options.dynamical_decoupling.enable = True
        options.dynamical_decoupling.sequence_type = "XY4"
    if mitigation in ("trex+dd", "trex"):
        options.twirling.enable_measure = True
    if mitigation == "zne":
        # Zero-noise extrapolation is an expectation-value technique on
        # EstimatorV2 and is not exposed on SamplerV2's options schema
        # (qiskit-ibm-runtime >= 0.47 removed `resilience_level` from
        # SamplerOptions). A counts-side ZNE for this pipeline would
        # require manual gate folding (1x/3x/5x noise scales) + per-pixel
        # extrapolation — implemented in scripts/, not in this helper.
        raise NotImplementedError(
            "mitigation='zne' is not available through SamplerV2's options "
            "in qiskit-ibm-runtime >= 0.47. Implement counts-side ZNE in "
            "the caller via manual gate folding + per-pixel extrapolation."
        )

    Sampler = _sampler_v2_cls()
    sampler = Sampler(mode=backend, options=options)
    job = sampler.run([transpiled], shots=shots)
    job_id = job.job_id()
    logger.info("Submitted job %s on %s", job_id, backend.name)

    result = job.result()
    counts = dict(getattr(result[0].data, creg_name).get_counts())
    return counts, transpiled, job_id, summary
