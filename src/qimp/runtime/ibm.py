"""IBM Quantum Runtime helpers — service singleton, backend resolution,
hardware execution via SamplerV2 with TREX + DD mitigation and
counts-side zero-noise extrapolation, Aer noise-model fallback, and
per-run artifact persistence.

All IBM-Runtime imports are local to this module: the rest of the
library imports only the typed helpers exposed here.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit

from qimp.runtime.zne import ZNE_METHODS, fold_gates, zne_extrapolate, zne_probabilities

__all__ = [
    "ZNE_METHODS",
    "aer_noisy_run",
    "fold_gates",
    "get_service",
    "hw_run",
    "is_run_complete",
    "list_backends",
    "persist_run",
    "pick_backend",
    "zne_extrapolate",
    "zne_probabilities",
]

# Default noise scales for mitigation="zne" (global folding 1x, 3x, 5x).
DEFAULT_ZNE_SCALES: tuple[float, ...] = (1.0, 3.0, 5.0)

logger = logging.getLogger(__name__)

# Module-level cache for the QiskitRuntimeService singleton. Tests reset
# this to None via the autouse `_reset_singleton` fixture.
_SERVICE: Any = None

# Retry policy for service construction; see get_service().
_SERVICE_RETRIES = 4
_SERVICE_BACKOFF_S = 3.0

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
            "qimp.runtime.ibm requires `pip install qimp-mi[ibm]` (qiskit-ibm-runtime)"
        )

    # Service construction resolves the account's cloud instances against
    # IBM's global catalog, a network call that fails transiently often
    # enough to kill an unattended multi-hour submission loop. Observed
    # failures include DNS resolution errors and SSL EOFs, both of which
    # surface as "check that you are using a valid API token" even when the
    # token is fine. Retry with backoff before believing that message.
    last: Exception | None = None
    for attempt in range(_SERVICE_RETRIES):
        try:
            _SERVICE = qrs_cls(instance=instance) if instance else qrs_cls()
            return _SERVICE
        except Exception as exc:
            last = exc
            if attempt + 1 < _SERVICE_RETRIES:
                delay = _SERVICE_BACKOFF_S * (2**attempt)
                logger.warning(
                    "QiskitRuntimeService init failed (attempt %d/%d), retrying in %.0fs: %s",
                    attempt + 1,
                    _SERVICE_RETRIES,
                    delay,
                    exc,
                )
                time.sleep(delay)

    raise RuntimeError(
        "Could not initialise QiskitRuntimeService after "
        f"{_SERVICE_RETRIES} attempts — is ~/.qiskit/qiskit-ibm.json present "
        "and the token valid? Note that transient network failures against "
        "IBM's global catalog also surface as a token error. "
        f"Underlying error: {last}"
    ) from last


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
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=min_qubits)

    if backend.num_qubits < min_qubits:
        raise ValueError(
            f"Backend {backend.name} has {backend.num_qubits} qubits, needs >= {min_qubits}"
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


def _validate_zne_args(scales: Sequence[float], method: str) -> tuple[float, ...]:
    if method not in ZNE_METHODS:
        raise ValueError(f"unknown ZNE method {method!r}; expected one of {ZNE_METHODS}")
    out = tuple(float(s) for s in scales)
    if len(out) < 2 or len(set(out)) != len(out) or min(out) < 1.0:
        raise ValueError(f"ZNE scales must be >= 2 distinct values >= 1, got {list(scales)}")
    return out


def _fold_for_zne(
    transpiled: QuantumCircuit, scales: tuple[float, ...], target: Any | None
) -> tuple[list[QuantumCircuit], list[float]]:
    """Fold an ISA circuit at each scale; return circuits and achieved scales."""
    folded = [fold_gates(transpiled, s, target=target) for s in scales]
    achieved = [float(c.metadata["zne_fold_scale"]) for c in folded]
    return folded, achieved


def _two_q_gate_count(qc: QuantumCircuit) -> int:
    """Two-qubit gates in ``qc``, barriers excluded."""
    return sum(
        1
        for instr in qc.data
        if instr.operation.num_qubits >= 2 and instr.operation.name != "barrier"
    )


def _zne_summary(
    scales: tuple[float, ...],
    achieved: list[float],
    method: str,
    folded: list[QuantumCircuit],
    raw_counts: list[dict[str, int]],
) -> dict[str, Any]:
    return {
        "zne_scales": list(scales),
        "zne_achieved_scales": achieved,
        "zne_method": method,
        "zne_depths": [c.depth() for c in folded],
        "zne_two_q_gate_counts": [_two_q_gate_count(c) for c in folded],
        "zne_raw_counts": raw_counts,
    }


def aer_noisy_run(
    qc: QuantumCircuit,
    *,
    backend: Any,
    shots: int = 4096,
    zne_scales: Sequence[float] | None = None,
    zne_method: str = "richardson",
    seed: int | None = None,
) -> dict[str, int]:
    """Run ``qc`` on an Aer simulator wired with ``backend``'s noise model.

    Uses :func:`AerSimulator.from_backend` which inherits the device's
    noise + coupling map + basis gates. Adds ``measure_all()`` if the
    circuit has no measurements (reuses ``qimp.testing._ensure_measured``).

    When ``zne_scales`` is given (e.g. ``(1, 3, 5)``), the transpiled
    circuit is folded at each scale with :func:`fold_gates`, all folded
    circuits are simulated with ``shots`` shots each, and the
    zero-noise extrapolated counts (:func:`zne_extrapolate` with
    ``zne_method``) are returned instead of the raw counts. ``seed`` fixes
    both the transpiler and the simulator seed.
    """
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    from qimp.testing import _ensure_measured

    sim = AerSimulator.from_backend(backend)
    measured = _ensure_measured(qc)
    transpiled = transpile(measured, sim, seed_transpiler=seed)
    if zne_scales is None:
        result = sim.run(transpiled, shots=shots, seed_simulator=seed).result()
        return dict(result.get_counts())

    scales = _validate_zne_args(zne_scales, zne_method)
    folded, achieved = _fold_for_zne(transpiled, scales, sim.target)
    result = sim.run(folded, shots=shots, seed_simulator=seed).result()
    raw = [dict(result.get_counts(i)) for i in range(len(folded))]
    return zne_extrapolate(raw, achieved, method=zne_method, shots=shots)


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
    dd_sequence: str = "XY4",
    optimization_level: int = 3,
    timeout_s: float = 1200.0,
    zne_scales: Sequence[float] = DEFAULT_ZNE_SCALES,
    zne_method: str = "richardson",
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
        ``'trex+dd'`` enables twirled readout error extinction and
        dynamical decoupling (default). ``'trex'`` enables only TREX,
        ``'dd'`` enables only DD, ``'none'`` disables all mitigation.
        ``'zne'`` adds counts-side zero-noise extrapolation on top of
        TREX+DD: the transpiled ISA circuit is folded at each of
        ``zne_scales`` with :func:`fold_gates`, the folded circuits are
        submitted as separate pubs of one SamplerV2 job, and each
        bitstring probability is extrapolated to zero noise with
        :func:`zne_extrapolate`. The returned counts are the extrapolated
        counts rescaled to ``shots``; ``transpiled_qc`` is the unfolded
        circuit. The five modes together support ablation studies of
        which layer contributes the recovered signal.
    dd_sequence:
        Decoupling sequence used when ``mitigation`` includes DD. ``'XY4'``
        (default) is the sequence used throughout the paper's campaign.
        ``'XpXm'`` and ``'XX'`` are the matched-pulse-count pair: both
        insert two pulses per idle window with identical timing, but
        XpXm's +/- alternation cancels pulse-amplitude error to first
        order while XX accumulates it, so the contrast between them
        separates a coherent pulse-error mechanism from an incoherent one
        (§7.4). These three are what SamplerV2 exposes.
    optimization_level:
        Transpiler optimisation level (0–3). Defaults to 3.
    timeout_s:
        ``timeout_s`` is currently advisory; recovery of long-running jobs is
        handled by the caller (the CLI sweep script in Task 15 persists
        ``job_id`` before awaiting ``job.result()``).
    zne_scales:
        Noise scales for ``mitigation='zne'`` (default ``(1, 3, 5)``). Odd
        integers give full global folds; other values add a partial fold
        of the last gates. Ignored by the other modes.
    zne_method:
        Extrapolation model for ``mitigation='zne'``: ``'richardson'``
        (default), ``'linear'`` or ``'exponential'``.

    Returns
    -------
    tuple
        ``(counts, transpiled_qc, job_id, transpile_summary)``. With
        ``mitigation='zne'`` the summary also holds ``zne_scales``,
        ``zne_achieved_scales`` (used as the abscissa), ``zne_method``,
        per-scale ``zne_depths`` and ``zne_two_q_gate_counts``, and the raw
        per-scale counts ``zne_raw_counts``, so callers that persist the
        summary keep everything needed to redo the fit.
    """
    _allowed_mitigation = ("trex+dd", "trex", "dd", "none", "zne")
    if mitigation not in _allowed_mitigation:
        raise ValueError(
            f"unknown mitigation {mitigation!r}; expected one of {_allowed_mitigation}"
        )
    if mitigation == "zne":
        scales = _validate_zne_args(zne_scales, zne_method)

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
    transpiled = transpile(measured, target=_target, optimization_level=optimization_level)
    summary = _transpile_summary(transpiled)

    Options = _sampler_options_cls()
    options = Options()
    if mitigation in ("trex+dd", "dd", "zne"):
        options.dynamical_decoupling.enable = True
        options.dynamical_decoupling.sequence_type = dd_sequence
    if mitigation in ("trex+dd", "trex", "zne"):
        options.twirling.enable_measure = True

    Sampler = _sampler_v2_cls()
    sampler = Sampler(mode=backend, options=options)
    if mitigation == "zne":
        # SamplerV2 has no ZNE option (qiskit-ibm-runtime >= 0.47 dropped
        # resilience_level from SamplerOptions), so fold by hand. Folding
        # happens on the ISA circuit, with barriers between copies and the
        # inverse gates translated into the target basis, so nothing
        # downstream can cancel U^dag U.
        folded, achieved = _fold_for_zne(transpiled, scales, _target)
        job = sampler.run(folded, shots=shots)
        job_id = job.job_id()
        logger.info("Submitted ZNE job %s on %s (scales %s)", job_id, backend.name, list(scales))
        result = job.result()
        raw = [dict(getattr(result[i].data, creg_name).get_counts()) for i in range(len(folded))]
        counts = zne_extrapolate(raw, achieved, method=zne_method, shots=shots)
        summary.update(_zne_summary(scales, achieved, zne_method, folded, raw))
        return counts, transpiled, job_id, summary

    job = sampler.run([transpiled], shots=shots)
    job_id = job.job_id()
    logger.info("Submitted job %s on %s", job_id, backend.name)

    result = job.result()
    counts = dict(getattr(result[0].data, creg_name).get_counts())
    return counts, transpiled, job_id, summary
