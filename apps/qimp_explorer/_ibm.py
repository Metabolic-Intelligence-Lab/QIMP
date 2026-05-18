"""IBM Quantum integration helpers — QASM export + Runtime submission.

Kept Streamlit-free so the helpers are unit-testable. The Runtime path is
gated by an optional import of ``qiskit-ibm-runtime`` (the ``[ibm]`` extra);
the QASM export path always works.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


__all__ = [
    "circuit_to_qasm3",
    "have_ibm_runtime",
    "list_ibm_backends",
    "run_on_ibm",
]


def circuit_to_qasm3(qc: QuantumCircuit) -> str:
    """Serialize a circuit to OpenQASM 3.

    The Composer at https://quantum.ibm.com/composer imports QASM 3 directly.
    Falls back to QASM 2 if Qiskit's qasm3 module isn't available (unlikely
    on Qiskit ≥ 1.0).
    """
    try:
        from qiskit import qasm3

        return qasm3.dumps(qc)
    except Exception:
        # Last-ditch fallback (legacy Qiskit).
        return qc.qasm()


def have_ibm_runtime() -> bool:
    """True if ``qiskit_ibm_runtime`` is importable."""
    try:
        import qiskit_ibm_runtime  # noqa: F401
    except ImportError:
        return False
    return True


def _make_service(token: str | None, channel: str) -> Any:
    """Instantiate ``QiskitRuntimeService`` from either a token or the saved account."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    if token:
        return QiskitRuntimeService(channel=channel, token=token)
    return QiskitRuntimeService(channel=channel)


def list_ibm_backends(
    *,
    token: str | None = None,
    channel: str = "ibm_quantum_platform",
    operational_only: bool = True,
    simulators: bool = False,
) -> list[str]:
    """Return the names of available IBM backends.

    Parameters
    ----------
    token
        Personal API token. If None, uses the saved account
        (``QiskitRuntimeService.save_account(...)``).
    channel
        ``"ibm_quantum_platform"`` (current) or ``"ibm_quantum"`` (legacy).
    operational_only
        Skip backends not currently accepting jobs.
    simulators
        Include cloud simulators in the result.

    Raises
    ------
    ImportError
        If ``qiskit-ibm-runtime`` is not installed.
    """
    if not have_ibm_runtime():
        raise ImportError("qiskit-ibm-runtime not installed; `pip install qimp-mi[ibm]`")
    service = _make_service(token, channel)
    backends = service.backends(operational=operational_only, simulator=simulators)
    return [b.name for b in backends]


def run_on_ibm(
    qc: QuantumCircuit,
    *,
    backend_name: str | None = None,
    token: str | None = None,
    channel: str = "ibm_quantum_platform",
    shots: int = 4096,
) -> dict[str, Any]:
    """Submit a circuit to an IBM backend via the Sampler primitive.

    Returns
    -------
    dict
        ``{"job_id": str, "backend": str, "status": str, "counts": dict | None}``.
        ``counts`` is ``None`` while the job is still running; the caller can
        re-fetch with :func:`retrieve_ibm_job`.

    Raises
    ------
    ImportError
        If ``qiskit-ibm-runtime`` is not installed.
    """
    if not have_ibm_runtime():
        raise ImportError("qiskit-ibm-runtime not installed; `pip install qimp-mi[ibm]`")
    from qiskit import transpile
    from qiskit_ibm_runtime import Sampler  # type: ignore[attr-defined]

    service = _make_service(token, channel)
    backend = (
        service.backend(backend_name)
        if backend_name
        else service.least_busy(operational=True, simulator=False)
    )

    # Ensure measurements exist.
    measured = qc
    if not any(instr.operation.name == "measure" for instr in qc.data):
        measured = qc.copy()
        measured.measure_all()

    transpiled = transpile(measured, backend=backend, optimization_level=1)
    sampler = Sampler(mode=backend)
    job = sampler.run([transpiled], shots=shots)
    return {
        "job_id": job.job_id(),
        "backend": backend.name,
        "status": str(job.status()),
        "counts": None,  # caller polls later
    }


def retrieve_ibm_job(
    job_id: str,
    *,
    token: str | None = None,
    channel: str = "ibm_quantum_platform",
) -> dict[str, Any]:
    """Poll the status of a previously submitted job, returning counts when done.

    The returned dict matches :func:`run_on_ibm` but with the latest
    ``status`` and the ``counts`` populated once the job has completed.
    """
    if not have_ibm_runtime():
        raise ImportError("qiskit-ibm-runtime not installed")
    service = _make_service(token, channel)
    job = service.job(job_id)
    status = str(job.status())
    counts: dict[str, int] | None = None
    if str(status).upper() in ("DONE", "COMPLETED"):
        try:
            result = job.result()
            counts = dict(result[0].data.meas.get_counts())
        except Exception as exc:
            return {"job_id": job_id, "status": f"error: {exc}", "counts": None}
    return {
        "job_id": job_id,
        "backend": getattr(job.backend(), "name", "unknown"),
        "status": status,
        "counts": counts,
    }
