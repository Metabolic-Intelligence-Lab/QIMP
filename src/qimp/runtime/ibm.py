"""IBM Quantum Runtime helpers — service singleton, backend resolution,
hardware execution via SamplerV2 with TREX + DD mitigation, Aer
noise-model fallback, and per-run artifact persistence.

All IBM-Runtime imports are local to this module: the rest of the
library imports only the typed helpers exposed here.
"""

from __future__ import annotations

import logging
from typing import Any

__all__ = ["get_service", "list_backends", "pick_backend"]

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
