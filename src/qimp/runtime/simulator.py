"""Singleton-per-device Aer simulator manager.

Probes for GPU support on first use and caches the result for the lifetime of
the process. Subsequent calls reuse the same `AerSimulator` instance.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from qiskit_aer import AerSimulator

__all__ = ["SimulatorManager", "get_simulator"]


class SimulatorManager:
    """One AerSimulator per (device, method) tuple, lazily instantiated."""

    _simulators: ClassVar[dict[tuple[str, str], Any]] = {}
    _gpu_available: ClassVar[bool | None] = None

    @classmethod
    def is_gpu_available(cls) -> bool:
        if cls._gpu_available is None:
            cls._gpu_available = cls._probe_gpu()
        return cls._gpu_available

    @classmethod
    def _probe_gpu(cls) -> bool:
        try:
            from qiskit import QuantumCircuit
            from qiskit_aer import AerSimulator

            sim = AerSimulator(device="GPU")
            probe = QuantumCircuit(2)
            probe.h(0)
            probe.measure_all()
            sim.run(probe, shots=1).result()
        except Exception as exc:
            logger.info("GPU simulator unavailable, falling back to CPU: %s", exc)
            return False
        logger.info("GPU simulator available")
        return True

    @classmethod
    def get(cls, device: str | None = None, method: str = "automatic") -> AerSimulator:
        """Return a cached AerSimulator for the given (device, method) pair.

        Parameters
        ----------
        device
            "CPU", "GPU", or None to auto-select (GPU if available, else CPU).
        method
            Qiskit Aer simulation method ("automatic", "statevector",
            "density_matrix", "stabilizer", "extended_stabilizer", "matrix_product_state",
            "tensor_network").
        """
        from qiskit_aer import AerSimulator

        if device is None:
            device = "GPU" if cls.is_gpu_available() else "CPU"
        key = (device, method)
        if key not in cls._simulators:
            cls._simulators[key] = AerSimulator(device=device, method=method)
            logger.debug("Created AerSimulator(device=%s, method=%s)", device, method)
        return cls._simulators[key]

    @classmethod
    def reset(cls) -> None:
        """Clear all cached simulators and the GPU probe result."""
        cls._simulators.clear()
        cls._gpu_available = None


def get_simulator(device: str | None = None, method: str = "automatic") -> AerSimulator:
    """Module-level shortcut for `SimulatorManager.get(...)`."""
    return SimulatorManager.get(device=device, method=method)
