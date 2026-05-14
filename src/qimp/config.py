"""Centralized configuration for QIMP."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Runtime configuration shared by encoders, processors, and the simulator manager.

    All qubit counts are user-provided; the library imposes no upper bound beyond
    what the chosen simulator can hold in memory.

    Attributes
    ----------
    n_spatial_qubits
        Number of qubits per spatial axis. Image size is 2^n × 2^n.
    q_intensity_qubits
        Number of qubits used to encode pixel intensity (NEQR / NCQI).
    m_image_qubits
        Number of qubits selecting between stacked images (multi-image FRQI).
    base_shots
        Default measurement shots for ideal/noisy simulations.
    optimization_method
        scipy.optimize method name for variational pipelines.
    max_iterations
        Optimizer iteration cap.
    convergence_threshold
        Stopping tolerance for variational objectives.
    gpu_enabled
        Allow the simulator manager to select a GPU backend if available.
    memory_pool_size
        Number of pre-allocated image buffers in the runtime memory pool.
    """

    n_spatial_qubits: int = 4
    q_intensity_qubits: int = 8
    m_image_qubits: int = 1
    base_shots: int = 10_000
    optimization_method: str = "COBYLA"
    max_iterations: int = 50
    convergence_threshold: float = 1e-6
    gpu_enabled: bool = True
    memory_pool_size: int = 100

    def __post_init__(self) -> None:
        if self.n_spatial_qubits < 1:
            raise ValueError("n_spatial_qubits must be >= 1")
        if self.q_intensity_qubits < 1:
            raise ValueError("q_intensity_qubits must be >= 1")
        if self.m_image_qubits < 0:
            raise ValueError("m_image_qubits must be >= 0")
        if self.base_shots < 100:
            logger.warning("Very low shot count (%d) — results will be noisy", self.base_shots)
