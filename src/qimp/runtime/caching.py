"""LRU-cached circuit construction wrappers.

The base FRQI/NEQR/QPIE circuits depend only on the qubit counts, so it pays to
cache the empty register layout and clone it for each new image.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from qiskit import QuantumCircuit

logger = logging.getLogger(__name__)

__all__ = ["adaptive_shots", "base_frqi_circuit", "clear_circuit_cache"]


@lru_cache(maxsize=64)
def base_frqi_circuit(n: int, m: int = 0) -> QuantumCircuit:
    """Return an empty FRQI scaffold for a 2^n × 2^n image and `m` selection qubits.

    Layout: 2n position qubits | m image-selection qubits | 1 color qubit.
    Hadamards are pre-applied to the 2n position qubits.

    The returned circuit is *shared*: callers must `.copy()` before mutating.
    Scales with n and m without bound (caller's RAM is the only limit).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if m < 0:
        raise ValueError("m must be >= 0")
    num_qubits = 2 * n + m + 1
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(2 * n))
    qc.barrier()
    return qc


def adaptive_shots(
    circuit_depth: int,
    convergence_history: list[float] | None = None,
    base_shots: int = 10_000,
    min_shots: int = 1_000,
    max_shots: int = 100_000,
) -> int:
    """Pick a shot count for the next variational iteration.

    Heuristic:
    - If the last 3 objective values are within stddev < 0.01, halve `base_shots`.
    - Otherwise scale `base_shots` linearly with `circuit_depth / 50`, capped at 2x.
    """
    if (
        convergence_history
        and len(convergence_history) >= 3
        and float(np.std(convergence_history[-3:])) < 0.01
    ):
        return max(base_shots // 2, min_shots)
    scaled = int(base_shots * min(2.0, 1.0 + circuit_depth / 50.0))
    return max(min_shots, min(scaled, max_shots))


def clear_circuit_cache() -> None:
    """Empty the lru cache on `base_frqi_circuit`."""
    base_frqi_circuit.cache_clear()
