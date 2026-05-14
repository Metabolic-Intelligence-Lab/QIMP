"""Tests for qimp.runtime."""

from __future__ import annotations

import numpy as np
import pytest

from qimp.runtime import (
    MemoryPool,
    adaptive_shots,
    base_frqi_circuit,
    clear_circuit_cache,
    clear_memory_pool,
    get_memory_pool,
    performance_monitor,
)


def test_memory_pool_shapes() -> None:
    pool = MemoryPool(max_buffers=4, image_size=8)
    img_buf = pool.get_image_buffer()
    ang_buf = pool.get_angle_buffer()
    assert img_buf.shape == (8, 8)
    assert ang_buf.shape == (64,)


def test_memory_pool_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        MemoryPool(max_buffers=0, image_size=8)
    with pytest.raises(ValueError):
        MemoryPool(max_buffers=4, image_size=0)


def test_memory_pool_reuses_buffers() -> None:
    pool = MemoryPool(max_buffers=2, image_size=4)
    a = pool.get_angle_buffer()
    pool.get_angle_buffer()
    c = pool.get_angle_buffer()  # wraps around
    assert np.shares_memory(a, c)  # same underlying storage reused


def test_memory_pool_buffers_are_zeroed() -> None:
    pool = MemoryPool(max_buffers=1, image_size=4)
    buf = pool.get_image_buffer()
    buf[:] = 7.0
    buf2 = pool.get_image_buffer()
    assert np.all(buf2 == 0.0)


def test_global_pool_is_singleton_per_size() -> None:
    clear_memory_pool()
    p1 = get_memory_pool(image_size=8)
    p2 = get_memory_pool(image_size=8)
    assert p1 is p2
    p3 = get_memory_pool(image_size=16)
    assert p1 is not p3
    clear_memory_pool()


def test_base_frqi_circuit_qubit_count(n_qubits: int) -> None:
    qc = base_frqi_circuit(n=n_qubits, m=0)
    assert qc.num_qubits == 2 * n_qubits + 1


@pytest.mark.parametrize("m", [0, 1, 2, 3])
def test_base_frqi_circuit_scales_with_m(m: int) -> None:
    qc = base_frqi_circuit(n=2, m=m)
    assert qc.num_qubits == 2 * 2 + m + 1


def test_base_frqi_circuit_is_cached() -> None:
    clear_circuit_cache()
    a = base_frqi_circuit(3, 1)
    b = base_frqi_circuit(3, 1)
    assert a is b


def test_base_frqi_circuit_rejects_invalid_n() -> None:
    with pytest.raises(ValueError):
        base_frqi_circuit(n=0)


def test_adaptive_shots_converged_history_halves() -> None:
    out = adaptive_shots(circuit_depth=10, convergence_history=[1.0, 1.0, 1.0], base_shots=10_000)
    assert out == 5_000


def test_adaptive_shots_unconverged_scales_with_depth() -> None:
    out = adaptive_shots(circuit_depth=100, convergence_history=None, base_shots=10_000)
    assert out == 20_000  # capped at 2x


def test_adaptive_shots_respects_caps() -> None:
    assert adaptive_shots(0, base_shots=100, min_shots=1_000, max_shots=100_000) == 1_000
    assert adaptive_shots(10_000, base_shots=200_000, min_shots=1_000, max_shots=50_000) == 50_000


def test_performance_monitor_runs_function() -> None:
    @performance_monitor
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5


def test_memory_pool_image_buffer_advances_cursor() -> None:
    """Regression: get_image_buffer used to NOT advance the cursor, so
    consecutive calls returned the same buffer.
    """
    pool = MemoryPool(max_buffers=3, image_size=4)
    a = pool.get_image_buffer()
    a[0, 0] = 7.0
    b = pool.get_image_buffer()
    assert not np.shares_memory(a, b)
    # a's data is untouched until we wrap around to its slot again.
    assert a[0, 0] == 7.0


def test_cyclic_increment_decrement_are_inverses() -> None:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    from qimp.runtime import cyclic_decrement, cyclic_increment

    qc = QuantumCircuit(3)
    qubits = [0, 1, 2]
    cyclic_increment(qc, qubits)
    cyclic_decrement(qc, qubits)
    sv = Statevector.from_instruction(qc)
    # |000⟩ + 1 - 1 should return to |000⟩.
    assert abs(sv.data[0] - 1.0) < 1e-10


def test_cyclic_increment_rejects_empty() -> None:
    import pytest
    from qiskit import QuantumCircuit

    from qimp.runtime import cyclic_increment

    qc = QuantumCircuit(1)
    with pytest.raises(ValueError, match="empty"):
        cyclic_increment(qc, [])
