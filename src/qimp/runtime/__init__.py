"""Runtime infrastructure: memory pooling, simulator manager, circuit caching, monitoring."""

# Imported for their side effects on `qimp.runtime.{ibm,circuits}` — public
# names from these modules are re-exported by later tasks.
from qimp.runtime import circuits as _circuits  # noqa: F401
from qimp.runtime import ibm as _ibm  # noqa: F401
from qimp.runtime.arithmetic_gates import cyclic_decrement, cyclic_increment
from qimp.runtime.caching import adaptive_shots, base_frqi_circuit, clear_circuit_cache
from qimp.runtime.memory_pool import MemoryPool, clear_memory_pool, get_memory_pool
from qimp.runtime.monitoring import performance_monitor
from qimp.runtime.simulator import SimulatorManager, get_simulator

__all__ = [
    "MemoryPool",
    "SimulatorManager",
    "adaptive_shots",
    "base_frqi_circuit",
    "clear_circuit_cache",
    "clear_memory_pool",
    "cyclic_decrement",
    "cyclic_increment",
    "get_memory_pool",
    "get_simulator",
    "performance_monitor",
]
