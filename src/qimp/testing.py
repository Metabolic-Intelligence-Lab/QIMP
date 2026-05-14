"""Test harnesses: ideal/noisy simulation and IBM Quantum hardware execution.

Reference: docs/tesi.pdf §3.1.5
"""

from __future__ import annotations

# TODO(Fase 3): build on top of runtime.simulator.SimulatorManager
#   - ideal_simulation(qc, shots) -> Counts
#   - noisy_simulation(qc, shots, noise_model=None) -> Counts
#   - device_test(qc, shots, backend_name=None) -> Counts  # least-busy IBM device
