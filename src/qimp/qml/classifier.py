"""Variational quantum image classifier (planned for v0.2).

The legacy `qml_and_qimp.py` script in `legacy/scripts/` is actually a batch
image-processor (GP computation over a folder), not a classifier. A proper
variational classifier on FRQI/NEQR/QPIE features will be added in Fase 4
using `qiskit-machine-learning` (see the `[qml]` optional extra).
"""

from __future__ import annotations

# TODO(Fase 4): VQC on FRQI features. Sketch:
#   1. Encode each training image with FrqiEncoder → list[QuantumCircuit]
#   2. Append a variational ansatz (qiskit.circuit.library.RealAmplitudes)
#   3. Train with qiskit_machine_learning.algorithms.VQC over n, q parametric.
