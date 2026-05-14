"""Green-Purple (GP) ratio computation via quantum circuits.

Application module (not part of the core tesi library). Implements (I_G - I_R) /
(I_G + I_R) on stacked microscopy channels using a variationally optimized
quantum sub-circuit. Used by the Metabolic Intelligence Lab on polarization
images.

Migrated from legacy/scripts/gp_quantum_4.py + quantize_analyze_quantum_gp_v3.py.
"""

from __future__ import annotations

# TODO(Fase 3): migrate from legacy/scripts/gp_quantum_4.py
#   - apply_gp_function(qc, n, m, params) -> QuantumCircuit
#   - combined_objective(mse, psnr, tv, alpha, beta, gamma) -> float
#   - optimize_gp(images, config: ProcessingConfig) -> OptimizationResult
