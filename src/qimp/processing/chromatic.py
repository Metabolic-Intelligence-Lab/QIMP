"""Chromatic transformations: color complement, color change, halving, classification.

Reference: docs/tesi.pdf §2.4, §3.1.1, §3.1.2
"""

from __future__ import annotations

# TODO(Fase 4):
#   - color_compl(qc, encoding, registers) -> QuantumCircuit       # FRQI / NEQR
#   - color_change(qc, encoding, registers, value) -> QuantumCircuit  # FRQI
#   - half_int(qc, registers) -> QuantumCircuit                    # NEQR (shift right)
#   - classify_compl(qc, registers, threshold) -> QuantumCircuit   # NEQR binary thresholding
