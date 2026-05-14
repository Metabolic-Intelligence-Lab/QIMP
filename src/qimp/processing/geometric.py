"""Geometric transformations: flip, coord-swap, rotation, restricted variants, position shift.

Each function operates on an already-encoded circuit and is parameterized by the
encoding type (FRQI / NEQR / QPIE) and the qubit registers, not by hard-coded indices.

Reference: docs/tesi.pdf §2.3, §3.1
"""

from __future__ import annotations

# TODO(Fase 4):
#   - axis_flip(qc, n, axis, encoding) -> QuantumCircuit
#   - coord_swap(qc, n, encoding) -> QuantumCircuit
#   - ort_rotation(qc, n, angle, encoding) -> QuantumCircuit  # angle in {90, 180, 270}
#   - restr_flip(qc, n, region, axis, encoding) -> QuantumCircuit
#   - restr_coord(qc, n, region, encoding) -> QuantumCircuit
#   - pos_shift(qc, n, axis, magnitude, direction, encoding) -> QuantumCircuit
# `encoding` is a string literal: "frqi" | "neqr" | "qpie".
