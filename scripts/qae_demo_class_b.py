"""Quantum Amplitude Estimation demo on Class-B autonomous ratio circuit.

This is a skeleton / outline of the Stage-E quantum-advantage
demonstration: layer Iterative Amplitude Estimation (IAE) on top of
the Class-B autonomous ratio state-prep, then estimate the fraction
of pixels in the image whose I_a/I_b ratio exceeds a chosen threshold.

The theoretical claim
---------------------
Classical Monte Carlo over N pixels needs O(1/ε²) samples to estimate
the fraction f = (# pixels with R(p) > τ) / N_pixels to additive
error ε. QAE with the state-prep ``A`` as oracle needs O(1/ε) queries
to ``A`` (and the same to ``A†``) for the same error. The quadratic
speed-up in sample complexity is provable for any expectation value
``E[f(R)]`` whose state-prep is a known unitary.

The state-prep
--------------
``A`` is the composition

    A = (mark-good oracle) ∘ class_b_ratio

The mark-good oracle is a multi-controlled X gate that flips a single
ancilla qubit (call it ``good_qubit``) when the quotient register
holds a value > τ (a classical threshold). Implementing the
"quotient > τ" predicate as a controlled X on ``good_qubit`` is
straightforward via the standard NEQR comparator pattern (see
``neqr_comparator`` in ``qimp.processing.arithmetic``).

The amplitude
-------------
After A applied to |0⟩^⊗N, the state is

    |Ψ⟩ = (1/√P) Σ_p |p⟩ |I_a(p)⟩ |I_b(p)⟩ |R(p)⟩ ⊗ (|0⟩ if R(p) ≤ τ else |1⟩)_good

The marginal probability of measuring good_qubit = 1 is

    a = f := |{p : R(p) > τ}| / N_pixels

QAE applies the Grover operator Q = A · S_0 · A^† · S_good a varying
number of times (powers-of-2 in IAE) and extracts ``a`` from the
interference pattern. Requires ``class_b_ratio_inv`` (the cleaned-up
inverse via the Stage-E inverses) so that ``A^†`` is well-defined as a
unitary on the same qubit space.

Practical concerns
------------------
- ``class_b_ratio`` at n=1, q=2 already takes ~24 qubits. Adding the
  mark-good oracle costs 1 qubit. Adding the ``q_div_restoring_inv``
  ancilla budget is "free" (same carries are reused).
- The Grover operator's depth is ~2x the state-prep depth, and IAE
  applies Q powers up to k = O(log(1/ε)) — for ε=1%, k ≈ 7, so up
  to 128 applications of Q. Compiled depth quickly hits 10⁴–10⁵.
- Statevector simulation is feasible at ~24 qubits but slow given
  the depth. The recommended simulator is AerSimulator(method='mps').
  Real-hardware execution is NOT in scope (the depth would put it
  far past the gate-noise floor of current Heron-r2 devices).

Status
------
**SKELETON / NOT YET IMPLEMENTED.** This file documents the architecture
and dependencies. Concrete next steps:

  1. Implement ``class_b_ratio_inv`` in ``qimp.processing.ratiometric_circuit``
     (depends on ``q_div_restoring_inv``, already provided by Stage E.a).
  2. Implement ``mark_good_oracle`` (NEQR-comparator-style: flip
     ``good_qubit`` iff ``quotient > τ``). Mirror it with
     ``mark_good_oracle_inv`` for QAE composability.
  3. Wire up Qiskit's ``IterativeAmplitudeEstimation`` with the
     state-prep above (or implement IAE manually if the qiskit-algorithms
     API is unstable).
  4. Run on AerSimulator(method='mps') at n=1, q=2 with several
     ε levels; compare against classical Monte Carlo on the same image.
     Plot estimate error vs query budget on a log-log scale; the
     QAE curve should show the O(1/ε) slope.

Anticipated runtime: full sweep ~hours on Aer-MPS at n=1, q=2.
This is a multi-session effort and should probably be tracked as its
own paper line.
"""

from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    print(__doc__)
    print()
    print("This script is a skeleton — not yet runnable.")
    print()
    print("Required follow-up primitives:")
    print("  - qimp.processing.ratiometric_circuit.class_b_ratio_inv")
    print("  - qimp.processing.ratiometric_circuit.mark_good_oracle")
    print("  - qimp.processing.ratiometric_circuit.mark_good_oracle_inv")
    print()
    print("Required Qiskit dependency:")
    print("  - qiskit_algorithms (for IterativeAmplitudeEstimation)")
    print("  - qiskit_aer with MPS support")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
