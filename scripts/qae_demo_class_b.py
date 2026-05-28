"""Quantum Amplitude Estimation demo on the Class-B autonomous ratio circuit.

The state-prep ``A`` is the composition

    A = mark_good_oracle ∘ class_b_ratio

so that after applying A to |0⟩^⊗N the marginal probability of measuring
the good_qubit in state |1⟩ equals the fraction of pixels in the image
whose I_a/I_b ratio strictly exceeds a classical threshold τ.

The script demonstrates a manual Maximum-Likelihood QAE (Suzuki et al.,
2020): sample at Grover-power k ∈ {0, 1, 2}, fit the underlying
amplitude θ from the per-k empirical good-qubit probabilities. We then
contrast the QAE estimate against a classical Monte Carlo baseline of
the same total query budget.

Runs on ``AerSimulator(method='mps')`` since the full state-prep has
~30 qubits (out of statevector range on a 16 GB laptop). Expected
runtime: 10–30 minutes per IQAE pass at n=1, q=2 on this laptop class;
the heavy depth (Grover iterations multiply the already-deep state-prep)
is why we cap k at 2 for this POC.

USAGE
    python scripts/qae_demo_class_b.py [--shots N] [--threshold T]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qimp.processing.ratiometric_circuit import (  # noqa: E402
    class_b_ratio,
    class_b_ratio_inv,
    mark_good_oracle,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# State-prep A and its inverse
# ---------------------------------------------------------------------------


def build_A(
    image_a: np.ndarray,
    image_b: np.ndarray,
    q: int,
    threshold: int,
) -> tuple[QuantumCircuit, dict, dict]:
    """Build the QAE state-prep A = class_b_ratio + mark_good_oracle +
    divzero-aware correction.

    For real microscopy data, some pixels naturally have I_b = 0 after
    quantisation. The divider's behaviour at divisor = 0 is well-defined
    (the quotient register fills with 1's, see q_div_restoring) but the
    resulting "good" qubit, computed by mark_good_oracle on the
    garbage quotient, will then be set to 1 for any positive threshold.
    To preserve the QAE semantics, we apply a final correction:

        good_corrected := good AND NOT div_zero_flag

    implemented as an X-CCX-X sandwich on (div_zero_flag, good, good_corrected)
    with good_corrected as a fresh |0⟩ ancilla. The whole pipeline
    (class_b_ratio → mark_good_oracle → correction) is self-inverse
    modulo the divzero handling, which the inverse function mirrors.

    Returns (qc_full, b_layout, extra_layout). The QAE algorithm should
    measure ``extra_layout["good_corrected"]`` rather than the raw
    "good" qubit.
    """
    qc, b_layout = class_b_ratio(image_a, image_b, q=q)
    start = qc.num_qubits
    q_w = q + 1
    # Allocate mark-good ancillae + correction ancilla
    good_reg = QuantumRegister(1, "good")
    thr_reg = QuantumRegister(q_w, "thr")
    sub_c_reg = QuantumRegister(q_w + 1, "sub_c")
    val_pad_reg = QuantumRegister(1, "val_pad")
    good_corr_reg = QuantumRegister(1, "good_corrected")
    qc.add_register(good_reg, thr_reg, sub_c_reg, val_pad_reg, good_corr_reg)
    extra_layout = {
        "good": start,
        "thr": list(range(start + 1, start + 1 + q_w)),
        "sub_c": list(range(start + 1 + q_w, start + 1 + q_w + q_w + 1)),
        "val_pad": start + 1 + q_w + q_w + 1,
        "good_corrected": start + 1 + q_w + q_w + 2,
    }
    # 1. mark_good_oracle sets good = (quotient > threshold)
    #    (possibly garbage when div_zero_flag = 1).
    mark_good_oracle(
        qc,
        value_qubits=b_layout["quotient"],
        threshold=threshold,
        good_qubit=extra_layout["good"],
        threshold_reg_qubits=extra_layout["thr"],
        sub_carry_qubits=extra_layout["sub_c"],
        value_pad_qubit=extra_layout["val_pad"],
    )
    # 2. divzero-aware correction: good_corrected = good AND NOT div_zero
    qc.x(b_layout["flag"])
    qc.ccx(b_layout["flag"], extra_layout["good"], extra_layout["good_corrected"])
    qc.x(b_layout["flag"])
    return qc, b_layout, extra_layout


def apply_A_inv(
    qc: QuantumCircuit,
    image_a: np.ndarray,
    image_b: np.ndarray,
    q: int,
    threshold: int,
    b_layout: dict,
    extra_layout: dict,
) -> None:
    """Apply A^†: reverse of build_A's body in reverse order.

    Forward order (in build_A):
      class_b_ratio → mark_good_oracle → X-CCX-X correction
    Inverse order:
      X-CCX-X correction (self-inverse) → mark_good_oracle (self-inverse)
      → class_b_ratio_inv.
    """
    qc.x(b_layout["flag"])
    qc.ccx(b_layout["flag"], extra_layout["good"], extra_layout["good_corrected"])
    qc.x(b_layout["flag"])
    mark_good_oracle(
        qc,
        value_qubits=b_layout["quotient"],
        threshold=threshold,
        good_qubit=extra_layout["good"],
        threshold_reg_qubits=extra_layout["thr"],
        sub_carry_qubits=extra_layout["sub_c"],
        value_pad_qubit=extra_layout["val_pad"],
    )
    class_b_ratio_inv(qc, image_a, image_b, q=q, layout=b_layout)


# ---------------------------------------------------------------------------
# Grover operator Q = A · S_0 · A^† · S_good
# ---------------------------------------------------------------------------


def apply_S_good(qc: QuantumCircuit, good_qubit: int) -> None:
    """Oracle phase flip: |good⟩ → -|good⟩."""
    qc.z(good_qubit)


def apply_S_0(qc: QuantumCircuit, all_qubits: list[int]) -> None:
    """Zero-state reflection: flip phase of |0…0⟩.

    Implementation: X on every qubit, multi-controlled Z on one of them
    (with the others as controls), X back.
    """
    for q in all_qubits:
        qc.x(q)
    # Multi-controlled Z on the last qubit, controlled by all others.
    # MCZ ≡ H · MCX · H on the target.
    target = all_qubits[-1]
    controls = all_qubits[:-1]
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)
    for q in all_qubits:
        qc.x(q)


def apply_A_forward(
    qc: QuantumCircuit,
    image_a: np.ndarray,
    image_b: np.ndarray,
    q: int,
    threshold: int,
    b_layout: dict,
    extra_layout: dict,
) -> None:
    """Re-apply A in place (the same gates as build_A, but on an existing
    circuit with the qubits already allocated). Used by the Grover
    operator's last step."""
    from qimp.processing.arithmetic import q_div_restoring
    from qimp.processing.ratiometric_circuit import dual_neqr_load
    dual_neqr_load(
        qc, image_a, image_b, q,
        position_qubits=b_layout["position"],
        intensity_a_qubits=b_layout["I_a"],
        intensity_b_qubits=b_layout["I_b"],
    )
    q_div_restoring(
        qc,
        dividend_qubits=b_layout["I_a"],
        divisor_qubits=b_layout["I_b"],
        quotient_qubits=b_layout["quotient"],
        work_qubits=b_layout["work"],
        divisor_pad_qubit=b_layout["pad"],
        c_qubits=b_layout["c"],
        div_zero_flag=b_layout["flag"],
    )
    mark_good_oracle(
        qc,
        value_qubits=b_layout["quotient"],
        threshold=threshold,
        good_qubit=extra_layout["good"],
        threshold_reg_qubits=extra_layout["thr"],
        sub_carry_qubits=extra_layout["sub_c"],
        value_pad_qubit=extra_layout["val_pad"],
    )
    qc.x(b_layout["flag"])
    qc.ccx(b_layout["flag"], extra_layout["good"], extra_layout["good_corrected"])
    qc.x(b_layout["flag"])


def apply_Q_once(
    qc: QuantumCircuit,
    image_a: np.ndarray,
    image_b: np.ndarray,
    q: int,
    threshold: int,
    b_layout: dict,
    extra_layout: dict,
) -> None:
    """Apply one Grover iteration Q = A · S_0 · A^† · S_good in place.

    The phase oracle ``S_good`` reflects about |good_corrected = 1⟩;
    the zero-state reflection acts on every qubit.
    """
    apply_S_good(qc, extra_layout["good_corrected"])
    apply_A_inv(qc, image_a, image_b, q, threshold, b_layout, extra_layout)
    apply_S_0(qc, list(range(qc.num_qubits)))
    apply_A_forward(qc, image_a, image_b, q, threshold, b_layout, extra_layout)


# ---------------------------------------------------------------------------
# Sampler: build (A · Q^k) circuit, run on MPS, return P(good = 1)
# ---------------------------------------------------------------------------


def measure_good_probability(
    image_a: np.ndarray,
    image_b: np.ndarray,
    q: int,
    threshold: int,
    grover_k: int,
    shots: int,
) -> float:
    """Build A · Q^k, measure good_qubit, return empirical P(good=1)."""
    qc, b_layout, extra_layout = build_A(image_a, image_b, q, threshold)
    for _ in range(grover_k):
        apply_Q_once(qc, image_a, image_b, q, threshold, b_layout, extra_layout)
    # Add a classical bit and measure the divzero-corrected good qubit.
    from qiskit import ClassicalRegister
    creg = ClassicalRegister(1, "c_good")
    qc.add_register(creg)
    qc.measure(extra_layout["good_corrected"], creg[0])

    sim = AerSimulator(method="matrix_product_state")
    # Transpile to MPS-supported gate set (mcx etc. need decomposition).
    qc_t = transpile(qc, sim, basis_gates=["id", "u", "cx"], optimization_level=0)
    result = sim.run(qc_t, shots=shots).result()
    counts = result.get_counts()
    n_good = counts.get("1", 0)
    return n_good / float(shots)


# ---------------------------------------------------------------------------
# Maximum-likelihood QAE postprocessing
# ---------------------------------------------------------------------------


def mlqae_estimate(
    probs: dict[int, float],
    shots_per_k: int,
) -> float:
    """Given empirical P(good=1) at each Grover power k, fit the
    underlying amplitude a = sin²(θ) by maximum likelihood.

    For a single Grover iteration, P(good=1 | k) = sin²((2k+1)·θ).
    The likelihood for ``shots`` independent Bernoulli trials at each
    k is a product; the log-likelihood is concave in θ over [0, π/2]
    so we maximise via grid search.
    """
    best_ll = -np.inf
    best_theta = 0.0
    for theta in np.linspace(0, np.pi / 2, 1001):
        ll = 0.0
        for k, p_emp in probs.items():
            p_th = np.sin((2 * k + 1) * theta) ** 2
            # Bernoulli log-likelihood; clip to avoid log(0).
            p_th = max(min(p_th, 1 - 1e-12), 1e-12)
            n_good = int(round(p_emp * shots_per_k))
            n_bad = shots_per_k - n_good
            ll += n_good * np.log(p_th) + n_bad * np.log(1 - p_th)
        if ll > best_ll:
            best_ll = ll
            best_theta = theta
    return float(np.sin(best_theta) ** 2)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qae_demo_class_b", description=__doc__)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--threshold", type=int, default=0,
                        help="Classical threshold τ. Pixels with R = I_a // I_b > τ are 'good'.")
    parser.add_argument("--max-k", type=int, default=2,
                        help="Largest Grover power. Total circuits run: max_k + 1.")
    parser.add_argument("--dataset", type=str, default="canonical",
                        choices=["canonical", "canonical_4x4", "fura2", "rogfp2", "synthetic"],
                        help="Which prepared dataset to use.")
    args = parser.parse_args(argv)

    DATA_DIR = Path(__file__).resolve().parents[1] / "paper" / "data_autonomous"
    if args.dataset == "canonical":
        d = np.load(DATA_DIR / "canonical_2x2.npz")
        image_a = d["I_a"].astype(np.int64)
        image_b = d["I_b"].astype(np.int64)
        print(f"Dataset: canonical microscopy frame (membraneStack…rbc3DM2.tif), "
              f"2×2 q=2 from signal-rich patch (8, 4) + 32×32.")
    elif args.dataset == "canonical_4x4":
        d = np.load(DATA_DIR / "canonical_4x4.npz")
        image_a = d["I_a"].astype(np.int64)
        image_b = d["I_b"].astype(np.int64)
        print(f"Dataset: canonical microscopy frame, 4×4 q=2 (n=2) — MPS oracle "
              f"~33 qubits, expected runtime is in the hours.")
    elif args.dataset == "fura2":
        d = np.load(DATA_DIR / "fura2_2x2.npz")
        image_a = d["I_a"].astype(np.int64)
        image_b = d["I_b"].astype(np.int64)
        print(f"Dataset: synthetic Fura-2 (Grynkiewicz 1985), 2×2 q=2.")
    elif args.dataset == "rogfp2":
        d = np.load(DATA_DIR / "rogfp2_2x2.npz")
        image_a = d["I_a"].astype(np.int64)
        image_b = d["I_b"].astype(np.int64)
        print(f"Dataset: synthetic roGFP2 (Schwarzländer 2008), 2×2 q=2.")
    else:
        # Backwards-compat: original hand-tuned synthetic.
        image_a = np.array([[3, 2], [3, 1]], dtype=np.int64)
        image_b = np.array([[1, 1], [3, 1]], dtype=np.int64)
        print("Dataset: hand-tuned synthetic 2×2 q=2 (original demo).")
    q = 2

    # Classical reference: how many pixels have I_a // I_b > threshold AND I_b > 0?
    R = np.where(image_b > 0, image_a.astype(np.int64) // np.maximum(image_b.astype(np.int64), 1), 0)
    valid_mask = image_b > 0
    good_mask = (R > args.threshold) & valid_mask
    n_pixels = image_a.size
    a_true = float(good_mask.sum()) / float(n_pixels)
    print(f"I_a = {image_a.tolist()}")
    print(f"I_b = {image_b.tolist()}")
    print(f"R = {R.tolist()}  (divzero pixels masked to 0)")
    print(f"Classical reference: fraction of pixels with R > {args.threshold} = "
          f"{a_true:.4f} ({int(good_mask.sum())} / {n_pixels})")

    print(f"\nRunning QAE with shots={args.shots} per k, max_k={args.max_k}")
    print(f"  (AerSimulator(method='mps'); each k may take minutes)")

    probs: dict[int, float] = {}
    for k in range(args.max_k + 1):
        print(f"  k={k}: building circuit and sampling...", flush=True)
        p = measure_good_probability(
            image_a, image_b, q, args.threshold, grover_k=k, shots=args.shots
        )
        probs[k] = p
        print(f"    P(good=1 | k={k}) = {p:.4f}   "
              f"(expected sin²((2k+1)·θ) where θ = arcsin(√{a_true:.3f}) = "
              f"{np.arcsin(np.sqrt(a_true)):.3f})")

    a_est = mlqae_estimate(probs, args.shots)
    print(f"\nMaximum-likelihood QAE estimate: â = {a_est:.4f}")
    print(f"Classical truth:                  a = {a_true:.4f}")
    print(f"Absolute error:                   |â - a| = {abs(a_est - a_true):.4f}")
    print(f"Total queries to A: {(args.max_k + 1) * args.shots}")
    print(f"For reference, classical MC with {(args.max_k + 1) * args.shots} samples "
          f"would have stderr ~ {np.sqrt(a_true * (1 - a_true) / ((args.max_k + 1) * args.shots)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
