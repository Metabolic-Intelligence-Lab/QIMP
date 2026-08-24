"""Tests for qimp.processing.ratiometric_circuit (autonomous ratiometric
quantum circuits).

Most of these tests are statevector-bounded: the Class-B ratio circuit
at n=1, q=2 uses ~24 qubits which is statevector-feasible on a 16 GB
laptop. The full Class-A and Class-C pipelines are past that budget even
at their smallest meaningful size (71 and 54 qubits at n=1, q=2,
q_frac=2), so they are verified on `AerSimulator(method='matrix_product_
state')` instead — the n=1 position register superposes only four pixel
branches, so the bond dimension stays small and the runs take seconds.
Larger (n, q) configurations are out of scope for the unit-test suite.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile

from qimp.processing.ratiometric_circuit import (
    affine_subtract_constant,
    class_a_gp_full,
    class_a_gp_prefix,
    class_b_ratio,
    class_b_ratio_inv,
    class_c_rogfp_full,
    decode_class_a_full,
    decode_class_a_prefix,
    decode_class_b_ratio,
    decode_class_c_rogfp,
    dual_neqr_load,
    dual_neqr_load_inv,
    mark_good_oracle,
)


def _set_register_to_int(qc: QuantumCircuit, qubits: list[int], value: int) -> None:
    for i, q in enumerate(qubits):
        if (value >> i) & 1:
            qc.x(q)


def _read_register(sv: Statevector, qubits: list[int]) -> int:
    idx = int(abs(sv.data).argmax())
    out = 0
    for i, q in enumerate(qubits):
        if (idx >> q) & 1:
            out |= 1 << i
    return out


def _statevector_to_counts(sv: Statevector) -> dict[str, int]:
    """Convert an Aer statevector to a deterministic counts dict (one
    shot per non-zero amplitude basis state) so the existing decoder
    can read it without further infrastructure.
    """
    probs = sv.probabilities_dict()
    counts: dict[str, int] = {}
    # Multiply by 1024 and round so we get integer counts; basis-state
    # circuits produce probabilities of 1/(2^n_pos) so 1024 covers the
    # n=1 (4-pixel) case with shot=256 per pixel — exact division.
    multiplier = 1024
    for state, p in probs.items():
        c = int(round(p * multiplier))
        if c > 0:
            counts[state] = c
    return counts


@pytest.mark.parametrize(
    "image_a, image_b",
    [
        # 2×2 images (n=1) with q=2: intensities ∈ {0..3}.
        # Pick combinations that avoid I_b = 0 anywhere.
        (np.array([[3, 2], [3, 1]]), np.array([[1, 1], [3, 1]])),
        (np.array([[3, 0], [2, 3]]), np.array([[1, 1], [1, 1]])),
        (np.array([[0, 0], [0, 0]]), np.array([[1, 2], [3, 1]])),  # all zero / all div
    ],
)
def test_class_b_ratio_2x2_no_divzero(
    image_a: np.ndarray, image_b: np.ndarray
) -> None:
    """At n=1, q=2 verify integer ratio matches the classical computation
    pixel-by-pixel."""
    q = 2
    qc, layout = class_b_ratio(image_a, image_b, q=q)
    sv = Statevector.from_instruction(qc)
    counts = _statevector_to_counts(sv)
    quotient, divzero = decode_class_b_ratio(
        counts, n=1, q=q, layout=layout, total_qubits=qc.num_qubits
    )
    # Classical reference: I_a // I_b where I_b ≠ 0.
    expected = np.zeros_like(image_a)
    for r in range(image_a.shape[0]):
        for c in range(image_a.shape[1]):
            if image_b[r, c] != 0:
                expected[r, c] = image_a[r, c] // image_b[r, c]
    # No divide-by-zero pixels in these inputs → flag should be all-False.
    assert not divzero.any(), f"unexpected div-zero flag: {divzero}"
    np.testing.assert_array_equal(quotient, expected)


def test_class_b_ratio_2x2_with_divzero() -> None:
    """At n=1, q=2, one pixel has I_b=0 → div-zero flag must be set there."""
    q = 2
    image_a = np.array([[3, 2], [1, 1]])
    image_b = np.array([[1, 0], [2, 1]])  # divzero at (0, 1)
    qc, layout = class_b_ratio(image_a, image_b, q=q)
    sv = Statevector.from_instruction(qc)
    counts = _statevector_to_counts(sv)
    quotient, divzero = decode_class_b_ratio(
        counts, n=1, q=q, layout=layout, total_qubits=qc.num_qubits
    )
    # Pixel (0,1) has I_b=0 → flag should be 1 there.
    assert divzero[0, 1], "div-zero flag not set on the I_b=0 pixel"
    # Other three pixels should have correct quotient and flag=0.
    for r, c in [(0, 0), (1, 0), (1, 1)]:
        assert not divzero[r, c], f"unexpected div-zero at ({r}, {c})"
        assert quotient[r, c] == image_a[r, c] // image_b[r, c]


# ---------------------------------------------------------------------------
# Class A — prefix stage (numerator + denominator)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "image_a, image_b",
    [
        (np.array([[3, 2], [1, 0]]), np.array([[1, 3], [2, 0]])),  # mixed signs + zero
        (np.array([[0, 1], [2, 3]]), np.array([[3, 2], [1, 0]])),  # range of values
        (np.array([[2, 2], [2, 2]]), np.array([[2, 2], [2, 2]])),  # equal: num=0 everywhere
    ],
)
def test_class_a_prefix_2x2(
    image_a: np.ndarray, image_b: np.ndarray
) -> None:
    """Verify that the Class-A prefix correctly computes
    num = I_a - I_b (signed) and den = I_a + I_b (unsigned) per pixel."""
    q = 2
    qc, layout = class_a_gp_prefix(image_a, image_b, q=q)
    sv = Statevector.from_instruction(qc)
    counts = _statevector_to_counts(sv)
    num_img, den_img = decode_class_a_prefix(
        counts, n=1, q=q, layout=layout, total_qubits=qc.num_qubits
    )
    expected_num = image_a.astype(np.int64) - image_b.astype(np.int64)
    expected_den = image_a.astype(np.int64) + image_b.astype(np.int64)
    np.testing.assert_array_equal(num_img, expected_num)
    np.testing.assert_array_equal(den_img, expected_den)


# ---------------------------------------------------------------------------
# Class C — affine_subtract_constant primitive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "q, R_val, c_value",
    [
        (2, 0, 0),  # 0 - 0 = 0
        (2, 0, 1),  # 0 - 1 = -1 (two's complement: 111 = 7 unsigned q+1)
        (2, 2, 1),  # 2 - 1 = 1
        (2, 1, 2),  # 1 - 2 = -1
        (2, 3, 3),  # 3 - 3 = 0
        (2, 3, 1),  # 3 - 1 = 2
        (3, 5, 7),  # 5 - 7 = -2 (signed 4-bit two's complement)
        (3, 7, 3),  # 7 - 3 = 4
        (3, 0, 7),  # 0 - 7 = -7
    ],
)
def _flat_amp_index(sv: Statevector) -> int:
    return int(abs(sv.data).argmax())


def _state_is_all_zero(sv: Statevector, n_qubits: int) -> bool:
    """Check that the high-probability state is |0…0⟩ (all qubits zero)."""
    idx = _flat_amp_index(sv)
    return idx == 0


# ---------------------------------------------------------------------------
# Stage F.1 — dual_neqr_load_inv and class_b_ratio_inv round-trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "image_a, image_b",
    [
        (np.array([[3, 2], [1, 0]]), np.array([[1, 3], [2, 1]])),
        (np.array([[0, 0], [3, 3]]), np.array([[1, 2], [0, 3]])),
    ],
)
def test_dual_neqr_load_inv_round_trip(
    image_a: np.ndarray, image_b: np.ndarray
) -> None:
    """dual_neqr_load followed by dual_neqr_load_inv returns the
    position + intensity registers to |0…0⟩."""
    q = 2
    n = 1
    pos = QuantumRegister(2 * n, "pos")
    Ia = QuantumRegister(q, "Ia")
    Ib = QuantumRegister(q, "Ib")
    qc = QuantumCircuit(pos, Ia, Ib)
    pos_idx = list(range(2 * n))
    Ia_idx = list(range(2 * n, 2 * n + q))
    Ib_idx = list(range(2 * n + q, 2 * n + 2 * q))
    dual_neqr_load(
        qc, image_a, image_b, q,
        position_qubits=pos_idx,
        intensity_a_qubits=Ia_idx,
        intensity_b_qubits=Ib_idx,
    )
    dual_neqr_load_inv(
        qc, image_a, image_b, q,
        position_qubits=pos_idx,
        intensity_a_qubits=Ia_idx,
        intensity_b_qubits=Ib_idx,
    )
    sv = Statevector.from_instruction(qc)
    assert _state_is_all_zero(sv, qc.num_qubits), (
        "dual_neqr_load round-trip did not return to |0…0⟩"
    )


# ---------------------------------------------------------------------------
# Stage F.2 — mark_good_oracle truth table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "q, value, threshold, expected_good",
    [
        (2, 0, 0, 0),  # 0 > 0 → False
        (2, 1, 0, 1),  # 1 > 0 → True
        (2, 2, 1, 1),  # 2 > 1 → True
        (2, 1, 2, 0),  # 1 > 2 → False
        (2, 3, 3, 0),  # 3 > 3 → False
        (3, 5, 4, 1),  # 5 > 4 → True
        (3, 4, 5, 0),  # 4 > 5 → False
        (3, 7, 0, 1),  # 7 > 0 → True
    ],
)
def test_mark_good_oracle_truth_table(
    q: int, value: int, threshold: int, expected_good: int
) -> None:
    """At small q, verify the oracle flips good_qubit iff value > threshold,
    and leaves value + ancilla registers untouched."""
    q_w = q + 1
    value_reg = QuantumRegister(q, "value")
    good = QuantumRegister(1, "good")
    thr_reg = QuantumRegister(q_w, "thr")
    sub_c = QuantumRegister(q_w + 1, "sub_c")
    val_pad = QuantumRegister(1, "val_pad")
    qc = QuantumCircuit(value_reg, good, thr_reg, sub_c, val_pad)
    value_idx = list(range(q))
    good_idx = q
    thr_idx = list(range(q + 1, q + 1 + q_w))
    sub_c_idx = list(range(q + 1 + q_w, q + 1 + q_w + q_w + 1))
    val_pad_idx = q + 1 + q_w + q_w + 1
    _set_register_to_int(qc, value_idx, value)
    mark_good_oracle(
        qc,
        value_qubits=value_idx,
        threshold=threshold,
        good_qubit=good_idx,
        threshold_reg_qubits=thr_idx,
        sub_carry_qubits=sub_c_idx,
        value_pad_qubit=val_pad_idx,
    )
    sv = Statevector.from_instruction(qc)
    assert _read_register(sv, [good_idx]) == expected_good, (
        f"q={q} value={value} thr={threshold}: good got "
        f"{_read_register(sv, [good_idx])}, expected {expected_good}"
    )
    # value preserved
    assert _read_register(sv, value_idx) == value
    # threshold register restored to |0⟩
    assert _read_register(sv, thr_idx) == 0
    # sub_carry restored to |0⟩
    for q_idx in sub_c_idx:
        assert _read_register(sv, [q_idx]) == 0
    # value_pad restored to |0⟩
    assert _read_register(sv, [val_pad_idx]) == 0


@pytest.mark.parametrize(
    "q, value, threshold",
    [(2, 1, 1), (2, 2, 1), (3, 5, 4)],
)
def test_mark_good_oracle_self_inverse(
    q: int, value: int, threshold: int
) -> None:
    """Calling mark_good_oracle twice cancels: good_qubit returns to |0⟩,
    everything else stays preserved."""
    q_w = q + 1
    value_reg = QuantumRegister(q, "value")
    good = QuantumRegister(1, "good")
    thr_reg = QuantumRegister(q_w, "thr")
    sub_c = QuantumRegister(q_w + 1, "sub_c")
    val_pad = QuantumRegister(1, "val_pad")
    qc = QuantumCircuit(value_reg, good, thr_reg, sub_c, val_pad)
    value_idx = list(range(q))
    good_idx = q
    thr_idx = list(range(q + 1, q + 1 + q_w))
    sub_c_idx = list(range(q + 1 + q_w, q + 1 + q_w + q_w + 1))
    val_pad_idx = q + 1 + q_w + q_w + 1
    _set_register_to_int(qc, value_idx, value)
    mark_good_oracle(qc, value_idx, threshold, good_idx, thr_idx, sub_c_idx, val_pad_idx)
    mark_good_oracle(qc, value_idx, threshold, good_idx, thr_idx, sub_c_idx, val_pad_idx)
    sv = Statevector.from_instruction(qc)
    assert _read_register(sv, [good_idx]) == 0
    assert _read_register(sv, value_idx) == value
    assert _read_register(sv, thr_idx) == 0
    for q_idx in sub_c_idx:
        assert _read_register(sv, [q_idx]) == 0
    assert _read_register(sv, [val_pad_idx]) == 0


# ---------------------------------------------------------------------------
# Stage F.1 — class_b_ratio_inv round-trip
# (Heavy: 24 qubits, same cost as forward class_b_ratio — ~20 min)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Stage F.3 — class_a_gp_full smoke test (build only; ~70-qubit circuit is
# past laptop statevector capacity, so we only verify the construction does
# not raise and the resulting circuit has the expected register cardinality)
# ---------------------------------------------------------------------------


def test_class_a_gp_full_builds() -> None:
    """class_a_gp_full at n=1, q=2, q_frac=2 builds without raising,
    and the layout has all the documented keys with consistent widths.
    A full statevector or MPS validation is out of scope here — each
    primitive (q_add, q_sub, q_add_ctrl, q_div_general, NEQR encode,
    bit-shift via CNOT, sign-bit copy) is bit-exact verified at small
    width in test_arithmetic.py; the composition's correctness follows
    from the position-independence invariant of all sub-primitives.
    """
    q = 2
    q_frac = 2
    image_a = np.array([[3, 2], [1, 0]], dtype=np.int64)
    image_b = np.array([[1, 3], [2, 1]], dtype=np.int64)
    qc, layout = class_a_gp_full(image_a, image_b, q=q, q_frac=q_frac)
    # Sanity check: layout has every expected key.
    expected_keys = {
        "position", "I_a", "I_b", "num", "den", "add_c", "sub_c",
        "sign", "ones_reg", "inc_carry", "num_scaled", "quotient",
        "div_work", "div_pad", "div_c", "div_flag",
    }
    assert set(layout.keys()) == expected_keys
    # Widths
    q_w = q + 1
    q_dividend = q_w + q_frac
    assert len(layout["position"]) == 2  # type: ignore[arg-type]
    assert len(layout["I_a"]) == q       # type: ignore[arg-type]
    assert len(layout["I_b"]) == q       # type: ignore[arg-type]
    assert len(layout["num"]) == q_w     # type: ignore[arg-type]
    assert len(layout["den"]) == q_w     # type: ignore[arg-type]
    assert len(layout["num_scaled"]) == q_dividend   # type: ignore[arg-type]
    assert len(layout["quotient"]) == q_dividend     # type: ignore[arg-type]
    assert len(layout["div_work"]) == q_w            # type: ignore[arg-type]
    # Total qubit count check.
    expected_total = (
        2 + q + q + q_w + q_w + q_w + q_w + 1 + q_w + (q_w + 1)
        + q_dividend + q_dividend + q_w + 1 + (q_dividend + 1) * (q_w + 2) + 1
    )
    assert qc.num_qubits == expected_total
    # Sanity: circuit has at least the per-pixel encoding gates plus
    # all the arithmetic ops — should be a few hundred ops.
    assert qc.size() > 100


@pytest.mark.parametrize(
    "image_a, image_b",
    [
        (np.array([[3, 2], [3, 1]]), np.array([[1, 1], [3, 1]])),
    ],
)
def test_class_b_ratio_round_trip(
    image_a: np.ndarray, image_b: np.ndarray
) -> None:
    """class_b_ratio followed by class_b_ratio_inv returns the entire
    circuit to |0…0⟩ — the prerequisite for QAE oracle composability."""
    q = 2
    qc, layout = class_b_ratio(image_a, image_b, q=q)
    class_b_ratio_inv(qc, image_a, image_b, q=q, layout=layout)
    sv = Statevector.from_instruction(qc)
    assert _state_is_all_zero(sv, qc.num_qubits), (
        "class_b_ratio round-trip did not return to |0…0⟩"
    )


@pytest.mark.parametrize(
    "q, R_val, c_value",
    # q=2: the exhaustive truth table. `c_value` ranges over the full
    # q+1-bit constant register, not just [0, 2^q): the roGFP reduced
    # reference R_red_fp = round(R_red * 2^q_frac) routinely exceeds 2^q,
    # and c_value > R_val is exactly the negative (sign-bit set) branch.
    [(2, r, c) for r in range(4) for c in range(8)]
    # q=3: endpoints and sign-flip boundary; the full 128-case sweep adds
    # statevector runs without adding coverage.
    + [(3, r, c) for r in (0, 1, 7) for c in (0, 1, 7, 8, 15)],
)
def test_affine_subtract_constant_truth_table(
    q: int, R_val: int, c_value: int
) -> None:
    """At small q, verify the affine constant-subtract is bit-exact: the
    output register encodes (R - c_value) mod 2^(q+1) in two's complement
    and the constant register is restored to |0⟩.
    """
    q_w = q + 1
    R_reg = QuantumRegister(q, "R")
    out_reg = QuantumRegister(q_w, "out")
    c_const = QuantumRegister(q_w, "c_const")
    sub_c = QuantumRegister(q_w + 1, "sub_c")
    qc = QuantumCircuit(R_reg, out_reg, c_const, sub_c)
    R_idx = list(range(q))
    out_idx = list(range(q, q + q_w))
    c_const_idx = list(range(q + q_w, q + 2 * q_w))
    sub_c_idx = list(range(q + 2 * q_w, q + 2 * q_w + q_w + 1))

    _set_register_to_int(qc, R_idx, R_val)
    affine_subtract_constant(
        qc,
        R_qubits=R_idx,
        c_value=c_value,
        output_signed_qubits=out_idx,
        c_const_register_qubits=c_const_idx,
        sub_carry_qubits=sub_c_idx,
    )
    sv = Statevector.from_instruction(qc)
    got_unsigned = _read_register(sv, out_idx)
    expected_unsigned = (R_val - c_value) & ((1 << q_w) - 1)
    assert got_unsigned == expected_unsigned, (
        f"q={q} R={R_val} c={c_value}: got {got_unsigned}, "
        f"expected {expected_unsigned}"
    )
    # Constant register restored to |0⟩
    assert _read_register(sv, c_const_idx) == 0
    # R is preserved
    assert _read_register(sv, R_idx) == R_val


# ---------------------------------------------------------------------------
# Stage F.4 — end-to-end Class-A GP and Class-C roGFP on MPS
#
# These are the manuscript §5.3 claims in miniature, and the only tests that
# exercise (a) `q_div_general` in its non-square configuration, (b) the
# composed full-pipeline decoders, and (c) the MPS execution path. They run
# at n=1, q=2, q_frac=2 (71 and 54 qubits) rather than the paper's q=4
# (137 and 114 qubits) to keep the suite fast; the composition is identical.
# ---------------------------------------------------------------------------


def _run_mps(qc: QuantumCircuit, shots: int = 512) -> dict[str, int]:
    """Measure every qubit and sample `qc` on the MPS backend.

    Both pipelines are basis-state circuits per pixel branch, so the
    per-pixel majority vote in the decoders is exact for any shot count
    that covers the 4^n branches; 512 leaves ~128 shots per pixel.
    """
    qc = qc.copy()
    qc.add_register(ClassicalRegister(qc.num_qubits))
    qc.measure(range(qc.num_qubits), range(qc.num_qubits))
    # Transpile to the MPS basis only — no backend, so no coupling-map
    # width cap (this mirrors scripts/run_autonomous_class_{a,c}_*_mps.py).
    qc_t = transpile(qc, basis_gates=["id", "u", "cx"], optimization_level=0)
    sim = AerSimulator(method="matrix_product_state", seed_simulator=20260610)
    counts: dict[str, int] = sim.run(qc_t, shots=shots).result().get_counts()
    return counts


def test_class_a_gp_full_end_to_end_mps() -> None:
    """Class A at n=1, q=2, q_frac=2 recovers the signed GP bit-exactly.

    The patch is chosen to span both saturating endpoints (GP = ±1, where
    one channel quantises to 0), a proper fraction, and zero. The ±1
    pixels are the regression guard on `decode_class_a_full`: there the
    quotient magnitude is exactly 2**q_frac, so it sets bit q_frac and a
    decoder reading only the low q_frac bits would report 0.0 instead.
    """
    n, q, q_frac = 1, 2, 2
    image_a = np.array([[3, 3], [0, 3]], dtype=np.int64)
    image_b = np.array([[0, 1], [1, 3]], dtype=np.int64)
    # den = I_a + I_b, num = |I_a - I_b|, sign = (I_b > I_a);
    # GP = (-1)^sign * ((num << q_frac) // den) / 2^q_frac
    expected_gp = np.array([[1.0, 0.5], [-1.0, 0.0]])

    qc, layout = class_a_gp_full(image_a, image_b, q=q, q_frac=q_frac)
    assert qc.num_qubits == 71
    counts = _run_mps(qc)
    gp, divzero = decode_class_a_full(
        counts, n, q, q_frac, layout, qc.num_qubits
    )
    # den = I_a + I_b > 0 at every pixel here, so no div-zero.
    assert not divzero.any(), f"unexpected div-zero flag: {divzero}"
    np.testing.assert_array_equal(gp, expected_gp)


def test_class_a_gp_full_divzero_flag_mps() -> None:
    """Class A sets the div-zero flag exactly where I_a + I_b == 0."""
    n, q, q_frac = 1, 2, 2
    image_a = np.array([[3, 0], [1, 2]], dtype=np.int64)
    image_b = np.array([[1, 0], [3, 2]], dtype=np.int64)  # den == 0 at (0, 1)

    qc, layout = class_a_gp_full(image_a, image_b, q=q, q_frac=q_frac)
    counts = _run_mps(qc)
    gp, divzero = decode_class_a_full(
        counts, n, q, q_frac, layout, qc.num_qubits
    )
    assert divzero[0, 1], "div-zero flag not set on the I_a + I_b == 0 pixel"
    # The other three decode exactly; GP is undefined on the flagged pixel.
    expected_valid = {(0, 0): 0.5, (1, 0): -0.5, (1, 1): 0.0}
    for (r, c), want in expected_valid.items():
        assert not divzero[r, c], f"unexpected div-zero at ({r}, {c})"
        assert gp[r, c] == want, f"GP at ({r}, {c}): got {gp[r, c]}, want {want}"


def test_class_c_rogfp_full_end_to_end_mps() -> None:
    """Class C at n=1, q=2, q_frac=2 recovers the calibrated redox index.

    The fractional ratio (shift by q_frac, then non-square divide) is what
    lifts the Class-B integer-quotient degeneracy, and the affine subtract
    runs in-circuit; only the calibration scalar is classical. One pixel is
    below R_red so `rc_signed` goes negative — the two's-complement
    sign-extension branch of the decoder.
    """
    n, q, q_frac = 1, 2, 2
    R_red, R_ox = 0.75, 3.0
    R_red_fp = round(R_red * (1 << q_frac))  # 3
    image_a = np.array([[3, 2], [0, 3]], dtype=np.int64)
    image_b = np.array([[2, 1], [3, 3]], dtype=np.int64)
    # ratio_fp = (I_a << q_frac) // I_b = [[6, 8], [0, 4]]
    # rc_signed = ratio_fp - R_red_fp = [[3, 5], [-3, 1]]
    # R_C = rc_signed / ((R_ox - R_red) * 2^q_frac) = rc_signed / 9
    expected_rc = np.array([[3.0, 5.0], [-3.0, 1.0]]) / 9.0

    qc, layout = class_c_rogfp_full(
        image_a, image_b, q=q, q_frac=q_frac, R_red_fp=R_red_fp
    )
    assert qc.num_qubits == 54
    counts = _run_mps(qc)
    rc, divzero = decode_class_c_rogfp(
        counts, n, q, q_frac, R_red, R_ox, layout, qc.num_qubits
    )
    assert not divzero.any(), f"unexpected div-zero flag: {divzero}"
    np.testing.assert_allclose(rc, expected_rc)


def test_class_c_rogfp_full_divzero_flag_mps() -> None:
    """Class C sets the div-zero flag exactly where I_b == 0."""
    n, q, q_frac = 1, 2, 2
    R_red, R_ox = 0.75, 3.0
    R_red_fp = round(R_red * (1 << q_frac))  # 3
    image_a = np.array([[3, 2], [1, 3]], dtype=np.int64)
    image_b = np.array([[2, 0], [1, 3]], dtype=np.int64)  # divzero at (0, 1)

    qc, layout = class_c_rogfp_full(
        image_a, image_b, q=q, q_frac=q_frac, R_red_fp=R_red_fp
    )
    counts = _run_mps(qc)
    rc, divzero = decode_class_c_rogfp(
        counts, n, q, q_frac, R_red, R_ox, layout, qc.num_qubits
    )
    assert divzero[0, 1], "div-zero flag not set on the I_b == 0 pixel"
    denom = (R_ox - R_red) * (1 << q_frac)
    for r, c in [(0, 0), (1, 0), (1, 1)]:
        assert not divzero[r, c], f"unexpected div-zero at ({r}, {c})"
        want = (
            (int(image_a[r, c]) << q_frac) // int(image_b[r, c]) - R_red_fp
        ) / denom
        assert rc[r, c] == pytest.approx(want), (
            f"R_C at ({r}, {c}): got {rc[r, c]}, want {want}"
        )
