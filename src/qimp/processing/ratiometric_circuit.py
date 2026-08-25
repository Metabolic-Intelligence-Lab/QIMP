"""Autonomous quantum circuits for per-pixel ratiometric image operators.

Each operator builder takes two NEQR-encoded intensity images and
constructs a circuit that **computes** the ratio (or normalised
difference, or calibrated ratio) on the quantum device — no
classically pre-computed target is loaded into the circuit. The
intended use of these circuits is twofold: (1) as a state-prep oracle
for downstream QAE / Grover quantum-advantage demonstrations on
aggregate image statistics (planned, see project plan)
(2) as a
reference autonomous primitive that answers the "target loading"
critique of the parameter-synthesis FRQI route in
``qimp.processing.gp_ratio``.

The three classes supported (in order of implementation):

  - **Class B** — simple ratio R = I_a / I_b (integer quotient form).
    Implemented in :func:`class_b_ratio` (this file).
  - **Class A** — Generalised Polarization
    GP = (I_a − G · I_b)/(I_a + G · I_b) with a classical G factor.
    Planned: :func:`class_a_gp` (extends Class B with a constant
    multiplier and signed numerator).
  - **Class C** — roGFP calibrated ratio
    R_C = (R − R_red)/(R_ox − R_red) with classical R_red, R_ox.
    Implemented end-to-end in :func:`class_c_rogfp_full` (fractional
    ratio via shift + non-square divide, then affine subtract).

All builders share the convention that intensities are q-bit unsigned
fixed point, position is 2n-bit, and the output register is a separate
q-bit register holding the unsigned quotient.

NOTE: The intermediate divider primitive used here lives in
``qimp.processing.arithmetic.q_div_restoring`` and is built on the
existing Cuccaro adders. Its inverse :func:`q_div_restoring_inv` will
be added in a follow-up commit and is needed only for QAE-style
state-prep reversibility — for the first POC, the divider's carry
ancilla is left dirty after circuit execution and the result is read
out by measurement.
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit

from qimp.processing.arithmetic import (
    q_add,
    q_add_ctrl,
    q_div_general,
    q_div_nonrestoring,
    q_div_nonrestoring_inv,
    q_div_restoring,
    q_div_restoring_inv,
    q_sub,
)

__all__ = [
    "affine_subtract_constant",
    "class_a_gp_full",
    "class_a_gp_prefix",
    "class_b_ratio",
    "class_b_ratio_inv",
    "class_c_rogfp_full",
    "decode_class_a_full",
    "decode_class_a_prefix",
    "decode_class_b_ratio",
    "decode_class_c_rogfp",
    "dual_neqr_load",
    "dual_neqr_load_inv",
    "mark_good_oracle",
    "mark_good_oracle_inv",
]


def _validate_dual_images(image_a: np.ndarray, image_b: np.ndarray, q: int) -> int:
    if q < 1:
        raise ValueError("q must be >= 1")
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"image_a and image_b must have same shape, got {image_a.shape} vs {image_b.shape}"
        )
    if image_a.ndim != 2 or image_a.shape[0] != image_a.shape[1]:
        raise ValueError(f"images must be square 2D, got {image_a.shape}")
    side = image_a.shape[0]
    if side < 2 or (side & (side - 1)) != 0:
        raise ValueError(f"image side must be a power of two ≥ 2, got {side}")
    max_int = (1 << q) - 1
    if image_a.min() < 0 or image_b.min() < 0:
        raise ValueError("NEQR requires non-negative intensities")
    if image_a.max() > max_int or image_b.max() > max_int:
        raise ValueError(f"max intensity exceeds 2^q - 1 = {max_int}; increase q or rescale")
    return int(math.log2(side))


def _encode_intensity_at_pixel(
    qc: QuantumCircuit,
    position_qubits: list[int],
    intensity_qubits: list[int],
    n: int,
    intensity: int,
    row: int,
    col: int,
) -> None:
    """Apply controlled-X gates so the position |row, col⟩ branch sets
    `intensity_qubits` to the value `intensity` (binary, little-endian).
    The position-register pattern matching is the standard NEQR pattern:
    flip non-target position bits, MCX onto the matching intensity bits,
    flip the position bits back.
    """
    q = len(intensity_qubits)
    x_bits = format(col, f"0{n}b")[::-1]  # LSB first
    y_bits = format(row, f"0{n}b")[::-1]
    x_flips = [position_qubits[i] for i, bit in enumerate(x_bits) if bit == "0"]
    y_flips = [position_qubits[n + i] for i, bit in enumerate(y_bits) if bit == "0"]
    all_flips = x_flips + y_flips
    for qbit in all_flips:
        qc.x(qbit)
    int_bits = format(intensity, f"0{q}b")[::-1]
    for i, bit in enumerate(int_bits):
        if bit == "1":
            qc.mcx(list(position_qubits), intensity_qubits[i])
    for qbit in all_flips:
        qc.x(qbit)


def dual_neqr_load(
    qc: QuantumCircuit,
    image_a: np.ndarray,
    image_b: np.ndarray,
    q: int,
    position_qubits: list[int],
    intensity_a_qubits: list[int],
    intensity_b_qubits: list[int],
) -> None:
    """Load two co-registered intensity images into NEQR registers
    sharing a single position register.

    The position register is placed into uniform superposition via H on
    every position qubit. For each pixel ``(row, col)`` with non-zero
    intensity, controlled-X gates write the intensity into the
    intensity registers conditioned on the position pattern.

    Parameters
    ----------
    qc
        Circuit to add the load operations to. Must already contain
        all the qubits passed in.
    image_a, image_b
        Square ``2^n × 2^n`` integer images with values in [0, 2^q − 1].
    q
        Intensity bit width.
    position_qubits
        ``2n``-bit position register, |0⟩ on entry, in uniform
        superposition on exit.
    intensity_a_qubits, intensity_b_qubits
        Two ``q``-bit intensity registers, |0⟩ on entry.
    """
    n = _validate_dual_images(image_a, image_b, q)
    if len(position_qubits) != 2 * n:
        raise ValueError(f"position_qubits must have 2n={2 * n} bits, got {len(position_qubits)}")
    if len(intensity_a_qubits) != q or len(intensity_b_qubits) != q:
        raise ValueError(
            f"intensity registers must have q={q} bits, got "
            f"{len(intensity_a_qubits)} and {len(intensity_b_qubits)}"
        )

    qc.h(position_qubits)
    qc.barrier()

    img_a = image_a.astype(np.int64)
    img_b = image_b.astype(np.int64)
    for row in range(1 << n):
        for col in range(1 << n):
            ia = int(img_a[row, col])
            ib = int(img_b[row, col])
            if ia == 0 and ib == 0:
                continue
            if ia != 0:
                _encode_intensity_at_pixel(qc, position_qubits, intensity_a_qubits, n, ia, row, col)
            if ib != 0:
                _encode_intensity_at_pixel(qc, position_qubits, intensity_b_qubits, n, ib, row, col)
            qc.barrier()


def dual_neqr_load_inv(
    qc: QuantumCircuit,
    image_a: np.ndarray,
    image_b: np.ndarray,
    q: int,
    position_qubits: list[int],
    intensity_a_qubits: list[int],
    intensity_b_qubits: list[int],
) -> None:
    """Inverse of :func:`dual_neqr_load`.

    Each per-pixel ``X-MCX-X`` block in the forward call is self-inverse,
    so re-applying it cancels the original encoding. The final Hadamard
    on the position register is also self-inverse. To match the
    forward's gate ORDER reversed:

      forward:  H(position) → encode pixel 0 → encode pixel 1 → … → encode pixel N
      inverse:  encode pixel N → encode pixel N-1 → … → encode pixel 0 → H(position)

    Since per-pixel encodings commute (they leave the position register
    invariant between pixels and act on disjoint intensity-bit
    combinations), the iteration order can match the forward's exactly;
    we still walk it in the same row-major order for stylistic
    consistency. The H is moved to the END of the body.
    """
    n = _validate_dual_images(image_a, image_b, q)
    if len(position_qubits) != 2 * n:
        raise ValueError(f"position_qubits must have 2n={2 * n} bits, got {len(position_qubits)}")
    if len(intensity_a_qubits) != q or len(intensity_b_qubits) != q:
        raise ValueError(
            f"intensity registers must have q={q} bits, got "
            f"{len(intensity_a_qubits)} and {len(intensity_b_qubits)}"
        )

    img_a = image_a.astype(np.int64)
    img_b = image_b.astype(np.int64)
    for row in range(1 << n):
        for col in range(1 << n):
            ia = int(img_a[row, col])
            ib = int(img_b[row, col])
            if ia == 0 and ib == 0:
                continue
            if ia != 0:
                _encode_intensity_at_pixel(qc, position_qubits, intensity_a_qubits, n, ia, row, col)
            if ib != 0:
                _encode_intensity_at_pixel(qc, position_qubits, intensity_b_qubits, n, ib, row, col)
            qc.barrier()
    qc.h(position_qubits)


def class_b_ratio(
    image_a: np.ndarray,
    image_b: np.ndarray,
    q: int,
    divider: str = "restoring",
) -> tuple[QuantumCircuit, dict[str, list[int] | int]]:
    """Build an autonomous Class-B integer-quotient ratio circuit.

    For each pixel position p in superposition, the circuit computes
    R(p) = I_a(p) // I_b(p) using reversible division (no classical
    pre-computation of R).

    Parameters
    ----------
    image_a, image_b
        2D arrays of the two intensity channels (shape 2^n × 2^n).
    q
        Width of each intensity register, in qubits.
    divider
        ``"restoring"`` (default, historical) or ``"nonrestoring"``
        (Thapliyal-style; ~50% fewer two-qubit gates after transpile,
        same output semantics, same ancilla footprint).

    Returns
    -------
    (qc, layout)
        ``qc`` is the assembled QuantumCircuit
        ``layout`` is a dict
        with qubit-index lists for each named register:
        ``position``, ``I_a``, ``I_b``, ``quotient``, ``work``,
        ``pad``, ``c``, ``flag``.

    The quotient is read out by measuring the ``quotient`` register.
    The ``flag`` qubit is set to 1 for pixels where I_b == 0 (the
    quotient at those pixels is undefined and the caller must mask
    them in the decoded image).

    Resource costs:
      - Qubits: ``2n + 4q + 2 + (q+1)(q+2)``. At n=1, q=2: 24 qubits.
      - Toffolis: O(q²) for the division × 1 per pixel branch (the
        division acts uniformly on the position superposition).
    """
    if divider not in ("restoring", "nonrestoring"):
        raise ValueError(f"divider must be 'restoring' or 'nonrestoring', got {divider!r}")
    n = _validate_dual_images(image_a, image_b, q)
    n_pos = 2 * n
    n_c = (q + 1) * (q + 2)

    # Qubit-index layout (allocate in order):
    layout: dict[str, list[int] | int] = {}
    cursor = 0
    layout["position"] = list(range(cursor, cursor + n_pos))
    cursor += n_pos
    layout["I_a"] = list(range(cursor, cursor + q))
    cursor += q
    layout["I_b"] = list(range(cursor, cursor + q))
    cursor += q
    layout["quotient"] = list(range(cursor, cursor + q))
    cursor += q
    layout["work"] = list(range(cursor, cursor + q))
    cursor += q
    layout["pad"] = cursor
    cursor += 1
    layout["c"] = list(range(cursor, cursor + n_c))
    cursor += n_c
    layout["flag"] = cursor
    cursor += 1
    total = cursor

    qc = QuantumCircuit(total)

    # `layout` is dict[str, list[int] | int]; narrow each entry once here so
    # the primitive calls below take properly-typed arguments.
    pos = layout["position"]
    Ia = layout["I_a"]
    Ib = layout["I_b"]
    quotient = layout["quotient"]
    work = layout["work"]
    pad = layout["pad"]
    c = layout["c"]
    flag = layout["flag"]
    assert isinstance(pos, list) and isinstance(Ia, list) and isinstance(Ib, list)
    assert isinstance(quotient, list) and isinstance(work, list) and isinstance(c, list)
    assert isinstance(pad, int) and isinstance(flag, int)

    dual_neqr_load(
        qc,
        image_a,
        image_b,
        q,
        position_qubits=pos,
        intensity_a_qubits=Ia,
        intensity_b_qubits=Ib,
    )

    # Class B integer ratio: I_a // I_b. Divider overwrites I_a register
    # with the remainder; I_b is preserved.
    div_fn = q_div_restoring if divider == "restoring" else q_div_nonrestoring
    div_fn(
        qc,
        dividend_qubits=Ia,
        divisor_qubits=Ib,
        quotient_qubits=quotient,
        work_qubits=work,
        divisor_pad_qubit=pad,
        c_qubits=c,
        div_zero_flag=flag,
    )
    return qc, layout


def decode_class_b_ratio(
    counts: dict[str, int],
    n: int,
    q: int,
    layout: dict[str, list[int] | int],
    total_qubits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode measurement counts into (quotient_image, divzero_mask).

    Parameters
    ----------
    counts
        Qiskit ``counts`` dict from ``Statevector.probabilities_dict``
        or a shot-based simulator.
    n, q
        Spatial qubit count and intensity width.
    layout
        The layout dict returned by :func:`class_b_ratio`.
    total_qubits
        Total qubit count of the circuit (so we know the bitstring
        width Qiskit produces).

    Returns
    -------
    (quotient_image, divzero_mask)
        Two ``2^n × 2^n`` arrays. ``quotient_image[row, col]`` holds
        the unsigned integer quotient ``I_a // I_b`` for that pixel
        (taken as the majority vote of measured intensity values).
        ``divzero_mask[row, col]`` is True if the div-zero flag was
        observed set for that pixel branch.
    """
    side = 1 << n
    quotient_img = np.zeros((side, side), dtype=np.int64)
    divzero_mask = np.zeros((side, side), dtype=bool)
    # Build histograms per pixel
    histograms: dict[tuple[int, int], dict[int, int]] = {}
    flag_counts: dict[tuple[int, int], dict[int, int]] = {}
    pos_qubits = layout["position"]
    quo_qubits = layout["quotient"]
    flag_qubit = layout["flag"]
    assert isinstance(pos_qubits, list)
    assert isinstance(quo_qubits, list)
    assert isinstance(flag_qubit, int)

    # Qiskit reports MSB → LSB, qubit i is at position total_qubits-1-i.
    # Defined once with `flat` as a parameter rather than closed over inside
    # the loop, matching decode_class_a_full / decode_class_c_rogfp.
    def bit_at(flat: str, qubit_idx: int) -> int:
        return int(flat[total_qubits - 1 - qubit_idx])

    for state, count in counts.items():
        flat = state.replace(" ", "")
        # Position layout: low n bits are x (col), high n bits are y (row).
        col = sum(bit_at(flat, pos_qubits[i]) << i for i in range(n))
        row = sum(bit_at(flat, pos_qubits[n + i]) << i for i in range(n))
        quo = sum(bit_at(flat, quo_qubits[i]) << i for i in range(q))
        flag = bit_at(flat, flag_qubit)

        histograms.setdefault((row, col), {})
        histograms[(row, col)][quo] = histograms[(row, col)].get(quo, 0) + count
        flag_counts.setdefault((row, col), {})
        flag_counts[(row, col)][flag] = flag_counts[(row, col)].get(flag, 0) + count

    for (row, col), votes in histograms.items():
        quotient_img[row, col] = max(votes, key=votes.__getitem__)
    for (row, col), flag_votes in flag_counts.items():
        divzero_mask[row, col] = max(flag_votes, key=flag_votes.__getitem__) == 1
    return quotient_img, divzero_mask


def class_b_ratio_inv(
    qc: QuantumCircuit,
    image_a: np.ndarray,
    image_b: np.ndarray,
    q: int,
    layout: dict[str, list[int] | int],
    divider: str = "restoring",
) -> None:
    """Exact inverse of :func:`class_b_ratio`.

    The ``divider`` argument must match the variant used in the forward
    call (``"restoring"`` or ``"nonrestoring"``).

    Reverses the divider+load composition in the standard order:
    forward = ``dual_neqr_load → q_div_*``, so the inverse is
    ``q_div_*_inv → dual_neqr_load_inv``.

    After applying this on top of a circuit that previously had
    :func:`class_b_ratio` applied, every register returns to ``|0⟩``:
    position, both intensity registers, the quotient, work, pad, carry,
    and div-zero flag.

    This is the building block for QAE-style oracle composition where
    the state-prep ``A`` and its conjugate ``A†`` are applied
    alternately inside the Grover operator.

    Parameters
    ----------
    qc
        The circuit to which the inverse is appended.
    image_a, image_b, q
        Must match the inputs that were used to build the forward
        ``class_b_ratio`` circuit (the inverse re-uses the same
        per-pixel encoding gates).
    layout
        The layout dict returned by :func:`class_b_ratio` — its qubit
        indices direct the inverse onto the right registers.
    """
    # First undo the divider step.
    if divider not in ("restoring", "nonrestoring"):
        raise ValueError(f"divider must be 'restoring' or 'nonrestoring', got {divider!r}")
    inv_fn = q_div_restoring_inv if divider == "restoring" else q_div_nonrestoring_inv
    inv_fn(
        qc,
        dividend_qubits=layout["I_a"],  # type: ignore[arg-type]
        divisor_qubits=layout["I_b"],  # type: ignore[arg-type]
        quotient_qubits=layout["quotient"],  # type: ignore[arg-type]
        work_qubits=layout["work"],  # type: ignore[arg-type]
        divisor_pad_qubit=layout["pad"],  # type: ignore[arg-type]
        c_qubits=layout["c"],  # type: ignore[arg-type]
        div_zero_flag=layout["flag"],  # type: ignore[arg-type]
    )
    # Then undo the dual-NEQR load.
    dual_neqr_load_inv(
        qc,
        image_a,
        image_b,
        q,
        position_qubits=layout["position"],  # type: ignore[arg-type]
        intensity_a_qubits=layout["I_a"],  # type: ignore[arg-type]
        intensity_b_qubits=layout["I_b"],  # type: ignore[arg-type]
    )


def mark_good_oracle(
    qc: QuantumCircuit,
    value_qubits: list[int],
    threshold: int,
    good_qubit: int,
    threshold_reg_qubits: list[int],
    sub_carry_qubits: list[int],
    value_pad_qubit: int,
) -> None:
    """Flip ``good_qubit`` iff ``value > threshold``.

    Implementation: build a (q+1)-bit ``value_padded`` register by
    appending a zero-pad ancilla to the q-bit value, materialise
    ``threshold + 1`` into a (q+1)-bit constant register, and subtract
    ``threshold_reg ← threshold_reg − value_padded``. The no-borrow bit
    of the result is 1 iff ``threshold + 1 ≥ value``, i.e.
    ``value ≤ threshold``
    CNOT that into ``good_qubit`` and then
    X-flip it so ``good_qubit ^= (value > threshold)``. Uncompute the
    subtract and the constant materialisation so all ancilla return to
    ``|0⟩``.

    Parameters
    ----------
    value_qubits
        ``q``-bit unsigned register holding the value to test. Preserved.
    threshold
        Classical integer in ``[0, 2^q − 1]``. The predicate
        ``value > threshold`` is checked.
    good_qubit
        Single qubit
        XORed with the predicate's truth value.
    threshold_reg_qubits
        ``q + 1``-bit ancilla, ``|0⟩`` on entry and exit. Holds
        ``threshold + 1`` during the test.
    sub_carry_qubits
        ``q + 2``-bit carry register for the (q+1)-wide subtract,
        ``|0⟩`` on entry and exit.
    value_pad_qubit
        Single ancilla, ``|0⟩`` on entry and exit. Used as the high
        zero-pad of ``value`` to widen it from q to q+1 bits for the
        subtract.

    The construction is its own inverse: applying it twice on the same
    inputs returns ``good_qubit`` to its initial state (two XORs cancel),
    and all ancillas are restored each call. See
    :func:`mark_good_oracle_inv` for the symmetric API.
    """
    q = len(value_qubits)
    q_w = q + 1
    if len(threshold_reg_qubits) != q_w:
        raise ValueError(
            f"threshold_reg_qubits must have q+1={q_w} bits, got {len(threshold_reg_qubits)}"
        )
    if len(sub_carry_qubits) != q_w + 1:
        raise ValueError(
            f"sub_carry_qubits must have q+2={q_w + 1} bits, got {len(sub_carry_qubits)}"
        )
    target = threshold + 1
    if target < 0 or target >= (1 << q_w):
        raise ValueError(
            f"threshold + 1 = {target} out of representable range "
            f"[0, 2^(q+1) - 1] = [0, {(1 << q_w) - 1}]"
        )

    # Materialise (threshold + 1) into threshold_reg_qubits.
    for i in range(q_w):
        if (target >> i) & 1:
            qc.x(threshold_reg_qubits[i])
    # Build value_padded by appending the dedicated value_pad_qubit
    # (guaranteed |0⟩) as the MSB of value.
    value_padded = [*list(value_qubits), value_pad_qubit]
    # Subtract: value_padded ← value_padded − threshold_reg.
    # The no-borrow bit is 1 iff value_padded ≥ threshold_reg, i.e.
    # value ≥ threshold + 1, i.e. value > threshold. That is exactly
    # the predicate we want. q_sub(a, b, c) computes b ← b − a, so
    # a = threshold_reg, b = value_padded.
    from qimp.processing.arithmetic import q_sub as _q_sub
    from qimp.processing.arithmetic import q_sub_inv as _q_sub_inv

    _q_sub(qc, threshold_reg_qubits, value_padded, sub_carry_qubits)
    qc.cx(sub_carry_qubits[-1], good_qubit)
    # Uncompute the subtract to restore value_padded (and hence
    # value_qubits + value_pad_qubit) to their pre-call state.
    _q_sub_inv(qc, threshold_reg_qubits, value_padded, sub_carry_qubits)
    # X-unset threshold_reg.
    for i in range(q_w):
        if (target >> i) & 1:
            qc.x(threshold_reg_qubits[i])


def mark_good_oracle_inv(
    qc: QuantumCircuit,
    value_qubits: list[int],
    threshold: int,
    good_qubit: int,
    threshold_reg_qubits: list[int],
    sub_carry_qubits: list[int],
    value_pad_qubit: int,
) -> None:
    """Inverse of :func:`mark_good_oracle`. Since the forward construction
    is self-inverse (the two XORs into ``good_qubit`` cancel; all ancilla
    sub-procedures are themselves uncomputed), the inverse is simply the
    forward function applied again.
    """
    mark_good_oracle(
        qc,
        value_qubits=value_qubits,
        threshold=threshold,
        good_qubit=good_qubit,
        threshold_reg_qubits=threshold_reg_qubits,
        sub_carry_qubits=sub_carry_qubits,
        value_pad_qubit=value_pad_qubit,
    )


# ============================================================================
# Class A — Generalised Polarization (G = 1 Möbius form)
# ============================================================================


def _copy_register(qc: QuantumCircuit, src: list[int], dst: list[int]) -> None:
    """In-place classical copy via CNOTs: dst ← dst ⊕ src.

    For basis-state inputs this is a true copy
    in superposition it
    creates entanglement between src and dst with the same per-branch
    value. Used to duplicate an NEQR intensity register so the original
    is preserved while the copy is consumed by arithmetic.
    """
    if len(src) != len(dst):
        raise ValueError(f"register widths differ: {len(src)} vs {len(dst)}")
    for s, d in zip(src, dst, strict=True):
        qc.cx(s, d)


def class_a_gp_prefix(
    image_a: np.ndarray,
    image_b: np.ndarray,
    q: int,
    G: float = 1.0,
) -> tuple[QuantumCircuit, dict[str, list[int] | int]]:
    """Class A GP **prefix stage**: autonomously compute the numerator
    (I_a − G·I_b, two's complement) and denominator (I_a + G·I_b,
    unsigned) registers from two NEQR-encoded intensity images.

    This is the first half of the full GP pipeline
    the second half
    (conditional |·| then fractional divider) is currently
    statevector-infeasible and pending the non-square ``q_div`` variant
    — see Stage C note at the end of this module. The split exists for
    two reasons: (1) the prefix is bit-exact-testable at small (n, q)
    while the full pipeline is not
    (2) the divider step uses a
    different carry-ancilla budget that benefits from a separate
    layout pass.

    Parameters
    ----------
    image_a, image_b
        Square ``2^n × 2^n`` integer images with values in
        ``[0, 2^q − 1]``.
    q
        Intensity bit width.
    G
        Classical GP G-factor. Only ``G = 1.0`` is currently supported;
        any other value raises ``NotImplementedError``. General G
        requires a ``q_mul_const`` composition to materialise
        ``G · I_b`` before the add/sub — straightforward to add and
        documented in the module-level note.

    Output layout (all little-endian):
      - ``position``  (2n) — shared NEQR position register, in
        uniform superposition after dual_neqr_load.
      - ``I_a``, ``I_b``  (q each) — NEQR intensities, preserved.
      - ``num``  (q + 1) — holds (I_a − I_b) two's-complement,
        signed q+1-bit (MSB = sign).
      - ``den``  (q + 1) — holds (I_a + I_b) unsigned q+1-bit.
      - ``add_c`` (q + 1) — dirty carry register from the add.
      - ``sub_c`` (q + 1) — dirty carry register from the sub.

    After this builder:
      - ``den[MSB] = carry-out of (I_a + I_b)`` (the bit normally
        written into a q+1-bit accumulator).
      - ``num[MSB] = sign bit`` (0 iff I_a ≥ I_b, 1 iff I_a < I_b).
    """
    if G != 1.0:
        raise NotImplementedError(
            f"G={G}: general G requires q_mul_const composition; "
            "currently only G=1.0 is supported. See the module-level "
            "Stage C note for the recipe."
        )
    n = _validate_dual_images(image_a, image_b, q)
    n_pos = 2 * n
    q_w = q + 1

    layout: dict[str, list[int] | int] = {}
    cursor = 0
    layout["position"] = list(range(cursor, cursor + n_pos))
    cursor += n_pos
    layout["I_a"] = list(range(cursor, cursor + q))
    cursor += q
    layout["I_b"] = list(range(cursor, cursor + q))
    cursor += q
    layout["num"] = list(range(cursor, cursor + q_w))
    cursor += q_w
    layout["den"] = list(range(cursor, cursor + q_w))
    cursor += q_w
    layout["add_c"] = list(range(cursor, cursor + q_w))
    cursor += q_w  # n+1 carries for the q-wide add (n = q)
    layout["sub_c"] = list(range(cursor, cursor + q_w))
    cursor += q_w  # n+1 carries for the q-wide sub
    total = cursor

    qc = QuantumCircuit(total)

    pos = layout["position"]
    Ia = layout["I_a"]
    Ib = layout["I_b"]
    num = layout["num"]
    den = layout["den"]
    add_c = layout["add_c"]
    sub_c = layout["sub_c"]
    assert isinstance(pos, list) and isinstance(Ia, list) and isinstance(Ib, list)
    assert isinstance(num, list) and isinstance(den, list)
    assert isinstance(add_c, list) and isinstance(sub_c, list)

    dual_neqr_load(
        qc,
        image_a,
        image_b,
        q,
        position_qubits=pos,
        intensity_a_qubits=Ia,
        intensity_b_qubits=Ib,
    )

    # den ← I_a (copy into low q bits; MSB stays 0)
    _copy_register(qc, Ia, den[:q])
    # den ← den + I_b at q-bit width; carry-out goes into den[q] (MSB).
    q_add(qc, Ib, list(den[:q]), add_c)
    qc.cx(add_c[-1], den[q])

    # num ← I_a (copy into low q bits; MSB stays 0)
    _copy_register(qc, Ia, num[:q])
    # num ← num - I_b at q-bit width. q_sub leaves the no-borrow bit
    # in sub_c[-1]; the result modular at 2^q in num[:q]. The sign bit
    # we want in num[q] is the NEGATION of the no-borrow bit.
    q_sub(qc, Ib, list(num[:q]), sub_c)
    qc.cx(sub_c[-1], num[q])
    qc.x(num[q])  # num[q] = 1 iff I_a < I_b (the two's-complement sign bit)

    return qc, layout


def decode_class_a_prefix(
    counts: dict[str, int],
    n: int,
    q: int,
    layout: dict[str, list[int] | int],
    total_qubits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode the prefix-stage output: (num_signed, den_unsigned) per pixel.

    Returns
    -------
    (num_image, den_image)
        Both ``2^n × 2^n`` int arrays. ``num_image`` is the signed
        two's-complement interpretation of the ``num`` register (sign
        bit at position q, magnitude at positions 0..q-1). ``den_image``
        is the unsigned (q+1)-bit sum.
    """
    side = 1 << n
    num_img = np.zeros((side, side), dtype=np.int64)
    den_img = np.zeros((side, side), dtype=np.int64)
    pos = layout["position"]
    num_q = layout["num"]
    den_q = layout["den"]
    assert isinstance(pos, list)
    assert isinstance(num_q, list)
    assert isinstance(den_q, list)
    histograms_num: dict[tuple[int, int], dict[int, int]] = {}
    histograms_den: dict[tuple[int, int], dict[int, int]] = {}

    def bit_at(flat: str, qubit_idx: int) -> int:
        return int(flat[total_qubits - 1 - qubit_idx])

    q_w = q + 1
    for state, count in counts.items():
        flat = state.replace(" ", "")
        col = sum(bit_at(flat, pos[i]) << i for i in range(n))
        row = sum(bit_at(flat, pos[n + i]) << i for i in range(n))
        num_unsigned = sum(bit_at(flat, num_q[i]) << i for i in range(q_w))
        # Convert two's-complement (q+1)-bit to signed int
        num_signed = num_unsigned - (1 << q_w) if num_unsigned >> q else num_unsigned
        den_val = sum(bit_at(flat, den_q[i]) << i for i in range(q_w))
        histograms_num.setdefault((row, col), {})
        histograms_num[(row, col)][num_signed] = (
            histograms_num[(row, col)].get(num_signed, 0) + count
        )
        histograms_den.setdefault((row, col), {})
        histograms_den[(row, col)][den_val] = histograms_den[(row, col)].get(den_val, 0) + count
    for (row, col), votes in histograms_num.items():
        num_img[row, col] = max(votes, key=votes.__getitem__)
    for (row, col), votes in histograms_den.items():
        den_img[row, col] = max(votes, key=votes.__getitem__)
    return num_img, den_img


# NOTE: The full Class A pipeline (sub + add + conditional-negate +
# fractional-scale + divide) requires registers totalling ~40 qubits
# at n=1, q=2, putting it past laptop statevector range. We expose the
# prefix stage above as the bit-exact-testable component; the
# divider step will be added in a follow-up commit and validated via
# the MPS simulator on AerSimulator(method='mps').


# ============================================================================
# Class C — roGFP calibrated ratio (affine reparametrisation primitive)
# ============================================================================


def affine_subtract_constant(
    qc: QuantumCircuit,
    R_qubits: list[int],
    c_value: int,
    output_signed_qubits: list[int],
    c_const_register_qubits: list[int],
    sub_carry_qubits: list[int],
) -> None:
    """Compute ``output_signed = R - c_value`` (signed two's-complement,
    width q + 1) where ``c_value`` is a classical non-negative integer
    and ``R`` is a ``q``-bit unsigned quantum register.

    The trick is to materialise the classical constant ``c_value`` into
    an ancilla register via X gates, run the standard ripple-carry
    subtractor, then X-unset the ancilla. The ancilla returns to ``|0⟩``
    on exit (provided it was ``|0⟩`` on entry).

    Parameters
    ----------
    R_qubits
        ``q``-bit unsigned input register
        preserved on exit.
    c_value
        Classical non-negative integer in ``[0, 2^(q+1) - 1]``.
    output_signed_qubits
        ``q + 1``-bit register, ``|0⟩`` on entry
        holds
        ``(R - c_value) mod 2^(q+1)`` on exit (MSB is the sign bit).
    c_const_register_qubits
        ``q + 1``-bit ancilla, ``|0⟩`` on entry and exit. Used to hold
        ``c_value`` temporarily during the subtract.
    sub_carry_qubits
        ``q + 2``-bit carry register for the subtractor, ``|0⟩`` on
        entry
        dirty on exit (carries from the subtract).

    Notes
    -----
    Cost: 2 · popcount(c_value) X gates + one ``q_sub`` of width
    ``q + 1`` ≈ ``4 (q + 1)`` Toffolis.

    This is the building block for the roGFP affine reparametrisation
    ``R_C = (R - R_red) / (R_ox - R_red)``. The division by the
    classical constant ``R_ox - R_red`` is naturally implemented
    downstream as a ``q_mul_const`` with ``k = 2^k_frac / (R_ox - R_red)``
    in fixed-point — see the recipe at the bottom of this module for
    the full composition.

    Composition example (Class B → affine subtract, autonomous Class C
    without the optional 1/(R_ox - R_red) rescaling)::

        from qiskit import QuantumRegister
        from qimp.processing.ratiometric_circuit import (
            class_b_ratio, affine_subtract_constant,
        )

        # Stage 1: build Class B (autonomous integer ratio R = I_a // I_b).
        qc, layout = class_b_ratio(image_a, image_b, q=2)

        # Stage 2: append Class C registers and apply affine subtract.
        start = qc.num_qubits
        q_w = q + 1
        R_C_reg = QuantumRegister(q_w, "R_C_signed")
        R_red_reg = QuantumRegister(q_w, "R_red_const")
        sub_c_reg = QuantumRegister(q_w + 1, "sub_carry_C")
        qc.add_register(R_C_reg, R_red_reg, sub_c_reg)

        R_C_idx     = list(range(start,             start + q_w))
        R_red_idx   = list(range(start + q_w,       start + 2 * q_w))
        sub_c_idx   = list(range(start + 2 * q_w,   start + 2 * q_w + q_w + 1))

        affine_subtract_constant(
            qc,
            R_qubits=layout["quotient"],   # = R from Class B
            c_value=R_red,
            output_signed_qubits=R_C_idx,
            c_const_register_qubits=R_red_idx,
            sub_carry_qubits=sub_c_idx,
        )

        # `R_C_idx` now holds (R - R_red) signed two's-complement.
        # The classical decoder recovers R_C(p) = signed_value(p) / (R_ox - R_red).

    For an in-circuit /(R_ox - R_red) rescaling, append a ``q_mul_const``
    with ``k_bits`` encoding ``2^k_frac / (R_ox - R_red)`` as a fixed-point
    integer
    the multiply consumes ``R_C_idx`` as its ``b`` argument and
    writes the scaled result to a fresh accumulator.

    At n=1, q=2 the whole pipeline costs ~35 qubits, past laptop
    statevector range
    component correctness is established by the
    bit-exact unit tests of each primitive (`q_div_restoring`,
    `affine_subtract_constant`).
    """
    q = len(R_qubits)
    q_w = q + 1
    if len(output_signed_qubits) != q_w:
        raise ValueError(
            f"output_signed_qubits must have q+1={q_w} bits, got {len(output_signed_qubits)}"
        )
    if len(c_const_register_qubits) != q_w:
        raise ValueError(
            f"c_const_register_qubits must have q+1={q_w} bits, got {len(c_const_register_qubits)}"
        )
    if len(sub_carry_qubits) != q_w + 1:
        raise ValueError(
            f"sub_carry_qubits must have q+2={q_w + 1} bits, got {len(sub_carry_qubits)}"
        )
    if c_value < 0 or c_value >= (1 << q_w):
        raise ValueError(f"c_value={c_value} out of range [0, 2^(q+1) - 1] = [0, {(1 << q_w) - 1}]")

    # Copy R (q-bit) into the low q bits of output_signed (q+1-bit).
    # MSB stays 0 — this is the zero-extension that turns the unsigned
    # value into a positive signed value before the subtract.
    for i in range(q):
        qc.cx(R_qubits[i], output_signed_qubits[i])

    # Materialise c_value into c_const_register via X gates on the
    # bit positions where c_value has a 1.
    for i in range(q_w):
        if (c_value >> i) & 1:
            qc.x(c_const_register_qubits[i])

    # output_signed ← output_signed - c_const_register (q+1-bit width).
    from qimp.processing.arithmetic import q_sub as _q_sub  # local to avoid cycle

    _q_sub(qc, c_const_register_qubits, output_signed_qubits, sub_carry_qubits)

    # X-unset c_const_register back to |0⟩.
    for i in range(q_w):
        if (c_value >> i) & 1:
            qc.x(c_const_register_qubits[i])


# ----- Class C top-level composition (Class B → affine subtract) ---------
#
# The full autonomous Class C circuit is the composition
#
#     dual_neqr_load
#     → q_div_restoring     (Class B integer ratio R = I_a // I_b)
#     → affine_subtract_constant  (R - R_red, signed q+1 bits)
#     → (optional) q_mul_const  (× 2^k / (R_ox - R_red), fixed-point)
#
# At n=1, q=2 the layout would total ~35 qubits — past statevector
# range for the typical 16 GB laptop. We therefore do NOT package the
# composition as a single function: instead we provide
# `affine_subtract_constant` as a bit-exact-testable primitive, and
# leave it to the caller to chain it with `class_b_ratio`'s output
# quotient register inside a larger QuantumCircuit. See
# `test_affine_subtract_constant` for an isolated bit-exact check;
# the chained composition is straightforward and not separately tested
# under statevector.


def class_a_gp_full(
    image_a: np.ndarray,
    image_b: np.ndarray,
    q: int,
    q_frac: int = 2,
) -> tuple[QuantumCircuit, dict[str, list[int] | int]]:
    """Full autonomous Class A GP circuit with G = 1 (Möbius normalised
    difference).

    For each pixel in superposition, the circuit computes:

        magnitude(p) = floor(|I_a(p) − I_b(p)| · 2^q_frac / (I_a(p) + I_b(p)))
        sign(p)      = 1  iff  I_b(p) > I_a(p)

    The classical GP is reconstructed by the decoder as
    ``gp(p) = (-1)^sign(p) · magnitude(p) / 2^q_frac``.

    Pipeline (composition of already-tested primitives):
      1. ``dual_neqr_load(I_a, I_b)``
      2. ``den ← I_a + I_b``  (q_w = q+1 bits)
      3. ``num ← I_a − I_b``  (signed q_w bits, two's complement)
      4. Copy num's MSB into a dedicated sign qubit.
      5. Conditional two's-complement negate of num: flip all bits
         (CNOT(sign, num[i])) then conditional ``+1`` via
         ``q_add_ctrl`` against a 1-valued constant register. Result:
         num holds |I_a − I_b| as an unsigned q_w-bit integer.
      6. Bit-shift num up by ``q_frac`` into a wider ``num_scaled``
         register (width q_w + q_frac).
      7. ``q_div_general(num_scaled, den)`` → quotient holds the
         fractional GP magnitude.

    Resource budget at n=1, q=2, q_frac=2: ~70 qubits — past laptop
    statevector. Use ``AerSimulator(method='mps')`` for execution.
    Each underlying primitive is bit-exact verified at small width;
    end-to-end correctness relies on the composition (every primitive
    is position-independent so the action lifts uniformly across the
    NEQR position superposition).

    Returns
    -------
    (qc, layout) with keys: ``position``, ``I_a``, ``I_b``, ``num``,
    ``den``, ``add_c``, ``sub_c``, ``sign``, ``ones_reg``,
    ``inc_carry``, ``num_scaled``, ``quotient``, ``div_work``,
    ``div_pad``, ``div_c``, ``div_flag``.
    """
    n = _validate_dual_images(image_a, image_b, q)
    if q_frac < 0:
        raise ValueError(f"q_frac must be >= 0, got {q_frac}")
    n_pos = 2 * n
    q_w = q + 1
    q_dividend = q_w + q_frac
    needed_div_c = (q_dividend + 1) * (q_w + 2)

    layout: dict[str, list[int] | int] = {}
    cursor = 0
    layout["position"] = list(range(cursor, cursor + n_pos))
    cursor += n_pos
    layout["I_a"] = list(range(cursor, cursor + q))
    cursor += q
    layout["I_b"] = list(range(cursor, cursor + q))
    cursor += q
    layout["num"] = list(range(cursor, cursor + q_w))
    cursor += q_w
    layout["den"] = list(range(cursor, cursor + q_w))
    cursor += q_w
    layout["add_c"] = list(range(cursor, cursor + q_w))
    cursor += q_w
    layout["sub_c"] = list(range(cursor, cursor + q_w))
    cursor += q_w
    layout["sign"] = cursor
    cursor += 1
    layout["ones_reg"] = list(range(cursor, cursor + q_w))
    cursor += q_w
    layout["inc_carry"] = list(range(cursor, cursor + q_w + 1))
    cursor += q_w + 1
    layout["num_scaled"] = list(range(cursor, cursor + q_dividend))
    cursor += q_dividend
    layout["quotient"] = list(range(cursor, cursor + q_dividend))
    cursor += q_dividend
    layout["div_work"] = list(range(cursor, cursor + q_w))
    cursor += q_w
    layout["div_pad"] = cursor
    cursor += 1
    layout["div_c"] = list(range(cursor, cursor + needed_div_c))
    cursor += needed_div_c
    layout["div_flag"] = cursor
    cursor += 1

    total = cursor
    qc = QuantumCircuit(total)

    # Helper refs (typed)
    Ia = layout["I_a"]
    Ib = layout["I_b"]
    num = layout["num"]
    den = layout["den"]
    add_c = layout["add_c"]
    sub_c = layout["sub_c"]
    sign = layout["sign"]
    ones_reg = layout["ones_reg"]
    inc_carry = layout["inc_carry"]
    num_scaled = layout["num_scaled"]
    quotient = layout["quotient"]
    div_work = layout["div_work"]
    div_pad = layout["div_pad"]
    div_c = layout["div_c"]
    div_flag = layout["div_flag"]
    assert isinstance(Ia, list) and isinstance(Ib, list)
    assert isinstance(num, list) and isinstance(den, list)
    assert isinstance(add_c, list) and isinstance(sub_c, list)
    assert isinstance(sign, int)
    assert isinstance(ones_reg, list) and isinstance(inc_carry, list)
    assert isinstance(num_scaled, list)
    assert isinstance(quotient, list)
    assert isinstance(div_work, list) and isinstance(div_pad, int)
    assert isinstance(div_c, list) and isinstance(div_flag, int)

    # 1. NEQR-encode both images.
    dual_neqr_load(
        qc,
        image_a,
        image_b,
        q,
        position_qubits=layout["position"],  # type: ignore[arg-type]
        intensity_a_qubits=Ia,
        intensity_b_qubits=Ib,
    )

    # 2. den ← I_a + I_b (q+1 bits, carry into den[q]).
    _copy_register(qc, Ia, den[:q])
    q_add(qc, Ib, list(den[:q]), add_c)
    qc.cx(add_c[-1], den[q])

    # 3. num ← I_a − I_b (two's complement, q+1 bits; MSB = sign bit).
    _copy_register(qc, Ia, num[:q])
    q_sub(qc, Ib, list(num[:q]), sub_c)
    qc.cx(sub_c[-1], num[q])
    qc.x(num[q])

    # 4. Copy sign bit out.
    qc.cx(num[q], sign)

    # 5. Conditional two's-complement negate: |num|.
    for i in range(q_w):
        qc.cx(sign, num[i])
    qc.x(ones_reg[0])  # ones_reg = 1
    q_add_ctrl(qc, sign, ones_reg, num, inc_carry)
    # ones_reg stays at 1 throughout the rest of the circuit; will be
    # uncomputed only by a full-pipeline inverse if needed.

    # 6. Bit-shift num up by q_frac → num_scaled (CNOT copy).
    for i in range(q_w):
        qc.cx(num[i], num_scaled[i + q_frac])

    # 7. Divide: quotient = num_scaled // den using non-square divider.
    q_div_general(
        qc,
        dividend_qubits=num_scaled,
        divisor_qubits=den,
        quotient_qubits=quotient,
        work_qubits=div_work,
        divisor_pad_qubit=div_pad,
        c_qubits=div_c,
        div_zero_flag=div_flag,
    )
    return qc, layout


def decode_class_a_full(
    counts: dict[str, int],
    n: int,
    q: int,
    q_frac: int,
    layout: dict[str, list[int] | int],
    total_qubits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode Class A full counts into (gp_signed_image, divzero_mask).

    Reads the ``sign`` qubit, the **full** ``quotient`` register, and the
    ``div_flag``. The magnitude must be read across the whole register,
    not just its low ``q_frac`` bits: since ``|I_a - I_b| <= I_a + I_b``
    the quotient is bounded by ``2**q_frac``, and it *attains* that value
    at the saturating ``GP = ±1`` endpoints (one channel quantised to 0),
    which sets bit ``q_frac`` and would be truncated away by a low-bits
    read. Bits above ``q_frac`` are identically zero, so the full read is
    exact at every pixel.

    Returns
    -------
    (gp_image, divzero_mask)
        ``gp_image[r, c] = (-1)^sign * magnitude / 2^q_frac``,
        floating-point in [−1, 1]. ``divzero_mask[r, c]`` is True for
        pixels where the denominator (I_a + I_b) was 0.
    """
    side = 1 << n
    gp_img = np.zeros((side, side), dtype=np.float64)
    divzero_mask = np.zeros((side, side), dtype=bool)
    pos_qubits = layout["position"]
    sign_q = layout["sign"]
    quo_qubits = layout["quotient"]
    flag_qubit = layout["div_flag"]
    assert isinstance(pos_qubits, list)
    assert isinstance(quo_qubits, list)
    assert isinstance(sign_q, int)
    assert isinstance(flag_qubit, int)

    histograms: dict[tuple[int, int], dict[float, int]] = {}
    flag_counts: dict[tuple[int, int], dict[int, int]] = {}

    def bit_at(flat: str, qubit_idx: int) -> int:
        return int(flat[total_qubits - 1 - qubit_idx])

    for state, count in counts.items():
        flat = state.replace(" ", "")
        col = sum(bit_at(flat, pos_qubits[i]) << i for i in range(n))
        row = sum(bit_at(flat, pos_qubits[n + i]) << i for i in range(n))
        mag = sum(bit_at(flat, quo_qubits[i]) << i for i in range(len(quo_qubits)))
        s = bit_at(flat, sign_q)
        f = bit_at(flat, flag_qubit)
        gp_val = (-1.0 if s else 1.0) * mag / float(1 << q_frac)
        histograms.setdefault((row, col), {})
        histograms[(row, col)][gp_val] = histograms[(row, col)].get(gp_val, 0) + count
        flag_counts.setdefault((row, col), {})
        flag_counts[(row, col)][f] = flag_counts[(row, col)].get(f, 0) + count

    for (row, col), votes in histograms.items():
        gp_img[row, col] = max(votes, key=votes.__getitem__)
    for (row, col), flag_votes in flag_counts.items():
        divzero_mask[row, col] = max(flag_votes, key=flag_votes.__getitem__) == 1
    return gp_img, divzero_mask


def class_c_rogfp_full(
    image_a: np.ndarray,
    image_b: np.ndarray,
    q: int,
    q_frac: int,
    R_red_fp: int,
) -> tuple[QuantumCircuit, dict[str, list[int] | int]]:
    """Full autonomous Class C roGFP calibrated-ratio circuit.

    For each pixel in superposition the circuit computes, entirely on the
    quantum device, the fixed-point quantities

        ratio(p)    = floor(I_a(p) * 2^q_frac / I_b(p))      (fractional R)
        rc_signed(p)= ratio(p) - R_red_fp                    (signed)

    where ``R_red_fp = round(R_red * 2^q_frac)`` is the classical reduced
    reference in fixed point. The classical decoder finishes the affine
    reparametrisation with the single classical calibration scalar

        R_C(p) = rc_signed(p) / ((R_ox - R_red) * 2^q_frac)  in [0, 1]

    — exactly as a wet-lab roGFP calibration applies (R_ox - R_red), and
    in direct parallel with the ``/2^q_frac`` scalar of the Class-A
    decoder. The fractional ratio (shift by ``q_frac`` then non-square
    divide) is what lifts the integer-quotient degeneracy that makes
    Class B uninformative for roGFP at small ``q`` (the sub-unit
    physiological F405/F488 ratio).

    Pipeline (composition of already-tested primitives):
      1. ``dual_neqr_load(I_a, I_b)``
      2. Bit-shift I_a up by ``q_frac`` into ``num_scaled`` (CNOT copy).
      3. ``q_div_general(num_scaled, I_b)`` → ``ratio`` holds the
         fixed-point fractional ratio
         div-zero flag set iff I_b == 0.
      4. ``affine_subtract_constant(ratio, R_red_fp)`` → ``rc_signed``
         holds ``ratio - R_red_fp`` (signed two's-complement).

    Resource budget at n=1, q=4, q_frac=4: ~114 qubits — past laptop
    statevector. Use ``AerSimulator(method='mps')``
    the bond dimension
    stays small because the position register superposes only ``4^n``
    pixel branches.

    Returns
    -------
    (qc, layout) with keys: ``position``, ``I_a``, ``I_b``,
    ``num_scaled``, ``ratio``, ``div_work``, ``div_pad``, ``div_c``,
    ``div_flag``, ``rc_signed``, ``rc_const``, ``rc_carry``.
    """
    n = _validate_dual_images(image_a, image_b, q)
    if q_frac < 0:
        raise ValueError(f"q_frac must be >= 0, got {q_frac}")
    n_pos = 2 * n
    w = q + q_frac  # dividend / quotient width
    needed_div_c = (w + 1) * (q + 2)
    rc_w = w + 1  # signed affine-output width

    layout: dict[str, list[int] | int] = {}
    cursor = 0
    layout["position"] = list(range(cursor, cursor + n_pos))
    cursor += n_pos
    layout["I_a"] = list(range(cursor, cursor + q))
    cursor += q
    layout["I_b"] = list(range(cursor, cursor + q))
    cursor += q
    layout["num_scaled"] = list(range(cursor, cursor + w))
    cursor += w
    layout["ratio"] = list(range(cursor, cursor + w))
    cursor += w
    layout["div_work"] = list(range(cursor, cursor + q))
    cursor += q
    layout["div_pad"] = cursor
    cursor += 1
    layout["div_c"] = list(range(cursor, cursor + needed_div_c))
    cursor += needed_div_c
    layout["div_flag"] = cursor
    cursor += 1
    layout["rc_signed"] = list(range(cursor, cursor + rc_w))
    cursor += rc_w
    layout["rc_const"] = list(range(cursor, cursor + rc_w))
    cursor += rc_w
    layout["rc_carry"] = list(range(cursor, cursor + rc_w + 1))
    cursor += rc_w + 1

    total = cursor
    qc = QuantumCircuit(total)

    Ia = layout["I_a"]
    Ib = layout["I_b"]
    num_scaled = layout["num_scaled"]
    ratio = layout["ratio"]
    div_work = layout["div_work"]
    div_pad = layout["div_pad"]
    div_c = layout["div_c"]
    div_flag = layout["div_flag"]
    rc_signed = layout["rc_signed"]
    rc_const = layout["rc_const"]
    rc_carry = layout["rc_carry"]
    assert isinstance(Ia, list) and isinstance(Ib, list)
    assert isinstance(num_scaled, list) and isinstance(ratio, list)
    assert isinstance(div_work, list) and isinstance(div_pad, int)
    assert isinstance(div_c, list) and isinstance(div_flag, int)
    assert isinstance(rc_signed, list) and isinstance(rc_const, list)
    assert isinstance(rc_carry, list)

    # 1. NEQR-encode both channels.
    dual_neqr_load(
        qc,
        image_a,
        image_b,
        q,
        position_qubits=layout["position"],  # type: ignore[arg-type]
        intensity_a_qubits=Ia,
        intensity_b_qubits=Ib,
    )

    # 2. num_scaled ← I_a << q_frac (so the quotient carries q_frac
    #    fractional bits of the ratio I_a / I_b).
    for i in range(q):
        qc.cx(Ia[i], num_scaled[i + q_frac])

    # 3. ratio = num_scaled // I_b (non-square divider); div-zero flag.
    q_div_general(
        qc,
        dividend_qubits=num_scaled,
        divisor_qubits=Ib,
        quotient_qubits=ratio,
        work_qubits=div_work,
        divisor_pad_qubit=div_pad,
        c_qubits=div_c,
        div_zero_flag=div_flag,
    )

    # 4. rc_signed = ratio - R_red_fp (signed two's-complement).
    affine_subtract_constant(
        qc,
        R_qubits=ratio,
        c_value=R_red_fp,
        output_signed_qubits=rc_signed,
        c_const_register_qubits=rc_const,
        sub_carry_qubits=rc_carry,
    )
    return qc, layout


def decode_class_c_rogfp(
    counts: dict[str, int],
    n: int,
    q: int,
    q_frac: int,
    R_red: float,
    R_ox: float,
    layout: dict[str, list[int] | int],
    total_qubits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode Class C counts into (R_C_image, divzero_mask).

    Reads the signed ``rc_signed`` register (= ratio - R_red_fp in
    fixed point) and the div-zero flag, then applies the single classical
    calibration scalar to recover the redox index
    ``R_C = rc_signed / ((R_ox - R_red) * 2^q_frac)``.
    """
    side = 1 << n
    rc_img = np.zeros((side, side), dtype=np.float64)
    divzero_mask = np.zeros((side, side), dtype=bool)
    pos_qubits = layout["position"]
    rc_qubits = layout["rc_signed"]
    flag_qubit = layout["div_flag"]
    assert isinstance(pos_qubits, list)
    assert isinstance(rc_qubits, list)
    assert isinstance(flag_qubit, int)
    rc_w = len(rc_qubits)
    denom = (R_ox - R_red) * float(1 << q_frac)

    def bit_at(flat: str, qubit_idx: int) -> int:
        return int(flat[total_qubits - 1 - qubit_idx])

    histograms: dict[tuple[int, int], dict[float, int]] = {}
    flag_counts: dict[tuple[int, int], dict[int, int]] = {}
    for state, count in counts.items():
        flat = state.replace(" ", "")
        col = sum(bit_at(flat, pos_qubits[i]) << i for i in range(n))
        row = sum(bit_at(flat, pos_qubits[n + i]) << i for i in range(n))
        raw = sum(bit_at(flat, rc_qubits[i]) << i for i in range(rc_w))
        if raw >= (1 << (rc_w - 1)):  # two's-complement sign extend
            raw -= 1 << rc_w
        rc_val = raw / denom
        f = bit_at(flat, flag_qubit)
        histograms.setdefault((row, col), {})
        histograms[(row, col)][rc_val] = histograms[(row, col)].get(rc_val, 0) + count
        flag_counts.setdefault((row, col), {})
        flag_counts[(row, col)][f] = flag_counts[(row, col)].get(f, 0) + count

    for (row, col), votes in histograms.items():
        rc_img[row, col] = max(votes, key=votes.__getitem__)
    for (row, col), flag_votes in flag_counts.items():
        divzero_mask[row, col] = max(flag_votes, key=flag_votes.__getitem__) == 1
    return rc_img, divzero_mask
