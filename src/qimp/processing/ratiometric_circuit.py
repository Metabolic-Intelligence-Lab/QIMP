"""Autonomous quantum circuits for per-pixel ratiometric image operators.

Each operator builder takes two NEQR-encoded intensity images and
constructs a circuit that **computes** the ratio (or normalised
difference, or calibrated ratio) on the quantum device — no
classically pre-computed target is loaded into the circuit. The
intended use of these circuits is twofold: (1) as a state-prep oracle
for downstream QAE / Grover quantum-advantage demonstrations on
aggregate image statistics (planned, see project plan); (2) as a
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
    Planned: :func:`class_c_rogfp`.

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
    q_div_restoring,
    q_sub,
)

__all__ = [
    "affine_subtract_constant",
    "class_a_gp_prefix",
    "class_b_ratio",
    "decode_class_a_prefix",
    "decode_class_b_ratio",
    "dual_neqr_load",
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
        raise ValueError(
            f"max intensity exceeds 2^q - 1 = {max_int}; increase q or rescale"
        )
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
        raise ValueError(
            f"position_qubits must have 2n={2 * n} bits, got {len(position_qubits)}"
        )
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
                _encode_intensity_at_pixel(
                    qc, position_qubits, intensity_a_qubits, n, ia, row, col
                )
            if ib != 0:
                _encode_intensity_at_pixel(
                    qc, position_qubits, intensity_b_qubits, n, ib, row, col
                )
            qc.barrier()


def class_b_ratio(image_a: np.ndarray, image_b: np.ndarray, q: int) -> tuple[QuantumCircuit, dict[str, list[int] | int]]:
    """Build an autonomous Class-B integer-quotient ratio circuit.

    For each pixel position p in superposition, the circuit computes
    R(p) = I_a(p) // I_b(p) using reversible restoring division (no
    classical pre-computation of R).

    Returns
    -------
    (qc, layout)
        ``qc`` is the assembled QuantumCircuit; ``layout`` is a dict
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
    n = _validate_dual_images(image_a, image_b, q)
    n_pos = 2 * n
    n_c = (q + 1) * (q + 2)

    # Qubit-index layout (allocate in order):
    layout: dict[str, list[int] | int] = {}
    cursor = 0
    layout["position"] = list(range(cursor, cursor + n_pos)); cursor += n_pos
    layout["I_a"] = list(range(cursor, cursor + q)); cursor += q
    layout["I_b"] = list(range(cursor, cursor + q)); cursor += q
    layout["quotient"] = list(range(cursor, cursor + q)); cursor += q
    layout["work"] = list(range(cursor, cursor + q)); cursor += q
    layout["pad"] = cursor; cursor += 1
    layout["c"] = list(range(cursor, cursor + n_c)); cursor += n_c
    layout["flag"] = cursor; cursor += 1
    total = cursor

    qc = QuantumCircuit(total)

    dual_neqr_load(
        qc, image_a, image_b, q,
        position_qubits=layout["position"],
        intensity_a_qubits=layout["I_a"],
        intensity_b_qubits=layout["I_b"],
    )

    # Class B integer ratio: I_a // I_b. Divider overwrites I_a register
    # with the remainder; I_b is preserved.
    q_div_restoring(
        qc,
        dividend_qubits=layout["I_a"],
        divisor_qubits=layout["I_b"],
        quotient_qubits=layout["quotient"],
        work_qubits=layout["work"],
        divisor_pad_qubit=layout["pad"],
        c_qubits=layout["c"],
        div_zero_flag=layout["flag"],
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

    for state, count in counts.items():
        flat = state.replace(" ", "")
        # Qiskit reports MSB → LSB, qubit i is at position total_qubits-1-i.
        def bit_at(qubit_idx: int) -> int:
            return int(flat[total_qubits - 1 - qubit_idx])

        # Position layout: low n bits are x (col), high n bits are y (row).
        col = sum(bit_at(pos_qubits[i]) << i for i in range(n))
        row = sum(bit_at(pos_qubits[n + i]) << i for i in range(n))
        quo = sum(bit_at(quo_qubits[i]) << i for i in range(q))
        flag = bit_at(flag_qubit)

        histograms.setdefault((row, col), {})
        histograms[(row, col)][quo] = histograms[(row, col)].get(quo, 0) + count
        flag_counts.setdefault((row, col), {})
        flag_counts[(row, col)][flag] = flag_counts[(row, col)].get(flag, 0) + count

    for (row, col), votes in histograms.items():
        quotient_img[row, col] = max(votes, key=votes.__getitem__)
    for (row, col), votes in flag_counts.items():
        divzero_mask[row, col] = max(votes, key=votes.__getitem__) == 1
    return quotient_img, divzero_mask


# ============================================================================
# Class A — Generalised Polarization (G = 1 Möbius form)
# ============================================================================


def _copy_register(qc: QuantumCircuit, src: list[int], dst: list[int]) -> None:
    """In-place classical copy via CNOTs: dst ← dst ⊕ src.

    For basis-state inputs this is a true copy; in superposition it
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

    This is the first half of the full GP pipeline; the second half
    (conditional |·| then fractional divider) is currently
    statevector-infeasible and pending the non-square ``q_div`` variant
    — see Stage C note at the end of this module. The split exists for
    two reasons: (1) the prefix is bit-exact-testable at small (n, q)
    while the full pipeline is not; (2) the divider step uses a
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
    layout["position"] = list(range(cursor, cursor + n_pos)); cursor += n_pos
    layout["I_a"] = list(range(cursor, cursor + q)); cursor += q
    layout["I_b"] = list(range(cursor, cursor + q)); cursor += q
    layout["num"] = list(range(cursor, cursor + q_w)); cursor += q_w
    layout["den"] = list(range(cursor, cursor + q_w)); cursor += q_w
    layout["add_c"] = list(range(cursor, cursor + q_w)); cursor += q_w  # n+1 carries for the q-wide add (n = q)
    layout["sub_c"] = list(range(cursor, cursor + q_w)); cursor += q_w  # n+1 carries for the q-wide sub
    total = cursor

    qc = QuantumCircuit(total)

    dual_neqr_load(
        qc, image_a, image_b, q,
        position_qubits=layout["position"],
        intensity_a_qubits=layout["I_a"],
        intensity_b_qubits=layout["I_b"],
    )

    Ia = layout["I_a"]; Ib = layout["I_b"]
    num = layout["num"]; den = layout["den"]
    add_c = layout["add_c"]; sub_c = layout["sub_c"]
    assert isinstance(Ia, list) and isinstance(Ib, list)
    assert isinstance(num, list) and isinstance(den, list)
    assert isinstance(add_c, list) and isinstance(sub_c, list)

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
        if num_unsigned >> q:
            num_signed = num_unsigned - (1 << q_w)
        else:
            num_signed = num_unsigned
        den_val = sum(bit_at(flat, den_q[i]) << i for i in range(q_w))
        histograms_num.setdefault((row, col), {})
        histograms_num[(row, col)][num_signed] = (
            histograms_num[(row, col)].get(num_signed, 0) + count
        )
        histograms_den.setdefault((row, col), {})
        histograms_den[(row, col)][den_val] = (
            histograms_den[(row, col)].get(den_val, 0) + count
        )
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
        ``q``-bit unsigned input register; preserved on exit.
    c_value
        Classical non-negative integer in ``[0, 2^(q+1) - 1]``.
    output_signed_qubits
        ``q + 1``-bit register, ``|0⟩`` on entry; holds
        ``(R - c_value) mod 2^(q+1)`` on exit (MSB is the sign bit).
    c_const_register_qubits
        ``q + 1``-bit ancilla, ``|0⟩`` on entry and exit. Used to hold
        ``c_value`` temporarily during the subtract.
    sub_carry_qubits
        ``q + 2``-bit carry register for the subtractor, ``|0⟩`` on
        entry; dirty on exit (carries from the subtract).

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
    integer; the multiply consumes ``R_C_idx`` as its ``b`` argument and
    writes the scaled result to a fresh accumulator.

    At n=1, q=2 the whole pipeline costs ~35 qubits, past laptop
    statevector range; component correctness is established by the
    bit-exact unit tests of each primitive (`q_div_restoring`,
    `affine_subtract_constant`).
    """
    q = len(R_qubits)
    q_w = q + 1
    if len(output_signed_qubits) != q_w:
        raise ValueError(
            f"output_signed_qubits must have q+1={q_w} bits, got "
            f"{len(output_signed_qubits)}"
        )
    if len(c_const_register_qubits) != q_w:
        raise ValueError(
            f"c_const_register_qubits must have q+1={q_w} bits, got "
            f"{len(c_const_register_qubits)}"
        )
    if len(sub_carry_qubits) != q_w + 1:
        raise ValueError(
            f"sub_carry_qubits must have q+2={q_w + 1} bits, got "
            f"{len(sub_carry_qubits)}"
        )
    if c_value < 0 or c_value >= (1 << q_w):
        raise ValueError(
            f"c_value={c_value} out of range [0, 2^(q+1) - 1] = "
            f"[0, {(1 << q_w) - 1}]"
        )

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
