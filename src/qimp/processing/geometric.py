"""Geometric transformations on encoded quantum images.

These operations manipulate **position qubits only** and therefore apply
identically to FRQI, NEQR, and QPIE. The caller supplies ``n`` (qubits per
spatial axis) and an offset that points to the first position qubit in the
target circuit:

============  =================================
Encoding      Position-qubit offset
============  =================================
FRQI          0  (qubits 0..2n-1)
QPIE          0  (qubits 0..2n-1)
NEQR          q  (qubits q..q+2n-1, with q the intensity-qubit count)
============  =================================

Qubit layout within the position block (LSB-first):
- ``pos_offset .. pos_offset + n - 1``  → X-axis qubits (column, x_0 … x_{n-1})
- ``pos_offset + n .. pos_offset + 2n - 1`` → Y-axis qubits (row, y_0 … y_{n-1})

Reference: docs/tesi.pdf §2.3
"""

from __future__ import annotations

from typing import Literal

from qiskit import QuantumCircuit

from qimp.runtime.arithmetic_gates import cyclic_decrement, cyclic_increment

Axis = Literal["x", "y"]

__all__ = [
    "axis_flip",
    "coord_swap",
    "ort_rotation",
    "pos_shift",
    "restr_coord_swap",
    "restr_flip",
]


def _x_qubits(n: int, pos_offset: int) -> list[int]:
    return list(range(pos_offset, pos_offset + n))


def _y_qubits(n: int, pos_offset: int) -> list[int]:
    return list(range(pos_offset + n, pos_offset + 2 * n))


def _validate_position_register(qc: QuantumCircuit, n: int, pos_offset: int) -> None:
    """Verify ``[pos_offset, pos_offset + 2n)`` fits inside the circuit.

    Raises a `ValueError` rather than producing wrong gates on a malformed
    invocation (e.g. forgetting `pos_offset=q` for an NEQR circuit).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if pos_offset < 0:
        raise ValueError(f"pos_offset must be >= 0, got {pos_offset}")
    end = pos_offset + 2 * n
    if end > qc.num_qubits:
        raise ValueError(
            f"position register [{pos_offset}, {end}) does not fit in circuit "
            f"with {qc.num_qubits} qubits"
        )


def axis_flip(qc: QuantumCircuit, n: int, axis: Axis, *, pos_offset: int = 0) -> QuantumCircuit:
    """Flip the image about the given axis.

    ``axis='x'`` flips about the X axis (inverts Y / row coordinate) by applying
    X gates to the Y-axis qubits. ``axis='y'`` flips about the Y axis (inverts
    X / column coordinate) by applying X gates to the X-axis qubits.

    Complexity: ``O(n)`` X gates.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    _validate_position_register(qc, n, pos_offset)
    targets = _y_qubits(n, pos_offset) if axis == "x" else _x_qubits(n, pos_offset)
    for q in targets:
        qc.x(q)
    return qc


def coord_swap(qc: QuantumCircuit, n: int, *, pos_offset: int = 0) -> QuantumCircuit:
    """Swap row and column coordinates: image transpose.

    Applies ``n`` SWAP gates between paired position qubits ``(x_i, y_i)``.
    Complexity: ``O(n)`` SWAPs ≡ ``3n`` CNOTs.
    """
    _validate_position_register(qc, n, pos_offset)
    for i in range(n):
        qc.swap(pos_offset + i, pos_offset + n + i)
    return qc


def ort_rotation(qc: QuantumCircuit, n: int, angle: int, *, pos_offset: int = 0) -> QuantumCircuit:
    """Rotate the image by 90°, 180°, or 270° counter-clockwise.

    Decomposition into flip + swap:

    - 90°:  coord_swap then axis_flip('y')   (swap rows/cols, then invert cols)
    - 180°: axis_flip('x') then axis_flip('y')
    - 270°: coord_swap then axis_flip('x')

    Reference: docs/tesi.pdf §2.3.3
    """
    _validate_position_register(qc, n, pos_offset)
    if angle == 90:
        coord_swap(qc, n, pos_offset=pos_offset)
        axis_flip(qc, n, "y", pos_offset=pos_offset)
    elif angle == 180:
        axis_flip(qc, n, "x", pos_offset=pos_offset)
        axis_flip(qc, n, "y", pos_offset=pos_offset)
    elif angle == 270:
        coord_swap(qc, n, pos_offset=pos_offset)
        axis_flip(qc, n, "x", pos_offset=pos_offset)
    else:
        raise ValueError(f"angle must be 90, 180, or 270 (degrees), got {angle}")
    return qc


def pos_shift(
    qc: QuantumCircuit,
    n: int,
    axis: Axis,
    *,
    direction: Literal["+", "-"] = "+",
    magnitude: int = 1,
    pos_offset: int = 0,
) -> QuantumCircuit:
    """Shift the image by ``magnitude`` pixels along ``axis``.

    Implemented as a cyclic increment / decrement on the n-qubit position
    register for that axis. Each cyclic step is a cascade of multi-controlled X
    gates of decreasing arity (Toffoli-tower), giving ``O(n²)`` gate complexity.

    Reference: docs/tesi.pdf §2.3.5
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    if magnitude < 0:
        raise ValueError("magnitude must be >= 0")
    if direction not in ("+", "-"):
        raise ValueError("direction must be '+' or '-'")
    _validate_position_register(qc, n, pos_offset)
    axis_qubits = _y_qubits(n, pos_offset) if axis == "y" else _x_qubits(n, pos_offset)

    for _ in range(magnitude):
        if direction == "+":
            cyclic_increment(qc, axis_qubits)
        else:
            cyclic_decrement(qc, axis_qubits)
    return qc


def restr_flip(
    qc: QuantumCircuit,
    n: int,
    axis: Axis,
    region_bits: str,
    *,
    pos_offset: int = 0,
) -> QuantumCircuit:
    """Flip only the pixels whose position prefix matches ``region_bits``.

    Each character of ``region_bits`` selects a higher-order qubit of the axis
    being preserved (the axis *not* being flipped). For example, with n=4 and
    ``region_bits="10"`` on axis='y': flip the column coordinate only for rows
    whose two MSBs are ``10`` (i.e. rows 8..11).

    Implemented as a multi-controlled X on each X-axis qubit, with controls on
    the higher-order qubits of the orthogonal axis taking the prefix as a
    bitmask.

    Reference: docs/tesi.pdf §2.3.4
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    _validate_position_register(qc, n, pos_offset)
    if axis == "x":
        targets = _y_qubits(n, pos_offset)
        ctrl_register = _x_qubits(n, pos_offset)
    else:
        targets = _x_qubits(n, pos_offset)
        ctrl_register = _y_qubits(n, pos_offset)

    if len(region_bits) > n or any(c not in "01" for c in region_bits):
        raise ValueError(
            f"region_bits must be at most {n} characters of '0'/'1', got {region_bits!r}"
        )

    # Use the highest-order |region_bits|-many qubits of the orthogonal axis as
    # controls. ctrl_register is LSB-first so the high-order qubits are last.
    high_ctrl_count = len(region_bits)
    controls = ctrl_register[n - high_ctrl_count :]
    # region_bits is MSB-first; controls list above is LSB-first → align by reversing.
    region_bits_lsb = region_bits[::-1]
    flips = [controls[i] for i, b in enumerate(region_bits_lsb) if b == "0"]

    for q in flips:
        qc.x(q)
    for tgt in targets:
        qc.mcx(controls, tgt)
    for q in flips:
        qc.x(q)
    return qc


def restr_coord_swap(
    qc: QuantumCircuit,
    n: int,
    region_bits: str,
    *,
    pos_offset: int = 0,
) -> QuantumCircuit:
    """Coord-swap restricted to a region selected by ``region_bits``.

    Decomposition idea: for each ``i``, apply a controlled-SWAP between
    ``x_i`` and ``y_i`` conditioned on the region prefix. Implemented as
    ``CSWAP = CNOT(b,a) → Toffoli(ctrl..., a, b) → CNOT(b, a)`` per pair.

    Reference: docs/tesi.pdf §2.3.4
    """
    if len(region_bits) > n or any(c not in "01" for c in region_bits):
        raise ValueError(
            f"region_bits must be at most {n} characters of '0'/'1', got {region_bits!r}"
        )
    _validate_position_register(qc, n, pos_offset)
    x_qs = _x_qubits(n, pos_offset)
    y_qs = _y_qubits(n, pos_offset)
    # Use the highest-order |region_bits| qubits of each axis as joint controls.
    high_ctrl_count = len(region_bits)
    region_bits_lsb = region_bits[::-1]
    # Joint controls: high X-bits and high Y-bits.
    controls = x_qs[n - high_ctrl_count :] + y_qs[n - high_ctrl_count :]
    flips: list[int] = []
    for i, bit in enumerate(region_bits_lsb):
        if bit == "0":
            flips.append(x_qs[n - high_ctrl_count + i])
            flips.append(y_qs[n - high_ctrl_count + i])

    for qubit_idx in flips:
        qc.x(qubit_idx)
    for i in range(n - high_ctrl_count):
        x_q, y_q = x_qs[i], y_qs[i]
        qc.cx(y_q, x_q)
        qc.mcx([*controls, x_q], y_q)
        qc.cx(y_q, x_q)
    for qubit_idx in flips:
        qc.x(qubit_idx)
    return qc
