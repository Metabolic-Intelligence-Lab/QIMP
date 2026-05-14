"""Green-Purple (GP) ratio computation via a parameterized quantum circuit.

Application module (not part of the core thesis library). Computes
``(I_G - I_R) / (I_G + I_R)`` on stacked microscopy channels by encoding the
two channels as a multi-image FRQI circuit and using two CRY gates per pixel
to write the GP ratio onto an auxiliary color qubit. Parameters are tuned via
classical optimisation (scipy.minimize, COBYLA) against the classical GP image.

Migrated from legacy/scripts/gp_quantum_4.py + quantize_analyze_quantum_gp_v3.py.
Works for arbitrary `n` (spatial qubits) and `m` (image-selection qubits).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import CRYGate, HGate

__all__ = [
    "OptimizationResult",
    "apply_gp_function",
    "classical_gp_image",
    "combined_objective",
]


def classical_gp_image(image_g: np.ndarray, image_r: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Classical reference: ``(G - R) / (G + R + eps)`` element-wise.

    Output is in float64 and unclipped.
    """
    if image_g.shape != image_r.shape:
        raise ValueError(f"Channel shape mismatch: {image_g.shape} vs {image_r.shape}")
    g = image_g.astype(np.float64)
    r = image_r.astype(np.float64)
    return (g - r) / (g + r + eps)


def combined_objective(
    mse_value: float,
    psnr_value: float,
    tv_value: float,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> float:
    """Weighted scalar objective: ``α·MSE − β·PSNR + γ·TV``.

    Minimising this drives MSE and TV down while pushing PSNR up.
    """
    return alpha * mse_value - beta * psnr_value + gamma * tv_value


def apply_gp_function(
    qc: QuantumCircuit,
    n: int,
    m: int,
    params: Sequence[Parameter | float],
) -> QuantumCircuit:
    """Append the parameterised GP sub-circuit in place.

    For each of the 2^(2n) pixel positions, two parameters drive (a) a controlled
    RY operating between the image-selection qubit and the color qubit
    (difference channel), and (b) a controlled RY + Hadamard on the color qubit
    (normalised difference). The circuit assumes the standard FRQI layout:
    `n` row qubits, `n` column qubits, `m` selection qubits, 1 color qubit
    (color qubit index = 2n+m).

    Parameters
    ----------
    qc
        Circuit to modify. Must already encode the FRQI representation of two
        channels (so it has 2n+m+1 qubits).
    n
        Spatial qubits per axis (so image size = 2^n × 2^n).
    m
        Image-selection qubits (so number of stacked images = 2^m).
    params
        Sequence of 2 · 2^(2n) parameters. Built externally via
        ``[Parameter(f"θ{i}") for i in range(2 * 2**(2*n))]`` and assigned later
        with ``qc.assign_parameters(...)``.

    Returns
    -------
    QuantumCircuit
        The same `qc`, mutated in place and returned for chaining.
    """
    expected_params = 2 * (1 << (2 * n))
    if len(params) != expected_params:
        raise ValueError(f"expected {expected_params} parameters, got {len(params)}")
    if qc.num_qubits != 2 * n + m + 1:
        raise ValueError(f"qc has {qc.num_qubits} qubits, expected 2n+m+1 = {2 * n + m + 1}")

    selection_qubit = 2 * n  # first image-selection qubit (also acts as 'difference' control)
    color_qubit = 2 * n + m

    param_idx = 0
    for pixel_idx in range(1 << (2 * n)):
        # Position bits, LSB-first → qubits 0..2n-1, matching the FRQI encoder.
        pos_bits = format(pixel_idx, f"0{2 * n}b")[::-1]
        flips = [q for q, bit in enumerate(pos_bits) if bit == "0"]

        for q in flips:
            qc.x(q)

        # Difference operation: controlled RY rotates the color qubit.
        qc.cry(params[param_idx], selection_qubit, color_qubit)
        param_idx += 1

        # Normalising operation: another CRY + a Hadamard on the color qubit.
        qc.append(CRYGate(params[param_idx]), [selection_qubit, color_qubit])
        qc.append(HGate(), [color_qubit])
        param_idx += 1

        for q in flips:
            qc.x(q)
        qc.barrier()

    return qc


@dataclass
class OptimizationResult:
    """Outcome of `optimize_gp`.

    Attributes
    ----------
    optimized_qc
        Circuit with parameters bound to their optimal values.
    optimized_params
        Numerical values of all parameters at the optimum.
    history_mse, history_psnr, history_tv, history_combined
        Per-iteration objective traces.
    history_time
        Wall-clock seconds elapsed at each iteration.
    """

    optimized_qc: QuantumCircuit
    optimized_params: np.ndarray
    history_mse: list[float]
    history_psnr: list[float]
    history_tv: list[float]
    history_combined: list[float]
    history_time: list[float]


# Full optimize_gp pipeline left as a follow-up: it ties together encoding,
# testing, decoding, and scipy.optimize.minimize. The pieces above are enough
# to compose it from a Jupyter notebook today (see notebooks/04 in Fase 5).
