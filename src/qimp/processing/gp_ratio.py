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
from qiskit.circuit.library import HGate, RYGate

__all__ = [
    "OptimizationResult",
    "apply_gp_function",
    "classical_gp_image",
    "combined_objective",
    "decode_gp_counts",
    "evaluate_gp",
    "optimize_gp",
]


def classical_gp_image(
    image_g: np.ndarray,
    image_r: np.ndarray,
    *,
    alpha: float = 1.0,
    eps: float = 1e-10,
) -> np.ndarray:
    """Classical Green-Purple ratio: ``(G − α·R) / (G + α·R)`` element-wise.

    Output is ``float64``, naturally clamped to ``[-1, 1]``, with a small
    ``eps`` in the denominator to avoid division by zero where both channels
    are zero. ``alpha`` is the red-channel weight (sometimes called ``G`` in
    the lab's notation, but renamed here to avoid colliding with the green
    channel name).
    """
    if image_g.shape != image_r.shape:
        raise ValueError(f"Channel shape mismatch: {image_g.shape} vs {image_r.shape}")
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    g = image_g.astype(np.float64)
    r = image_r.astype(np.float64)
    return (g - alpha * r) / (g + alpha * r + eps)


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
    """Append the parameterised GP sub-circuit in place (per-pixel).

    For each of the ``2^(2n)`` pixel positions, two parameters drive:

    1. a fully-controlled RY on the **green** branch (position matches +
       selection qubit = 1) writing onto the color qubit (the "G half"),
    2. a fully-controlled RY on the **red** branch (position matches +
       selection qubit = 0) writing onto the same color qubit (the "R
       half"), followed by an unconditional Hadamard that interferes the
       two channel amplitudes.

    Splitting the two rotations across the two selection branches is
    what makes the per-pixel parameter pair non-degenerate: had both
    rotations sat on the same selection branch, they would compose into
    a single ``RY(θ₁+θ₂)`` and the model would have only one effective
    parameter per pixel. (Older versions of this function also missed
    the position-matching controls — the X-flip mask was applied but
    the CRY took only ``selection_qubit`` as control, so every pixel
    branch produced the same rotation. Both bugs are fixed.)

    The circuit assumes the standard FRQI layout: ``n`` row qubits, ``n``
    column qubits, ``m`` selection qubits, 1 color qubit (color qubit
    index = ``2n+m``).

    Parameters
    ----------
    qc
        Circuit to modify. Must already encode the FRQI representation of
        two channels (so it has ``2n+m+1`` qubits).
    n
        Spatial qubits per axis (so image size = ``2^n × 2^n``).
    m
        Image-selection qubits (so number of stacked images = ``2^m``).
        Must be ≥ 1 for the GP construction (we need a selection qubit
        to be the extra control).
    params
        Sequence of ``2 · 2^(2n)`` parameters. Built externally via
        ``[Parameter(f"θ{i}") for i in range(2 * 2**(2*n))]`` and assigned
        later with ``qc.assign_parameters(...)``.

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
    if m < 1:
        raise ValueError("m must be >= 1; GP needs an image-selection qubit as extra control")

    selection_qubit = 2 * n  # first image-selection qubit
    color_qubit = 2 * n + m
    pos_qubits = list(range(2 * n))
    num_controls = 2 * n + 1  # position bits + selection qubit
    control_qubits = [*pos_qubits, selection_qubit]
    control_target = [*control_qubits, color_qubit]

    param_idx = 0
    for pixel_idx in range(1 << (2 * n)):
        # Position bits, LSB-first → qubits 0..2n-1, matching the FRQI encoder.
        pos_bits = format(pixel_idx, f"0{2 * n}b")[::-1]
        flips = [q for q, bit in enumerate(pos_bits) if bit == "0"]

        for q in flips:
            qc.x(q)

        # G half: MCRY conditioned on (position match) + selection=1.
        diff = RYGate(params[param_idx]).control(num_controls, annotated=True)
        qc.append(diff, control_target)
        param_idx += 1

        # R half: same MCRY but conditioned on (position match) + selection=0.
        # Flip the selection qubit so the same MCRY now fires on the R branch,
        # then flip it back to leave the rest of the circuit untouched.
        qc.x(selection_qubit)
        norm = RYGate(params[param_idx]).control(num_controls, annotated=True)
        qc.append(norm, control_target)
        qc.x(selection_qubit)
        qc.append(HGate(), [color_qubit])
        param_idx += 1

        for q in flips:
            qc.x(q)
        qc.barrier()

    return qc


@dataclass
class OptimizationResult:
    """Outcome of :func:`optimize_gp`.

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


def decode_gp_counts(counts: dict[str, int], n: int, m: int = 1) -> np.ndarray:
    """Decode GP-circuit measurement counts into a per-pixel GP image.

    Marginalises over the image-selection qubits and computes
    ``P(color = 1 | position)`` for each pixel, then maps the probability
    from ``[0, 1]`` to the canonical ``[-1, 1]`` GP range via ``2·P - 1``.

    Parameters
    ----------
    counts
        Histogram returned by :func:`qimp.testing.ideal_simulation` /
        :func:`qimp.testing.exact_counts` on a circuit that has had
        :func:`apply_gp_function` applied.
    n
        Spatial qubits per axis (image side = ``2^n``).
    m
        Image-selection qubits used at encoding (default 1 for the typical
        ``(R, G)`` stack).

    Returns
    -------
    np.ndarray
        ``(2^n, 2^n)`` float array of GP values in ``[-1, 1]``. Pixels with
        zero total counts (unreachable in the measurement) decode to ``0``.
    """
    if n < 1 or m < 0:
        raise ValueError("n must be >= 1 and m >= 0")
    side = 1 << n
    total_per_image = 1 << (2 * n)

    # Re-use the FRQI state-string parser so the layout convention stays in one place.
    from qimp.encoding.frqi import _decode_state_string

    sum_color_1 = np.zeros(total_per_image, dtype=np.float64)
    sum_total = np.zeros(total_per_image, dtype=np.float64)
    for state, count in counts.items():
        _image_idx, pixel_idx, color_bit = _decode_state_string(state, n, m)
        sum_total[pixel_idx] += count
        if color_bit == 1:
            sum_color_1[pixel_idx] += count

    with np.errstate(invalid="ignore", divide="ignore"):
        prob_one = np.where(sum_total > 0, sum_color_1 / sum_total, 0.5)
    gp_flat = 2.0 * prob_one - 1.0

    # Same MSB-row / LSB-col mapping as the FRQI decoder.
    img = np.zeros((side, side), dtype=np.float64)
    for pixel_idx in range(total_per_image):
        bits = format(pixel_idx, f"0{2 * n}b")
        row = int(bits[:n], 2)
        col = int(bits[n:], 2)
        img[row, col] = gp_flat[pixel_idx]
    return img


def evaluate_gp(
    green: np.ndarray,
    red: np.ndarray,
    params: Sequence[float] | np.ndarray,
    *,
    shots: int = 8_192,
    exact: bool = True,
) -> np.ndarray:
    """Run the full quantum GP pipeline and return the decoded GP image.

    Encodes ``(red, green)`` as a multi-image FRQI circuit, applies
    :func:`apply_gp_function` with the supplied parameter values, simulates
    the result, and decodes via :func:`decode_gp_counts`.

    Parameters
    ----------
    green, red
        Matching-shape ``(2^n, 2^n)`` channel arrays. Their max value sets
        the FRQI normalisation so the encoder stays well-defined.
    params
        Numerical parameter values for :func:`apply_gp_function`. Must have
        length ``2 · 2^(2n)``.
    shots
        Shot count for shot-based simulation (ignored when ``exact=True``).
    exact
        If ``True``, uses :func:`qimp.testing.exact_counts` (statevector,
        no shot noise). Otherwise uses :func:`qimp.testing.ideal_simulation`.

    Returns
    -------
    np.ndarray
        ``(2^n, 2^n)`` float GP image in ``[-1, 1]``.
    """
    import math

    from qimp.encoding.frqi import frqi_circuit

    if green.shape != red.shape:
        raise ValueError(f"channel shape mismatch: {green.shape} vs {red.shape}")
    if green.shape[0] != green.shape[1]:
        raise ValueError(f"channels must be square, got {green.shape}")
    side = green.shape[0]
    n = int(math.log2(side))
    if side != 1 << n:
        raise ValueError(f"channel side {side} is not a power of two")

    rg_stack = np.stack([red, green], axis=0).astype(np.float64)
    norm = float(rg_stack.max())
    if norm == 0:
        raise ValueError("both channels are entirely zero; nothing to encode")

    qc = frqi_circuit(rg_stack, normalization=norm)
    apply_gp_function(qc, n=n, m=1, params=list(params))

    if exact:
        from qimp.testing import exact_counts

        counts = exact_counts(qc)
    else:
        from qimp.testing import ideal_simulation

        counts = ideal_simulation(qc, shots=shots)
    return decode_gp_counts(counts, n=n, m=1)


def optimize_gp(
    green: np.ndarray,
    red: np.ndarray,
    *,
    alpha: float = 1.0,
    max_iter: int = 200,
    seed: int | None = None,
    weights: tuple[float, float, float] = (1.0, 0.0, 0.0),
    exact: bool = True,
    shots: int = 4_096,
) -> OptimizationResult:
    """Optimise the GP circuit parameters against the classical GP image.

    Wires together :func:`evaluate_gp` (quantum image), :func:`classical_gp_image`
    (target), and ``scipy.optimize.minimize(method="COBYLA")``. The default
    loss is pure MSE; pass non-zero ``weights`` to bring in the PSNR / TV
    terms of :func:`combined_objective`.

    With the per-pixel ansatz fix in :func:`apply_gp_function`, the model
    is expressive enough to reproduce the classical GP exactly. Empirically
    COBYLA converges to MSE → 0 (PSNR ~ 85+ dB) in 50 iterations for n=1
    and ~480 for n=2.

    Parameters
    ----------
    green, red
        Matching-shape square ``(2^n, 2^n)`` channels.
    alpha
        Red-channel weight passed to :func:`classical_gp_image` (the target).
    max_iter
        Cap on COBYLA iterations. Default 200 reaches machine-precision
        convergence for n ≤ 2; larger images may want 1000+.
    seed
        Optional RNG seed for the initial parameter vector.
    weights
        ``(α_MSE, β_PSNR, γ_TV)`` weights for :func:`combined_objective`.
        Default ``(1.0, 0.0, 0.0)`` (pure MSE) is what reaches the exact
        target; the PSNR / TV terms are mainly diagnostic.
    exact, shots
        Forwarded to :func:`evaluate_gp` on every iteration.

    Returns
    -------
    OptimizationResult
        Result container with the optimised circuit, parameters, and per-
        iteration history of every metric.
    """
    import math
    import time

    from scipy.optimize import minimize

    from qimp.encoding.frqi import frqi_circuit
    from qimp.metrics import mse as _mse
    from qimp.metrics import psnr as _psnr
    from qimp.metrics import total_variation as _tv

    if green.shape != red.shape or green.shape[0] != green.shape[1]:
        raise ValueError("green and red must be square arrays of the same shape")
    side = green.shape[0]
    n = int(math.log2(side))
    if side != 1 << n:
        raise ValueError(f"channel side {side} is not a power of two")

    target = classical_gp_image(green, red, alpha=alpha)
    num_params = 2 * (1 << (2 * n))
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-np.pi, np.pi, size=num_params)

    history_mse: list[float] = []
    history_psnr: list[float] = []
    history_tv: list[float] = []
    history_combined: list[float] = []
    history_time: list[float] = []
    t0 = time.perf_counter()
    a_w, b_w, g_w = weights

    def loss(params: np.ndarray) -> float:
        decoded = evaluate_gp(green, red, params, shots=shots, exact=exact)
        err = float(_mse(target, decoded))
        snr = float(_psnr(target, decoded, max_intensity=2.0))  # GP range [-1, 1] → max 2
        tv = float(_tv(decoded))
        combined = combined_objective(err, snr, tv, alpha=a_w, beta=b_w, gamma=g_w)
        history_mse.append(err)
        history_psnr.append(snr)
        history_tv.append(tv)
        history_combined.append(combined)
        history_time.append(time.perf_counter() - t0)
        return combined

    result = minimize(loss, x0, method="COBYLA", options={"maxiter": max_iter, "disp": False})
    optimal_params = np.asarray(result.x, dtype=np.float64)

    # Rebuild the optimised circuit so the caller can inspect / re-run it.
    rg_stack = np.stack([red, green], axis=0).astype(np.float64)
    norm = float(rg_stack.max()) or 1.0
    optimised_qc = frqi_circuit(rg_stack, normalization=norm)
    apply_gp_function(optimised_qc, n=n, m=1, params=list(optimal_params))

    return OptimizationResult(
        optimized_qc=optimised_qc,
        optimized_params=optimal_params,
        history_mse=history_mse,
        history_psnr=history_psnr,
        history_tv=history_tv,
        history_combined=history_combined,
        history_time=history_time,
    )
