"""Counts-side zero-noise extrapolation (ZNE) helpers.

SamplerV2 exposes no ZNE option, so ZNE on bitstring distributions is done
by hand in two steps:

1. :func:`fold_gates` amplifies the gate noise of a circuit by global
   unitary folding, ``U -> U (U^dag U)^k``, optionally with a partial fold
   of the last gates for fractional scales. Final measurements are kept at
   the end and never folded.
2. :func:`zne_extrapolate` (or :func:`zne_probabilities`) fits each
   bitstring probability as a function of the noise scale and evaluates the
   fit at zero noise.

Folding is meant to be applied to a circuit that is already transpiled to
the target ISA: the copies are separated by barriers and the inverse gates
are rewritten into the target basis without optimisation, so nothing
cancels ``U^dag U`` afterwards. Only gate noise is amplified; readout error
is not scaled by folding and is left to TREX or another readout method.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Barrier, Delay, Gate

__all__ = [
    "ZNE_METHODS",
    "fold_gates",
    "zne_extrapolate",
    "zne_probabilities",
]

ZNE_METHODS = ("richardson", "linear", "exponential")

# Operations that may appear in the foldable (unitary) part besides gates.
_PASSIVE_OPS = (Barrier, Delay)


def _split_final_measurements(qc: QuantumCircuit) -> tuple[list[Any], list[Any]]:
    """Split ``qc.data`` into (body, tail).

    The tail holds the final measurements, plus barriers that sit only
    between the unitary part and those measurements. A measurement followed
    by a gate on the same qubit is a mid-circuit measurement and raises.
    """
    tail_idx: set[int] = set()
    blocked: set[Any] = set()  # qubits touched by a later body operation
    for idx in range(len(qc.data) - 1, -1, -1):
        instr = qc.data[idx]
        op = instr.operation
        qubits = set(instr.qubits)
        if op.name == "measure":
            if qubits & blocked:
                raise ValueError(
                    "fold_gates: mid-circuit measurement found; only circuits whose "
                    "measurements are all final can be folded"
                )
            tail_idx.add(idx)
        elif isinstance(op, Barrier) and not (qubits & blocked):
            tail_idx.add(idx)
        else:
            if not isinstance(op, (Gate, *_PASSIVE_OPS)):
                raise ValueError(f"fold_gates: non-unitary operation {op.name!r} cannot be folded")
            blocked |= qubits
    body = [instr for i, instr in enumerate(qc.data) if i not in tail_idx]
    tail = [instr for i, instr in enumerate(qc.data) if i in tail_idx]
    return body, tail


def _translate_to_target(qc: QuantumCircuit, target: Any) -> QuantumCircuit:
    """Rewrite gates outside ``target`` (e.g. ``sxdg``) into its basis.

    Uses only :class:`BasisTranslator`, so no gate is cancelled or merged.
    The layout of ``qc`` is carried over to the result.
    """
    from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import BasisTranslator

    pm = PassManager(
        [
            BasisTranslator(
                SessionEquivalenceLibrary,
                target_basis=list(target.operation_names),
                target=target,
            )
        ]
    )
    out = pm.run(qc)
    out._layout = qc._layout
    out.metadata = dict(qc.metadata or {})
    return out


def fold_gates(qc: QuantumCircuit, scale: float, *, target: Any | None = None) -> QuantumCircuit:
    """Return ``qc`` with its gate noise amplified by global unitary folding.

    The unitary part ``U`` (everything before the final measurements) is
    replaced by ``U (U^dag U)^k``. For an odd integer ``scale`` this gives
    ``k = (scale - 1) / 2`` full folds (1 -> U, 3 -> U U^dag U, ...). For
    other values the fractional remainder is realised by also folding the
    last ``m`` gates ``L`` of ``U`` once (``... L^dag L``), with ``m`` chosen
    so that the achieved scale ``1 + 2k + 2m/d`` (``d`` = number of gates
    in ``U``) is as close as possible to the request. The achieved scale is
    stored in ``folded.metadata["zne_fold_scale"]``.

    Final measurements (and the barriers just before them) are kept once,
    at the end. Barriers on the active qubits separate the copies, so a
    later optimisation pass cannot cancel ``U^dag U``. Fold after
    transpilation to the device ISA, and pass the device ``target`` so the
    inverse gates (``sxdg``, ``rz(-t)``, ...) are rewritten into the basis
    without optimisation; the circuit layout is preserved.

    Parameters
    ----------
    qc
        Circuit to fold, ideally already transpiled to the device ISA.
        Measurements, if any, must all be final.
    scale
        Requested noise scale, ``>= 1``. Odd integers give full folds.
    target
        Optional transpiler ``Target``. When given, gates outside its basis
        (introduced by the inverse) are translated into it.

    Returns
    -------
    QuantumCircuit
        The folded circuit, with the achieved scale in
        ``metadata["zne_fold_scale"]``.

    Raises
    ------
    ValueError
        For ``scale < 1``, mid-circuit measurements, or other non-unitary
        operations (reset, control flow) in the unitary part.
    """
    scale = float(scale)
    if not math.isfinite(scale) or scale < 1.0:
        raise ValueError(f"fold_gates: scale must be >= 1, got {scale}")

    body_instrs, tail_instrs = _split_final_measurements(qc)
    n_gates = sum(1 for instr in body_instrs if not isinstance(instr.operation, Barrier))

    half = (scale - 1.0) / 2.0
    k = math.floor(half + 1e-9)
    m = round((half - k) * n_gates) if n_gates else 0
    if m >= n_gates and n_gates:
        k, m = k + 1, 0
    achieved = 1.0 + 2.0 * k + (2.0 * m / n_gates if n_gates else 0.0)

    body = qc.copy_empty_like()
    for instr in body_instrs:
        body.append(instr)
    active = sorted(
        {
            q
            for instr in body_instrs
            if not isinstance(instr.operation, Barrier)
            for q in instr.qubits
        },
        key=lambda q: qc.find_bit(q).index,
    )

    folded = body.copy()
    if active and (k or m):
        body_inv = body.inverse()
        for _ in range(k):
            folded.barrier(active)
            folded.compose(body_inv, inplace=True)
            folded.barrier(active)
            folded.compose(body, inplace=True)
        if m:
            gate_instrs = [i for i in body_instrs if not isinstance(i.operation, Barrier)]
            last = qc.copy_empty_like()
            for instr in gate_instrs[-m:]:
                last.append(instr)
            folded.barrier(active)
            folded.compose(last.inverse(), inplace=True)
            folded.barrier(active)
            folded.compose(last, inplace=True)
        folded.barrier(active)
    for instr in tail_instrs:
        folded.append(instr)

    folded.metadata = {**(qc.metadata or {}), "zne_fold_scale": achieved}
    if target is not None:
        folded = _translate_to_target(folded, target)
    return folded


def _richardson_weights(x: np.ndarray) -> np.ndarray:
    """Lagrange weights that evaluate the interpolating polynomial at 0."""
    w = np.ones_like(x)
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j:
                w[i] *= x[j] / (x[j] - x[i])
    return w


def _exponential_fit(x: np.ndarray, p: np.ndarray, asymptote: float, floor: float) -> float:
    """Fit ``p(x) = asymptote + B exp(-c x)`` and return ``p(0)``.

    The asymptote is the uniform probability ``2^-n``, the value a bitstring
    reaches under global depolarising noise. When the deviations from the
    asymptote change sign or vanish the model is not identifiable and a
    linear least-squares fit is used instead.
    """
    delta = p - asymptote
    if np.all(delta > floor) or np.all(delta < -floor):
        _slope, intercept = np.polyfit(x, np.log(np.abs(delta)), 1)
        return float(asymptote + np.sign(delta[0]) * math.exp(min(intercept, 50.0)))
    return float(np.polyfit(x, p, 1)[-1])


def zne_probabilities(
    counts_per_scale: Sequence[Mapping[str, float]],
    scales: Sequence[float],
    *,
    method: str = "richardson",
) -> dict[str, float]:
    """Extrapolate bitstring probabilities to zero noise.

    Each ``counts_per_scale[i]`` is normalised to a probability
    distribution, and for every bitstring seen at any scale ``p(scale)`` is
    extrapolated to ``scale = 0``. Negative estimates are clipped to zero
    and the result is renormalised to sum to one.

    Parameters
    ----------
    counts_per_scale
        One counts (or probabilities) dict per noise scale.
    scales
        Noise scales matching ``counts_per_scale``; at least two, distinct.
    method
        ``"richardson"``: polynomial of degree ``len(scales) - 1`` through
        all points (exact for a quadratic trend with three scales).
        ``"linear"``: least-squares straight line. ``"exponential"``:
        ``p = 2^-n + B exp(-c scale)``, the decay of a global depolarising
        channel, with a linear fallback per bitstring.

    Returns
    -------
    dict[str, float]
        Zero-noise probabilities; bitstrings with zero estimate are dropped.

    Raises
    ------
    ValueError
        For an unknown method, mismatched lengths, fewer than two distinct
        scales, empty counts, or an all-zero estimate.
    """
    if method not in ZNE_METHODS:
        raise ValueError(f"unknown ZNE method {method!r}; expected one of {ZNE_METHODS}")
    if len(counts_per_scale) != len(scales):
        raise ValueError(f"got {len(counts_per_scale)} count dicts for {len(scales)} scales")
    x = np.asarray(scales, dtype=float)
    if len(x) < 2 or len(set(x.tolist())) != len(x):
        raise ValueError(f"ZNE needs at least two distinct scales, got {list(scales)}")

    keys = sorted({k for counts in counts_per_scale for k in counts})
    if not keys:
        raise ValueError("ZNE: all count dicts are empty")
    probs = np.zeros((len(x), len(keys)))
    totals = np.zeros(len(x))
    for i, counts in enumerate(counts_per_scale):
        totals[i] = float(sum(counts.values()))
        if totals[i] <= 0:
            raise ValueError(f"ZNE: counts at scale {scales[i]} are empty")
        for j, key in enumerate(keys):
            probs[i, j] = float(counts.get(key, 0)) / totals[i]

    if method == "richardson":
        est = _richardson_weights(x) @ probs
    elif method == "linear":
        est = np.polyfit(x, probs, 1)[-1]
    else:
        floor = 0.5 / float(totals.max()) if totals.max() > 1 else 1e-12
        est = np.array(
            [
                _exponential_fit(x, probs[:, j], 2.0 ** -len(key.replace(" ", "")), floor)
                for j, key in enumerate(keys)
            ]
        )

    est = np.clip(np.asarray(est, dtype=float), 0.0, 1.0)
    # Round-off from the fits (log, lstsq, exp) leaves residues of order
    # 1e-16 on bitstrings whose estimate is zero; they differ between BLAS
    # builds, so treat anything below the resolution of any shot count as 0.
    est[est < 1e-12] = 0.0
    total = float(est.sum())
    if total <= 0:
        raise ValueError("ZNE: every extrapolated probability is <= 0")
    return {key: float(v / total) for key, v in zip(keys, est, strict=True) if v > 0}


def zne_extrapolate(
    counts_per_scale: Sequence[Mapping[str, float]],
    scales: Sequence[float],
    *,
    method: str = "richardson",
    shots: int | None = None,
) -> dict[str, int]:
    """Zero-noise extrapolated counts, in the same form as sampler counts.

    Runs :func:`zne_probabilities` and rescales the result to integer
    counts with largest-remainder rounding, so the counts sum exactly to
    ``shots``.

    Parameters
    ----------
    counts_per_scale
        One counts dict per noise scale.
    scales
        Noise scales matching ``counts_per_scale``.
    method
        Extrapolation model, one of ``ZNE_METHODS``.
    shots
        Total of the returned counts. Defaults to the total count at the
        smallest scale.

    Returns
    -------
    dict[str, int]
        Extrapolated counts; zero-count bitstrings are dropped.

    Raises
    ------
    ValueError
        For non-positive ``shots`` or any error from
        :func:`zne_probabilities`.
    """
    probs = zne_probabilities(counts_per_scale, scales, method=method)
    if shots is None:
        i_min = int(np.argmin(np.asarray(scales, dtype=float)))
        shots = round(sum(counts_per_scale[i_min].values()))
    if shots <= 0:
        raise ValueError(f"shots must be positive, got {shots}")

    keys = list(probs)
    raw = np.array([probs[k] for k in keys]) * shots
    base = np.floor(raw).astype(int)
    remainder = shots - int(base.sum())
    if remainder > 0:
        order = np.argsort(-(raw - base), kind="stable")
        base[order[:remainder]] += 1
    return {k: int(c) for k, c in zip(keys, base, strict=True) if c > 0}
