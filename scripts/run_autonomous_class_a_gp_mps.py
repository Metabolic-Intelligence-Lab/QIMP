"""End-to-end Class-A Laurdan Generalized-Polarization on AerSimulator(mps).

This is the first *end-to-end* run of the autonomous Class-A GP pipeline
(`class_a_gp_full`) — previously only its component primitives were exercised.
It computes the biologically-interpreted Laurdan GP observable

    gp(p) = (I_a(p) - I_b(p)) / (I_a(p) + I_b(p))  in [-1, +1]

reversibly inside the circuit on an experimental Laurdan-stained
erythrocyte-membrane frame, at n=1, q=4, q_frac=4 (142 qubits), and verifies
the quantum-decoded GP against the classical reference bit-exactly.

Two implementation points that make the demo faithful:
  * The two emission channels are quantised on a **shared photometric scale**
    (common lo/hi across R and G) so the GP magnitude is biophysically
    meaningful rather than an artefact of per-channel renormalisation.
  * The decoder reads the **full** quotient register (all q_dividend bits),
    not just the low q_frac bits, so the GP=±1 boundary (one channel = 0,
    magnitude = 2^q_frac) is recovered exactly.

Run with the bundled venv:  /tmp/qimpvenv/bin/python scripts/run_autonomous_class_a_gp_mps.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image
from qiskit import ClassicalRegister, transpile
from qiskit_aer import AerSimulator

from qimp.processing.ratiometric_circuit import (
    class_a_gp_full,
    decode_class_a_full,
)

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data" / "immagini" / "trainQML" / \
    "membraneStack_Sample011_L_UV_DC_001rbc3DM2.tif"
OUT = REPO / "paper" / "data_autonomous"

N = 1
Q = 4
Q_FRAC = 4
SHOTS = 8192


def block_mean(arr: np.ndarray, target_side: int) -> np.ndarray:
    h = arr.shape[0]
    b = h // target_side
    arr = arr[: b * target_side, : b * target_side]
    return arr.reshape(target_side, b, target_side, b).mean(axis=(1, 3))


def shared_quantise(a: np.ndarray, b: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray]:
    """Quantise both channels onto a single shared [lo, hi] -> [0, 2^q-1] scale."""
    lo = min(float(a.min()), float(b.min()))
    hi = max(float(a.max()), float(b.max()))
    span = (1 << q) - 1
    def f(x: np.ndarray) -> np.ndarray:
        return np.clip(np.round((x - lo) / (hi - lo) * span), 0, span).astype(np.int64)
    return f(a), f(b)


def classical_gp(Ia: np.ndarray, Ib: np.ndarray, q_frac: int) -> tuple[np.ndarray, np.ndarray]:
    den = Ia + Ib
    num = np.abs(Ia - Ib)
    sign = (Ib > Ia).astype(int)
    mag = np.where(den > 0, (num * (1 << q_frac)) // np.maximum(den, 1), 0)
    gp = np.where(den > 0, (-1.0) ** sign * mag / (1 << q_frac), np.nan)
    return gp, den == 0


def select_patch(R: np.ndarray, G: np.ndarray, n: int, q: int, q_frac: int):
    """Pick the n-level patch with the most sign-diverse, value-diverse GP map
    (prefers patches mixing positive and negative GP, then diversity)."""
    side = 1 << n
    P = side * 8
    best = None
    for y in range(0, R.shape[0] - P, 2):
        for x in range(0, R.shape[1] - P, 2):
            A = block_mean(R[y : y + P, x : x + P], side)
            B = block_mean(G[y : y + P, x : x + P], side)
            Ia, Ib = shared_quantise(A, B, q)
            gp, dz = classical_gp(Ia, Ib, q_frac)
            valid = ~dz
            if valid.sum() == 0:
                continue
            vals = gp[valid]
            has_pos = bool((vals > 0).any())
            has_neg = bool((vals < 0).any())
            nuniq = len(set(np.round(vals, 4).tolist()))
            score = (has_pos and has_neg, nuniq, float(np.nanmax(vals) - np.nanmin(vals)))
            if best is None or score > best[0]:
                best = (score, y, x, Ia, Ib, gp, dz)
    return best


def main() -> int:
    img = np.asarray(Image.open(SRC))
    R = img[..., 0].astype(np.float64)
    G = img[..., 1].astype(np.float64)

    score, y, x, Ia, Ib, gp_ref, dz_ref = select_patch(R, G, N, Q, Q_FRAC)
    print(f"Selected Laurdan patch offset=({y},{x}), n={N}, q={Q}, q_frac={Q_FRAC}")
    print(f"  I_a (R) = {Ia.tolist()}")
    print(f"  I_b (G) = {Ib.tolist()}")
    print(f"  classical GP = {np.round(gp_ref, 4).tolist()}")
    print(f"  divzero      = {dz_ref.tolist()}")

    print("\nBuilding class_a_gp_full…")
    t0 = time.time()
    qc, layout = class_a_gp_full(Ia, Ib, q=Q, q_frac=Q_FRAC)
    print(f"  qubits = {qc.num_qubits}, build = {time.time() - t0:.2f} s")

    qc.add_register(ClassicalRegister(qc.num_qubits))
    qc.measure(range(qc.num_qubits), range(qc.num_qubits))
    sim = AerSimulator(method="matrix_product_state")
    t0 = time.time()
    # Transpile to the MPS basis only — no backend, so no coupling-map width cap.
    qc_t = transpile(qc, basis_gates=["id", "u", "cx"], optimization_level=0)
    print(f"  transpiled size = {qc_t.size()}, transpile = {time.time() - t0:.2f} s")

    print(f"\nRunning MPS, shots={SHOTS}…")
    t0 = time.time()
    counts = sim.run(qc_t, shots=SHOTS).result().get_counts()
    print(f"  run = {time.time() - t0:.2f} s")

    gp_q, dz_q = decode_class_a_full(
        counts, N, Q, Q_FRAC, layout, qc.num_qubits
    )
    print(f"\nquantum GP  = {np.round(gp_q, 4).tolist()}")
    print(f"quantum dz  = {dz_q.tolist()}")

    valid = ~dz_ref
    gp_match = int(np.isclose(gp_q[valid], gp_ref[valid]).sum())
    gp_total = int(valid.sum())
    dz_match = int((dz_q == dz_ref).sum())
    print(f"\nGP bit-exact match: {gp_match}/{gp_total} valid pixels")
    print(f"divzero-flag match: {dz_match}/{Ia.size} pixels")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / "class_a_gp_n1_q4.npz",
        I_a=Ia, I_b=Ib, gp_classical=gp_ref, gp_quantum=gp_q,
        divzero_classical=dz_ref, divzero_quantum=dz_q,
        patch_yx=np.array([y, x]), n=N, q=Q, q_frac=Q_FRAC,
        qubits=qc.num_qubits, transpiled_size=qc_t.size(), shots=SHOTS,
    )
    print(f"\nWrote {(OUT / 'class_a_gp_n1_q4.npz').name}")
    return 0 if (gp_match == gp_total and dz_match == Ia.size) else 1


if __name__ == "__main__":
    raise SystemExit(main())
