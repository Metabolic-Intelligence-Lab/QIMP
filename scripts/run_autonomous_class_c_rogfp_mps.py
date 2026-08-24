"""End-to-end Class-C roGFP calibrated redox ratio on AerSimulator(mps).

First end-to-end run of the autonomous Class-C pipeline (`class_c_rogfp_full`),
the operator the §6.5.4 integer-quotient demonstration was degenerate on. It
computes the calibrated roGFP redox index

    R_C(p) = (R(p) - R_red) / (R_ox - R_red),   R(p) = I_a(p) / I_b(p)  (F405/F488)

reversibly inside the circuit on a synthetic roGFP2 dual-excitation image, at
n=1, q=4, q_frac=4 (~114 qubits), and verifies the quantum-decoded R_C against
the classical fixed-point reference bit-exactly.

The key over the Class-B integer ratio (which floors the sub-unit physiological
F405/F488 ratio to {0,1} and is therefore degenerate, §6.5.4): the fractional
ratio R~ = floor(I_a*2^q_frac / I_b) is computed by shifting I_a up by q_frac and
dividing with the non-square divider, giving q_frac fractional bits — exactly the
Class-A construction. The affine subtract (R~ - R_red_fp) runs in-circuit; the
single calibration scalar /((R_ox-R_red)*2^q_frac) is applied classically in the
decoder, in direct parallel with Class-A's /2^q_frac.

Quantisation uses an absolute photometric zero (lo=0, shared hi across channels):
fluorescence is measured from 0, so dark-but-nonzero F488 maps to a small positive
integer rather than a spurious divide-by-zero. Calibration constants are taken
from the synthetic generator's own reduced/oxidised endpoints
(R_red = I405_red/I488_red = 0.85, R_ox = I405_ox/I488_ox = 25.0) so R_C lands in
[0,1]; the operator itself is the §5.2 affine reparametrisation.

Run:  /tmp/qimpvenv/bin/python scripts/run_autonomous_class_c_rogfp_mps.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from qiskit import ClassicalRegister, transpile
from qiskit_aer import AerSimulator

from qimp.processing.ratiometric_circuit import (
    class_c_rogfp_full,
    decode_class_c_rogfp,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "data_autonomous"
sys.path.insert(0, str(REPO / "scripts"))
from prepare_autonomous_datasets import synthesise_rogfp  # noqa: E402

N = 1
Q = 4
Q_FRAC = 4
SHOTS = 8192
SIDE = 8
SEED = 2
R_RED = 0.85      # = I405_red / I488_red of the synthesise_rogfp generator
R_OX = 25.0       # = I405_ox  / I488_ox
R_RED_FP = round(R_RED * (1 << Q_FRAC))   # fixed-point reduced reference = 14


def quantise_abs(x: np.ndarray, hi: float, q: int) -> np.ndarray:
    """Quantise from absolute photometric zero onto [0, 2^q-1] with shared hi."""
    span = (1 << q) - 1
    return np.clip(np.round(x / hi * span), 0, span).astype(np.int64)


def classical_rc(Ia: np.ndarray, Ib: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dz = Ib == 0
    ratio_fp = np.where(~dz, (Ia * (1 << Q_FRAC)) // np.maximum(Ib, 1), 0)
    rc = (ratio_fp - R_RED_FP) / ((R_OX - R_RED) * (1 << Q_FRAC))
    return np.where(~dz, rc, np.nan), dz


def select_patch():
    """Deterministically pick the 2x2 patch with the most diverse R_C, no
    divzero, and quotient fitting the q+q_frac-bit register."""
    f405, f488, oxd = synthesise_rogfp(side=SIDE, seed=SEED)
    hi = max(float(f405.max()), float(f488.max()))
    A = quantise_abs(f405, hi, Q)
    B = quantise_abs(f488, hi, Q)
    w_max = (1 << (Q + Q_FRAC)) - 1
    best = None
    for y in range(SIDE - 1):
        for x in range(SIDE - 1):
            Ia = A[y : y + 2, x : x + 2]
            Ib = B[y : y + 2, x : x + 2]
            if (Ib == 0).any():
                continue
            if ((Ia * (1 << Q_FRAC)) // Ib > w_max).any():
                continue
            rc, _ = classical_rc(Ia, Ib)
            nuniq = len(set(np.round(rc.flatten(), 4).tolist()))
            score = (nuniq, float(np.nanmax(rc) - np.nanmin(rc)))
            if best is None or score > best[0]:
                best = (score, y, x, Ia.copy(), Ib.copy(), oxd[y : y + 2, x : x + 2].copy())
    return best


def main() -> int:
    _, y, x, Ia, Ib, oxd = select_patch()
    rc_ref, dz_ref = classical_rc(Ia, Ib)
    print(f"Selected synthetic roGFP patch offset=({y},{x}), n={N}, q={Q}, q_frac={Q_FRAC}")
    print(f"  I_a (F405) = {Ia.tolist()}")
    print(f"  I_b (F488) = {Ib.tolist()}")
    print(f"  R = I_a/I_b (fixed-point /16) = {((Ia*(1<<Q_FRAC))//Ib).tolist()}")
    print(f"  R_red={R_RED}, R_ox={R_OX}, R_red_fp={R_RED_FP}")
    print(f"  classical R_C = {np.round(rc_ref, 4).tolist()}")
    print(f"  (true OxD     = {np.round(oxd, 3).tolist()})")

    print("\nBuilding class_c_rogfp_full…")
    t0 = time.time()
    qc, layout = class_c_rogfp_full(Ia, Ib, q=Q, q_frac=Q_FRAC, R_red_fp=R_RED_FP)
    print(f"  qubits = {qc.num_qubits}, build = {time.time() - t0:.2f} s")

    qc.add_register(ClassicalRegister(qc.num_qubits))
    qc.measure(range(qc.num_qubits), range(qc.num_qubits))
    sim = AerSimulator(method="matrix_product_state")
    t0 = time.time()
    qc_t = transpile(qc, basis_gates=["id", "u", "cx"], optimization_level=0)
    print(f"  transpiled size = {qc_t.size()}, transpile = {time.time() - t0:.2f} s")

    print(f"\nRunning MPS, shots={SHOTS}…")
    t0 = time.time()
    counts = sim.run(qc_t, shots=SHOTS).result().get_counts()
    print(f"  run = {time.time() - t0:.2f} s")

    rc_q, dz_q = decode_class_c_rogfp(
        counts, N, Q, Q_FRAC, R_RED, R_OX, layout, qc.num_qubits
    )
    print(f"\nquantum R_C = {np.round(rc_q, 4).tolist()}")

    valid = ~dz_ref
    rc_match = int(np.isclose(rc_q[valid], rc_ref[valid]).sum())
    rc_total = int(valid.sum())
    dz_match = int((dz_q == dz_ref).sum())
    print(f"\nR_C bit-exact match: {rc_match}/{rc_total} valid pixels")
    print(f"divzero-flag match:  {dz_match}/{Ia.size} pixels")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / "class_c_rogfp_n1_q4.npz",
        I_a=Ia, I_b=Ib, rc_classical=rc_ref, rc_quantum=rc_q,
        divzero_classical=dz_ref, divzero_quantum=dz_q, oxd_true=oxd,
        patch_yx=np.array([y, x]), n=N, q=Q, q_frac=Q_FRAC,
        R_red=R_RED, R_ox=R_OX, R_red_fp=R_RED_FP,
        qubits=qc.num_qubits, transpiled_size=qc_t.size(), shots=SHOTS,
    )
    print(f"\nWrote {(OUT / 'class_c_rogfp_n1_q4.npz').name}")
    return 0 if (rc_match == rc_total and dz_match == Ia.size) else 1


if __name__ == "__main__":
    raise SystemExit(main())
