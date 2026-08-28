"""Run the autonomous Class-B integer-ratio circuit on the 8×8 (n=3)
Laurdan canonical microscopy frame, via AerSimulator(method='matrix_product_state') with
shot-based measurement (statevector at 28 qubits = 4 GiB would take
~10 hours via Statevector.from_instruction).

Output:
  paper/data_autonomous/canonical_8x8.npz
  paper/figures_autonomous/fig7_n3_laurdan_quantum_decode.png  (generated separately)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image as PilImage
from qiskit import ClassicalRegister, transpile
from qiskit_aer import AerSimulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qimp.processing.ratiometric_circuit import (  # noqa: E402
    class_b_ratio,
    decode_class_b_ratio,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

REPO = Path(__file__).resolve().parents[1]
DATA_OUT = REPO / "paper" / "data_autonomous"

SHOTS = 4096   # 64 pixels × 64 per branch on average


def prepare_canonical(target_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Block-mean the canonical frame to 2^target_n × 2^target_n, quantise to q=2.
    Uses a square patch at offset (8, 4) sized to give the requested target side.
    """
    side = 1 << target_n
    src = (REPO / "data" / "immagini" / "trainQML"
           / "membraneStack_Sample011_L_UV_DC_001rbc3DM2.tif")
    img = np.asarray(PilImage.open(src))
    R = img[..., 0].astype(np.float64)
    G = img[..., 1].astype(np.float64)
    # Largest patch we can take at (8, 4): the image is 110×110.
    # Available rows: 110 - 8 = 102; cols: 110 - 4 = 106.
    # The largest square patch at (8, 4) is 102×102.
    # We need patch_side to be a multiple of `side` for clean block-mean.
    max_avail = min(110 - 8, 110 - 4)  # 102
    patch_side = (max_avail // side) * side
    if patch_side < side:
        raise ValueError(f"target n={target_n} (side={side}) too large for the 110×110 frame")
    block = patch_side // side
    R_patch = R[8:8 + patch_side, 4:4 + patch_side]
    G_patch = G[8:8 + patch_side, 4:4 + patch_side]
    R_d = R_patch.reshape(side, block, side, block).mean(axis=(1, 3))
    G_d = G_patch.reshape(side, block, side, block).mean(axis=(1, 3))

    def quantise(arr, q=2):
        lo, hi = float(arr.min()), float(arr.max())
        if hi == lo:
            return np.zeros_like(arr, dtype=np.int64)
        scaled = (arr - lo) / (hi - lo) * float((1 << q) - 1)
        return np.clip(np.round(scaled), 0, (1 << q) - 1).astype(np.int64)

    I_a = quantise(R_d)
    I_b = quantise(G_d)
    R_classical = np.where(I_b > 0, I_a // np.maximum(I_b, 1), 0)
    divzero = (I_b == 0)
    return I_a, I_b, R_classical, divzero


def prepare_canonical_8x8():
    """Backwards-compat alias for the n=3 case."""
    return prepare_canonical(target_n=3)


def main() -> int:
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    print("Preparing 8×8 q=2 Laurdan dataset…")
    I_a, I_b, R_classical, divzero = prepare_canonical_8x8()
    print(f"  I_a shape={I_a.shape}, dtype={I_a.dtype}")
    print(f"  divzero pixels: {int(divzero.sum())} / 64")
    print(f"  classical R range: [{R_classical.min()}, {R_classical.max()}]")

    print("\nBuilding autonomous Class-B circuit at n=3, q=2…")
    t0 = time.time()
    qc, layout = class_b_ratio(I_a, I_b, q=2)
    print(f"  qubits  = {qc.num_qubits}")
    print(f"  size    = {qc.size()}")
    print(f"  build elapsed = {time.time() - t0:.2f} s")

    print(f"\nAdding measurement (all qubits) + transpiling to {{id, u, cx}}…")
    t0 = time.time()
    creg = ClassicalRegister(qc.num_qubits, "meas")
    qc.add_register(creg)
    qc.measure(range(qc.num_qubits), range(qc.num_qubits))
    sim = AerSimulator(method="matrix_product_state")
    qc_t = transpile(qc, sim, basis_gates=["id", "u", "cx"], optimization_level=0)
    print(f"  transpile elapsed = {time.time() - t0:.2f} s")
    print(f"  transpiled size = {qc_t.size()}")

    print(f"\nRunning AerSimulator(method='matrix_product_state') with shots={SHOTS}…")
    t0 = time.time()
    result = sim.run(qc_t, shots=SHOTS).result()
    counts = result.get_counts()
    print(f"  MPS run elapsed = {time.time() - t0:.2f} s")
    print(f"  unique bitstrings observed: {len(counts)}")

    print("\nDecoding quantum-decoded ratio image + divzero mask…")
    quotient_quantum, divzero_quantum = decode_class_b_ratio(
        counts, n=3, q=2, layout=layout, total_qubits=qc.num_qubits,
    )

    # Match against classical reference for valid pixels.
    R_classical_valid = np.where(divzero, -1, R_classical)
    R_quantum_valid = np.where(divzero_quantum, -1, quotient_quantum)
    match = (R_classical_valid == R_quantum_valid).all()
    n_match = int((R_classical_valid == R_quantum_valid).sum())
    print(f"\nMatch (all 64 pixels including divzero flag): {match}")
    print(f"  Matching pixels: {n_match} / 64")
    if not match:
        diff = (R_classical_valid != R_quantum_valid)
        print(f"  Disagreeing positions: {np.argwhere(diff)[:5].tolist()}")
        print(f"  classical:\n{R_classical_valid}")
        print(f"  quantum:\n{R_quantum_valid}")

    np.savez(
        DATA_OUT / "canonical_8x8.npz",
        I_a=I_a, I_b=I_b,
        R_classical=R_classical, divzero_classical=divzero,
        R_quantum=quotient_quantum, divzero_quantum=divzero_quantum,
        qubits=qc.num_qubits, gates=qc.size(),
        match_count=n_match, shots=SHOTS,
    )
    print(f"\nSaved {DATA_OUT}/canonical_8x8.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
