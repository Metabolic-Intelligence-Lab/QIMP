"""Run the autonomous Class-B integer-ratio circuit on the 4×4 (n=2)
Laurdan canonical microscopy frame, decode the quotient image from the
Aer statevector, and emit:

  paper/data_autonomous/canonical_4x4.npz   — 4×4 q=2 input + decoded output
  paper/figures_autonomous/fig6_n2_laurdan_quantum_decode.png

The full circuit is 26 qubits (n=2, q=2, Class B layout), which is
1 GiB of complex128 state — borderline on a 16 GB laptop but tractable.
Expected runtime: 5-15 minutes for the build + statevector +
counts-extraction at this scale.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image as PilImage
from qiskit.quantum_info import Statevector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qimp.processing.ratiometric_circuit import (  # noqa: E402
    class_b_ratio,
    decode_class_b_ratio,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

REPO = Path(__file__).resolve().parents[1]
DATA_OUT = REPO / "paper" / "data_autonomous"
FIG_OUT = REPO / "paper" / "figures_autonomous"


# ---------------------------------------------------------------------------
# Data preparation: 4×4 q=2 from a signal-rich patch of the canonical frame
# ---------------------------------------------------------------------------


def prepare_canonical_4x4() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (I_a, I_b, R_classical, divzero_mask) at 4×4 q=2 from a
    signal-rich 32×32 patch of the Laurdan canonical frame."""
    src = (REPO / "data" / "immagini" / "trainQML"
           / "membraneStack_Sample011_L_UV_DC_001rbc3DM2.tif")
    img = np.asarray(PilImage.open(src))
    R = img[..., 0].astype(np.float64)
    G = img[..., 1].astype(np.float64)
    # Use the same signal-rich 32×32 patch identified for the 2×2 dataset
    # (offset (8, 4); brute-force scan max R+G mean).
    patch_y, patch_x, patch_side = 8, 4, 32
    R_patch = R[patch_y:patch_y + patch_side, patch_x:patch_x + patch_side]
    G_patch = G[patch_y:patch_y + patch_side, patch_x:patch_x + patch_side]
    # Block-mean 32×32 → 4×4 (block = 8).
    R_4 = R_patch.reshape(4, 8, 4, 8).mean(axis=(1, 3))
    G_4 = G_patch.reshape(4, 8, 4, 8).mean(axis=(1, 3))

    def quantise(arr, q=2):
        lo, hi = float(arr.min()), float(arr.max())
        if hi == lo:
            return np.zeros_like(arr, dtype=np.int64)
        scaled = (arr - lo) / (hi - lo) * float((1 << q) - 1)
        return np.clip(np.round(scaled), 0, (1 << q) - 1).astype(np.int64)

    I_a = quantise(R_4)
    I_b = quantise(G_4)
    R_classical = np.where(I_b > 0, I_a // np.maximum(I_b, 1), 0)
    divzero = (I_b == 0)
    return I_a, I_b, R_classical, divzero


# ---------------------------------------------------------------------------
# Run + decode
# ---------------------------------------------------------------------------


def main() -> int:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    print("Preparing 4×4 q=2 Laurdan dataset…")
    I_a, I_b, R_classical, divzero = prepare_canonical_4x4()
    print(f"  I_a = {I_a.tolist()}")
    print(f"  I_b = {I_b.tolist()}")
    print(f"  R_classical (with divzero masked to 0) = {R_classical.tolist()}")
    print(f"  divzero pixels: {int(divzero.sum())} / 16")

    print("\nBuilding autonomous Class-B circuit at n=2, q=2…")
    t0 = time.time()
    qc, layout = class_b_ratio(I_a, I_b, q=2)
    print(f"  qubits  = {qc.num_qubits}")
    print(f"  size    = {qc.size()}")
    print(f"  build elapsed = {time.time() - t0:.2f} s")

    print("\nRunning Aer statevector (~1 GiB at 26 qubits)…")
    t0 = time.time()
    sv = Statevector.from_instruction(qc)
    print(f"  statevector elapsed = {time.time() - t0:.2f} s")

    print("\nExtracting basis-state counts (deterministic)…")
    t0 = time.time()
    probs = sv.probabilities_dict()
    # Convert to int counts (1 per non-zero probability state).
    # At a basis-state circuit each position branch has probability
    # 1 / 2^(2n) = 1/16; collapse with arbitrary multiplier.
    multiplier = 1024
    counts = {state: int(round(p * multiplier)) for state, p in probs.items() if p > 1e-9}
    counts = {k: v for k, v in counts.items() if v > 0}
    print(f"  basis states with weight: {len(counts)}")
    print(f"  extract elapsed = {time.time() - t0:.2f} s")

    print("\nDecoding quantum-decoded ratio image + divzero mask…")
    quotient_quantum, divzero_quantum = decode_class_b_ratio(
        counts, n=2, q=2, layout=layout, total_qubits=qc.num_qubits,
    )
    print(f"  Quantum-decoded R = {quotient_quantum.tolist()}")
    print(f"  Quantum divzero flag = {divzero_quantum.tolist()}")

    # Save data
    np.savez(
        DATA_OUT / "canonical_4x4.npz",
        I_a=I_a, I_b=I_b, R_classical=R_classical,
        divzero_classical=divzero,
        R_quantum=quotient_quantum, divzero_quantum=divzero_quantum,
        qubits=qc.num_qubits, gates=qc.size(),
    )
    print(f"\nSaved {DATA_OUT}/canonical_4x4.npz")

    # Match check
    R_classical_valid = np.where(divzero, -1, R_classical)
    R_quantum_valid = np.where(divzero_quantum, -1, quotient_quantum)
    match = (R_classical_valid == R_quantum_valid).all()
    print(f"\nBit-exact match (valid pixels): {match}")
    if not match:
        diff = (R_classical_valid != R_quantum_valid)
        print(f"  Pixels disagreeing: {int(diff.sum())} / 16")
        print(f"  classical: {R_classical_valid.tolist()}")
        print(f"  quantum:   {R_quantum_valid.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
