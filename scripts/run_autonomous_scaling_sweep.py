"""Sweep the autonomous Class-B circuit over n ∈ {2, 3, 4, 5} on the
Laurdan canonical frame at q=2, all via AerSimulator(method='mps') with
4096 shots. Records qubit count, transpiled depth, MPS runtime, and
the bit-exact match rate vs the classical reference.

Output table is written as paper/data_autonomous/scaling_sweep.csv and
each (n, .npz) is saved in paper/data_autonomous/canonical_{side}x{side}.npz.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
from qiskit import ClassicalRegister, transpile
from qiskit_aer import AerSimulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from qimp.processing.ratiometric_circuit import (  # noqa: E402
    class_b_ratio,
    decode_class_b_ratio,
)
from prepare_autonomous_datasets import (  # noqa: E402
    synthesise_fura2, synthesise_rogfp, quantise_to_q,
)
from run_autonomous_n3_laurdan import prepare_canonical  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

REPO = Path(__file__).resolve().parents[1]
DATA_OUT = REPO / "paper" / "data_autonomous"
SHOTS = 4096


def _prepare_dataset(name: str, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    side = 1 << n
    if name == "canonical":
        return prepare_canonical(target_n=n)
    if name == "fura2":
        f340, f380, _ = synthesise_fura2(side=side, seed=1)
        I_a = quantise_to_q(f340, 2)
        I_b = quantise_to_q(f380, 2)
    elif name == "rogfp2":
        f405, f488, _ = synthesise_rogfp(side=side, seed=2)
        I_a = quantise_to_q(f405, 2)
        I_b = quantise_to_q(f488, 2)
    else:
        raise ValueError(f"unknown dataset {name}")
    R_classical = np.where(I_b > 0, I_a // np.maximum(I_b, 1), 0)
    divzero = (I_b == 0)
    return I_a.astype(np.int64), I_b.astype(np.int64), R_classical, divzero


def run_one(n: int, dataset: str = "canonical") -> dict:
    side = 1 << n
    print(f"\n=== {dataset} n={n}  ({side}×{side}) =================================")
    t0 = time.time()
    I_a, I_b, R_classical, divzero = _prepare_dataset(dataset, n)
    print(f"  data prep: {time.time() - t0:.2f} s  "
          f"({int(divzero.sum())} divzero / {side * side})")

    t0 = time.time()
    qc, layout = class_b_ratio(I_a, I_b, q=2)
    n_qubits = qc.num_qubits
    n_gates_logical = qc.size()
    print(f"  build: {time.time() - t0:.2f} s  ({n_qubits} qubits, {n_gates_logical} logical gates)")

    t0 = time.time()
    creg = ClassicalRegister(n_qubits, "meas")
    qc.add_register(creg)
    qc.measure(range(n_qubits), range(n_qubits))
    sim = AerSimulator(method="matrix_product_state")
    qc_t = transpile(qc, sim, basis_gates=["id", "u", "cx"], optimization_level=0)
    n_gates_transpiled = qc_t.size()
    print(f"  transpile: {time.time() - t0:.2f} s  ({n_gates_transpiled} transpiled gates)")

    t0 = time.time()
    result = sim.run(qc_t, shots=SHOTS).result()
    counts = result.get_counts()
    mps_time = time.time() - t0
    print(f"  MPS run: {mps_time:.2f} s  ({len(counts)} unique bitstrings)")

    quotient_quantum, divzero_quantum = decode_class_b_ratio(
        counts, n=n, q=2, layout=layout, total_qubits=n_qubits,
    )
    R_classical_valid = np.where(divzero, -1, R_classical)
    R_quantum_valid = np.where(divzero_quantum, -1, quotient_quantum)
    match = (R_classical_valid == R_quantum_valid)
    match_count = int(match.sum())
    pct = match_count / (side * side) * 100.0
    print(f"  match: {match_count} / {side * side}  ({pct:.1f}%)")

    out_file = DATA_OUT / f"{dataset}_{side}x{side}.npz"
    np.savez(
        out_file,
        I_a=I_a, I_b=I_b,
        R_classical=R_classical, divzero_classical=divzero,
        R_quantum=quotient_quantum, divzero_quantum=divzero_quantum,
        n_qubits=n_qubits, n_gates_logical=n_gates_logical,
        n_gates_transpiled=n_gates_transpiled,
        mps_seconds=mps_time, shots=SHOTS,
        match_count=match_count,
    )
    print(f"  saved {out_file.name}")

    return {
        "dataset": dataset,
        "n": n,
        "side": side,
        "n_pixels": side * side,
        "n_qubits": n_qubits,
        "n_gates_logical": n_gates_logical,
        "n_gates_transpiled": n_gates_transpiled,
        "divzero_pixels": int(divzero.sum()),
        "mps_seconds": round(mps_time, 2),
        "shots": SHOTS,
        "match_count": match_count,
        "match_pct": round(pct, 1),
    }


def main() -> int:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    datasets = ["canonical", "fura2", "rogfp2"]
    # n_max=5 for canonical, 4 for the synthetic ones (smaller patches OK
    # but the QAE oracle at that scale stays comfortable on MPS).
    plan = {"canonical": (2, 5), "fura2": (2, 5), "rogfp2": (2, 5)}
    for ds in datasets:
        n_lo, n_hi = plan[ds]
        for n in range(n_lo, n_hi + 1):
            try:
                rows.append(run_one(n, dataset=ds))
            except Exception as e:
                print(f"  *** {ds} n={n} FAILED: {e}")
                rows.append({"dataset": ds, "n": n, "side": 1 << n, "error": str(e)[:100]})

    # CSV report
    csv_path = DATA_OUT / "scaling_sweep.csv"
    if rows:
        fields = list({k for r in rows for k in r})
        # Put 'n' first for readability
        fields = ["n"] + [f for f in fields if f != "n"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"\n\n=== Sweep summary written to {csv_path} ===")
    for r in rows:
        print(
            f"  {r.get('dataset', '?'):>10} n={r.get('n'):>2}: "
            f"{r.get('n_qubits', '?'):>3} qubits, "
            f"{r.get('mps_seconds', '?'):>7} s MPS, "
            f"{r.get('match_count', '?')}/{r.get('n_pixels', '?')} match"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
