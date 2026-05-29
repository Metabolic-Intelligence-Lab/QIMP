"""
Extended Class B Aer-MPS sweep at high shot count.

Re-runs n=5 (32x32) and adds n=6 (64x64) on Laurdan canonical, Fura-2 and
roGFP2 synthetic datasets at 16 shots/pixel (16384 shots for n=5, 65536
shots for n=6). The 4-shot/pixel n=5 result of the original sweep was
shot-limited at 97-99% match; this sweep is designed to resolve whether
the residual gap is purely sampling noise.

Output is paper/data_autonomous/highshot_n{5,6}_{dataset}.npz and a CSV
summary at paper/data_autonomous/highshot_sweep.csv.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
from qiskit import ClassicalRegister, transpile
from qiskit_aer import AerSimulator

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from qimp.processing.ratiometric_circuit import (  # noqa: E402
    class_b_ratio,
    decode_class_b_ratio,
)
from prepare_autonomous_datasets import (  # noqa: E402
    synthesise_fura2, synthesise_rogfp, quantise_to_q,
)
from run_autonomous_n3_laurdan import prepare_canonical  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

DATA_OUT = REPO / "paper" / "data_autonomous"


def _prepare_dataset(name: str, n: int):
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


def run_one(n: int, dataset: str, shots_per_pixel: int = 16) -> dict:
    side = 1 << n
    n_pixels = side * side
    shots = shots_per_pixel * n_pixels
    print(f"\n=== {dataset} n={n}  ({side}x{side} = {n_pixels} pixels, shots={shots}) ===")

    t0 = time.time()
    I_a, I_b, R_classical, divzero = _prepare_dataset(dataset, n)
    print(f"  data prep: {time.time() - t0:.2f} s  "
          f"({int(divzero.sum())} divzero / {n_pixels})")

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
    result = sim.run(qc_t, shots=shots).result()
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
    pct = match_count / n_pixels * 100.0
    print(f"  match: {match_count} / {n_pixels}  ({pct:.2f}%)")

    out_file = DATA_OUT / f"highshot_n{n}_{dataset}.npz"
    np.savez(
        out_file,
        I_a=I_a, I_b=I_b,
        R_classical=R_classical, divzero_classical=divzero,
        R_quantum=quotient_quantum, divzero_quantum=divzero_quantum,
        n_qubits=n_qubits, n_gates_logical=n_gates_logical,
        n_gates_transpiled=n_gates_transpiled,
        mps_seconds=mps_time, shots=shots, shots_per_pixel=shots_per_pixel,
        match_count=match_count,
    )
    print(f"  saved {out_file.name}")

    return {
        "dataset": dataset,
        "n": n,
        "side": side,
        "n_pixels": n_pixels,
        "n_qubits": n_qubits,
        "n_gates_logical": n_gates_logical,
        "n_gates_transpiled": n_gates_transpiled,
        "divzero_pixels": int(divzero.sum()),
        "mps_seconds": round(mps_time, 2),
        "shots": shots,
        "shots_per_pixel": shots_per_pixel,
        "match_count": match_count,
        "match_pct": round(pct, 2),
    }


def main() -> int:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    datasets = ["canonical", "fura2", "rogfp2"]
    n_values = [5, 6]
    for n in n_values:
        for ds in datasets:
            try:
                rows.append(run_one(n=n, dataset=ds, shots_per_pixel=16))
            except Exception as e:
                print(f"  *** {ds} n={n} FAILED: {e}")
                rows.append({"dataset": ds, "n": n, "side": 1 << n,
                             "error": str(e)[:200]})

    csv_path = DATA_OUT / "highshot_sweep.csv"
    if rows:
        fields = list({k for r in rows for k in r})
        fields = ["n"] + [f for f in fields if f != "n"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"\n=== high-shot sweep CSV: {csv_path} ===")
    for r in rows:
        if "error" in r:
            print(f"  {r['dataset']:>10} n={r['n']:>2}: ERROR {r['error']}")
        else:
            print(
                f"  {r.get('dataset', '?'):>10} n={r.get('n'):>2}: "
                f"{r.get('n_qubits', '?'):>3}q, "
                f"shots={r.get('shots', '?'):>5}, "
                f"{r.get('mps_seconds', '?'):>7} s MPS, "
                f"match {r.get('match_count', '?')}/{r.get('n_pixels', '?')} "
                f"({r.get('match_pct', '?'):.2f}%)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
