"""Noise-model prediction for the lookup-divider Class-B circuits (campaign 17 protocol).
Transpiles each target to ibm_kingston (best of 16 seeds by two-qubit count), simulates with
AerSimulator.from_backend (current calibration), and scores argmax, flat-field and the
identity statistics exactly as the hardware analysis does.
Usage: .venv/bin/python scripts/predict_lookup_noise.py [--shots-8 65536]"""
from __future__ import annotations
import argparse, json, sys, tempfile
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from qiskit import transpile
from qiskit_aer import AerSimulator
from qimp.processing.ratiometric_circuit import class_b_ratio
from qimp.runtime import ibm
from run_hardware_class_b_nonrestoring import load_dataset
from analyse_hw_signal import analyse
from arithmetic_identity import analyse_run

TARGETS = [("fourvalue", 1, 4096), ("canonical_shared", 1, 4096), ("fourvalue", 2, 16384), ("random4", 2, 16384),
           ("canonical_shared", 2, 16384), ("random4", 3, 65536), ("canonical_shared", 3, 65536), ("fura2", 3, 65536), ("balanced8", 3, 65536)]

def best_transpile(qc, tgt, seeds=16):
    best = None
    for s in range(seeds):
        t = transpile(qc, target=tgt, optimization_level=3, seed_transpiler=s)
        c = t.count_ops().get("cz", 0) + t.count_ops().get("cx", 0) + t.count_ops().get("ecr", 0)
        if best is None or c < best[0]: best = (c, s, t)
    return best

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--backend", default="ibm_kingston"); a = ap.parse_args()
    be = ibm.get_service().backend(a.backend); tgt = be.target
    sim = AerSimulator.from_backend(be)
    out = {}
    for ds, n, shots in TARGETS:
        Ia, Ib = load_dataset(ds, n, 2)
        qc, lay = class_b_ratio(Ia, Ib, q=2, divider="lookup", load="ucry"); qc.measure_all()
        n2, seed, t = best_transpile(qc, tgt)
        counts = sim.run(t, shots=shots, seed_simulator=1).result().get_counts()
        r = analyse(counts, (Ia, Ib), "lookup", n, 2)
        P = np.array([px["histogram"] for px in r["pixels"]]); tr = [px["true"] for px in r["pixels"]]
        valid = [i for i, x in enumerate(tr) if x is not None]
        am = P.argmax(axis=1); ff = (P - P.mean(axis=0, keepdims=True)).argmax(axis=1)
        acc_am = float(np.mean([am[i] == tr[i] for i in valid])); acc_ff = float(np.mean([ff[i] == tr[i] for i in valid]))
        with tempfile.TemporaryDirectory() as td:
            d = Path(td); json.dump(counts, open(d / "counts.json", "w"))
            json.dump(dict(dataset=ds, n=n, q=2, divider="lookup", load="ucry", backend=a.backend, label="pred"), open(d / "metadata.json", "w"))
            m, px, (const, const1) = analyse_run(d)
        p1 = [p for p in px if p["q_ge1"]]
        row = dict(dataset=ds, n=n, shots=shots, routed_2q=int(n2), seed=seed, argmax_acc=acc_am, flatfield_acc=acc_ff, n_valid=len(valid),
                   joint_true=int(sum(p["joint_argmax_true"] for p in px)), joint_const=int(const), joint_q1=int(sum(p["joint_argmax_true"] for p in p1)), n_q1=len(p1), const_q1=int(const1),
                   identity_excess_q1=float(np.nanmean([p["identity_rate"] - p["identity_cross"] for p in p1])) if p1 else float("nan"))
        out[f"{ds}_n{n}"] = row
        print(f"{ds:17s} n={n} 2q {n2:4d} (seed {seed})  argmax {acc_am*100:5.1f} %  flat-field {acc_ff*100:5.1f} %  joint {row['joint_true']}/{len(px)} (null {const})  joint q>=1 {row['joint_q1']}/{len(p1)} (null {const1})  excess(q>=1) {row['identity_excess_q1']:+.3f}", flush=True)
    json.dump(out, open(REPO / f"paper/data_autonomous/lookup_noise_prediction_{a.backend}.json", "w"), indent=2)

if __name__ == "__main__":
    main()
