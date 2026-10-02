"""Loading cost vs arithmetic cost: when does the O(1/eps) query advantage
repay the NEQR load? (paper v2, §6.4)

Measures, in {id,u,cx} CX at optimization_level=0:
  * L(n, q): the dual NEQR load of two random q-bit images of side 2^n;
  * C_A(q): the arithmetic-plus-predicate part of the QAE state-prep A
    (non-restoring divider + threshold predicate), which is independent of n;
and reports M* = L / C_A, the number of oracle calls after which a
once-paid load is amortised in the pre-loaded regime, together with the
budget M_x at which the 1000-seed MLQAE RMSE first falls below Monte Carlo
(from paper/data_autonomous/qae_scaling.json).

Usage:  .venv/bin/python scripts/loading_crossover.py [--nmax 6]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from qiskit import QuantumCircuit, transpile
from qimp.processing.ratiometric_circuit import dual_neqr_load, class_b_ratio
from qiskit.circuit.library import MCXVChain
BASIS = ["id", "u", "cx"]


def with_vchain(qc: QuantumCircuit) -> QuantumCircuit:
    """Rewrite every k-controlled X (k >= 3) with the ancilla-assisted V-chain
    synthesis (k - 2 clean ancillae, linear cost), leaving other gates as is."""
    kmax = max((inst.operation.num_qubits - 1 for inst in qc.data if inst.operation.name.startswith("mc")), default=0)
    n_anc = max(kmax - 2, 0)
    out = QuantumCircuit(qc.num_qubits + n_anc)
    anc = list(range(qc.num_qubits, qc.num_qubits + n_anc))
    for inst in qc.data:
        qargs = [qc.find_bit(b).index for b in inst.qubits]
        if inst.operation.name.startswith("mc") and len(qargs) - 1 >= 3:
            k = len(qargs) - 1
            out.append(MCXVChain(k), qargs[:-1] + [qargs[-1]] + anc[: k - 2])
        else:
            out.append(inst.operation, qargs)
    return out

def cx(qc):
    return transpile(qc, basis_gates=BASIS, optimization_level=0).count_ops().get("cx", 0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nmax", type=int, default=6)
    ap.add_argument("--q", type=int, default=2)
    ap.add_argument("--divider", default="nonrestoring", choices=["restoring", "nonrestoring", "lookup"])
    ap.add_argument("--json", default=str(REPO / "paper/data_autonomous/loading_crossover.json"))
    a = ap.parse_args(); q = a.q
    rng = np.random.default_rng(0)
    # C_A: class_b_ratio at n=1 minus its own load, plus predicate (~ q_sub + inverse on q+1 bits)
    side = 2
    Ia = rng.integers(0, 2**q, (side, side)); Ib = rng.integers(1, 2**q, (side, side))
    full, _ = class_b_ratio(Ia, Ib, q=q, divider=a.divider)
    qc = QuantumCircuit(2 + 2*q); dual_neqr_load(qc, Ia, Ib, q, [0, 1], list(range(2, 2+q)), list(range(2+q, 2+2*q)))
    load1 = cx(qc); div = cx(full) - load1
    pred = 2 * 14 * (q + 1)   # q_sub + q_sub_inv on a (q+1)-bit window, 14 CX per bit
    C_A = div + pred
    rows = []
    for n in range(1, a.nmax + 1):
        side = 2**n
        Ia = rng.integers(0, 2**q, (side, side)); Ib = rng.integers(0, 2**q, (side, side))
        qc = QuantumCircuit(2*n + 2*q)
        dual_neqr_load(qc, Ia, Ib, q, list(range(2*n)), list(range(2*n, 2*n+q)), list(range(2*n+q, 2*n+2*q)))
        L = cx(qc)
        Lv = cx(with_vchain(qc))
        rows.append(dict(n=n, side=side, pixels=side*side, load_cx=L, load_cx_vchain=Lv,
                         arith_cx=C_A, M_star=L / C_A, M_star_vchain=Lv / C_A))
        print(f"n={n} side={side:3d}  L={L:7d} CX  L_vchain={Lv:7d} CX  C_A={C_A} CX  "
              f"M*={L/C_A:8.1f}  M*_vchain={Lv/C_A:8.1f}", flush=True)
    sc = json.load(open(REPO / "paper/data_autonomous/qae_scaling.json"))
    cross = {}
    for name, d in sc["datasets"].items():
        r = [x for x in d["rows"] if x["qae_rmse"] < x["mc_rmse"]]
        cross[name] = min(x["M"] for x in r) if r else None
        print(name, "MLQAE below MC from M =", cross[name],
              " ratio at M=256:", [round(x["mc_rmse"]/x["qae_rmse"],2) for x in d["rows"] if x["M"]==256])
    json.dump(dict(q=q, divider=a.divider, C_A=C_A, divider_cx=div, predicate_cx=pred, rows=rows, mlqae_below_mc_from_M=cross),
              open(a.json, "w"), indent=2)
    print("wrote", a.json)

if __name__ == "__main__":
    main()
