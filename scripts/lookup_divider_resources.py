"""Resource comparison of the two dividers of the library, for §3.5 of the manuscript.

For each intensity width q: CNOTs after transpilation to a linear basis and qubit count, for
the long-division non-restoring divider and for the truth-table divider, plus the width at
which the long-division construction takes over. Writes a JSON artefact and a figure.

Usage: .venv/bin/python scripts/lookup_divider_resources.py [--max-q 4]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qimp.processing.arithmetic import q_div_lookup, q_div_nonrestoring

BASIS = ["cx", "rz", "sx", "x"]

def counts(qc: QuantumCircuit) -> int:
    return transpile(qc, basis_gates=BASIS, optimization_level=3, seed_transpiler=0).count_ops().get("cx", 0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max-q", type=int, default=4)
    a = ap.parse_args()
    rows = []
    for q in range(1, a.max_q + 1):
        t0 = time.time()
        lk = QuantumCircuit(3 * q + 1)
        q_div_lookup(lk, list(range(q)), list(range(q, 2 * q)), list(range(2 * q, 3 * q)), 3 * q)
        lk_cx, synth_s = counts(lk), time.time() - t0
        n_c = (q + 1) * (q + 2); nq = 4 * q + 2 + n_c
        ld = QuantumCircuit(nq)
        q_div_nonrestoring(ld, dividend_qubits=list(range(q)), divisor_qubits=list(range(q, 2 * q)),
                           quotient_qubits=list(range(2 * q, 3 * q)), work_qubits=list(range(3 * q, 4 * q)),
                           divisor_pad_qubit=4 * q, c_qubits=list(range(4 * q + 1, 4 * q + 1 + n_c)),
                           div_zero_flag=4 * q + 1 + n_c)
        rows.append(dict(q=q, lookup_cx=lk_cx, lookup_qubits=3 * q + 1, longdiv_cx=counts(ld), longdiv_qubits=nq,
                         synthesis_seconds=round(synth_s, 2)))
        r = rows[-1]
        print(f"q={q}: truth-table {r['lookup_cx']:5d} CNOTs on {r['lookup_qubits']:2d} qubits | "
              f"long division {r['longdiv_cx']:5d} on {r['longdiv_qubits']:2d} | synthesis {r['synthesis_seconds']} s", flush=True)
    cross = next((r["q"] for r in rows if r["lookup_cx"] > r["longdiv_cx"]), None)
    print(f"long division takes over at q = {cross}")
    json.dump(dict(rows=rows, crossover_q=cross), open(REPO / "paper/data_autonomous/divider_resources.json", "w"), indent=2)
    qs = [r["q"] for r in rows]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.semilogy(qs, [r["lookup_cx"] for r in rows], "o-", label="truth-table synthesis")
    ax.semilogy(qs, [r["longdiv_cx"] for r in rows], "s-", label="long division (non-restoring)")
    if cross:
        ax.axvline(cross - 0.5, color="0.6", ls=":", lw=1)
        ax.text(cross - 0.45, min(r["lookup_cx"] for r in rows) * 1.2, "crossover", fontsize=8, color="0.4")
    ax.set_xlabel("intensity width $q$"); ax.set_ylabel("CNOTs after transpilation")
    ax.set_xticks(qs); ax.legend(fontsize=8); fig.tight_layout()
    out = REPO / "paper/figures_autonomous/fig_divider_crossover.png"
    fig.savefig(out, dpi=200); print("wrote", out)

if __name__ == "__main__":
    main()
