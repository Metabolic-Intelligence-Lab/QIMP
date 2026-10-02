"""Resource table for the reversible-arithmetic primitive library (paper v2, §3).

For every primitive and every width q in a range, build the primitive in
isolation and report

  * logical counts straight from the construction: CNOT, Toffoli, C^3X
    (and the q-controlled MCX of the divide-by-zero detector);
  * the closed-form prediction for those counts, checked against the build;
  * CX count and depth after transpilation to {id, u, cx} at
    optimization_level=0 (the convention used throughout the paper) and
    at optimization_level=3;
  * data qubits, ancilla qubits (carry + pad), and a T-count under the
    standard 7 T per Toffoli with C^3X counted as 3 Toffolis (one clean
    ancilla), which is the accounting of Thapliyal et al. (2018).

Usage:
    .venv/bin/python scripts/primitive_resources.py [--qmax 8] [--json out.json] [--md out.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qiskit import QuantumCircuit, transpile  # noqa: E402

from qimp.processing import arithmetic as ar  # noqa: E402

BASIS = ["id", "u", "cx"]


# --------------------------------------------------------------------------
# builders: each returns (circuit, data_qubits, ancilla_qubits)
# --------------------------------------------------------------------------
def build_add(q: int):
    qc = QuantumCircuit(3 * q + 1)
    a, b, c = list(range(q)), list(range(q, 2 * q)), list(range(2 * q, 3 * q + 1))
    ar.q_add(qc, a, b, c)
    return qc, 2 * q, q + 1


def build_sub(q: int):
    qc = QuantumCircuit(3 * q + 1)
    a, b, c = list(range(q)), list(range(q, 2 * q)), list(range(2 * q, 3 * q + 1))
    ar.q_sub(qc, a, b, c)
    return qc, 2 * q, q + 1


def build_add_ctrl(q: int):
    qc = QuantumCircuit(3 * q + 2)
    a, b, c = list(range(q)), list(range(q, 2 * q)), list(range(2 * q, 3 * q + 1))
    ar.q_add_ctrl(qc, 3 * q + 1, a, b, c)
    return qc, 2 * q + 1, q + 1


def build_sub_ctrl(q: int):
    qc = QuantumCircuit(3 * q + 2)
    a, b, c = list(range(q)), list(range(q, 2 * q)), list(range(2 * q, 3 * q + 1))
    ar.q_sub_ctrl(qc, 3 * q + 1, a, b, c)
    return qc, 2 * q + 1, q + 1


def build_add_sub_ctrl(q: int):
    qc = QuantumCircuit(3 * q + 2)
    a, b, c = list(range(q)), list(range(q, 2 * q)), list(range(2 * q, 3 * q + 1))
    ar.q_add_sub_ctrl(qc, 3 * q + 1, a, b, c)
    return qc, 2 * q + 1, q + 1


def build_mul_const(q: int, k_bits=(1, 0, 1)):
    # k = 5 (two set bits), m = 3: accumulator q+m, carries popcount*(q+2), guard 1
    m = len(k_bits)
    pop = sum(k_bits)
    nb = q
    nacc = q + m
    nc = pop * (q + 2)
    qc = QuantumCircuit(nb + nacc + nc + 1)
    b = list(range(nb))
    acc = list(range(nb, nb + nacc))
    c = list(range(nb + nacc, nb + nacc + nc))
    guard = nb + nacc + nc
    ar.q_mul_const(qc, b, list(k_bits), acc, c, guard)
    return qc, nb + nacc, nc + 1


def _div_layout(q: int):
    nc = (q + 1) * (q + 2)
    total = 4 * q + 2 + nc
    qc = QuantumCircuit(total)
    dividend = list(range(q))
    divisor = list(range(q, 2 * q))
    quotient = list(range(2 * q, 3 * q))
    work = list(range(3 * q, 4 * q))
    pad = 4 * q
    flag = 4 * q + 1
    c = list(range(4 * q + 2, total))
    return qc, dividend, divisor, quotient, work, pad, c, flag, nc


def build_div_restoring(q: int):
    qc, dd, dv, qt, wk, pad, c, flag, nc = _div_layout(q)
    ar.q_div_restoring(qc, dd, dv, qt, wk, pad, c, flag)
    # data: dividend, divisor, quotient, flag ; ancilla: work, pad, carries
    return qc, 3 * q + 1, q + 1 + nc


def build_div_nonrestoring(q: int):
    qc, dd, dv, qt, wk, pad, c, flag, nc = _div_layout(q)
    ar.q_div_nonrestoring(qc, dd, dv, qt, wk, pad, c, flag)
    return qc, 3 * q + 1, q + 1 + nc


# --------------------------------------------------------------------------
# closed forms (logical counts), w = q + 1 is the divider window width
# --------------------------------------------------------------------------
def cf_add(q):
    return dict(cx=2 * q, ccx=2 * q, c3x=0)


def cf_sub(q):
    return dict(cx=2 * q, ccx=2 * q, c3x=0, x=2 * q + 2)


def cf_add_ctrl(q):
    return dict(cx=0, ccx=2 * q, c3x=2 * q)


def cf_sub_ctrl(q):
    return dict(cx=2 * q + 2, ccx=2 * q, c3x=2 * q)


def cf_add_sub_ctrl(q):
    return dict(cx=4 * q + 2, ccx=2 * q, c3x=0)


def cf_div_restoring(q):
    w = q + 1
    return dict(
        cx=q * (6 * w + 3),
        ccx=6 * q * w + (1 if q == 2 else 0),   # + div-zero MCX when q == 2
        c3x=2 * q * w + (1 if q == 3 else 0),   # div-zero MCX is C^3X when q == 3
        mcx_q=(1 if q >= 4 else 0),
    )


def cf_div_nonrestoring(q):
    w = q + 1
    return dict(
        cx=(2 * w + 1) + (q - 1) * (4 * w + 3) + (q - 1),
        ccx=2 * w * (q + 1) + (1 if q == 2 else 0),
        c3x=2 * w + (1 if q == 3 else 0),
        mcx_q=(1 if q >= 4 else 0),
    )


PRIMITIVES = [
    ("q_add", build_add, cf_add),
    ("q_sub", build_sub, cf_sub),
    ("q_add_ctrl", build_add_ctrl, cf_add_ctrl),
    ("q_sub_ctrl", build_sub_ctrl, cf_sub_ctrl),
    ("q_add_sub_ctrl", build_add_sub_ctrl, cf_add_sub_ctrl),
    ("q_mul_const(k=5)", build_mul_const, None),
    ("q_div_restoring", build_div_restoring, cf_div_restoring),
    ("q_div_nonrestoring", build_div_nonrestoring, cf_div_nonrestoring),
]


def logical_counts(qc: QuantumCircuit, q: int):
    ops = qc.count_ops()
    out = dict(x=ops.get("x", 0), cx=ops.get("cx", 0), ccx=ops.get("ccx", 0), c3x=0, mcx_q=0)
    for inst in qc.data:
        name = inst.operation.name
        nq = inst.operation.num_qubits
        if name in ("mcx", "mcx_gray", "c3x") or (name.startswith("mc") and nq >= 4):
            if nq == 4:
                out["c3x"] += 1
            elif nq == 3:
                out["ccx"] += 1
            else:
                out["mcx_q"] += 1
    return out


def t_count(lc):
    # 7 T per Toffoli; C^3X = 3 Toffoli with one clean ancilla (21 T);
    # a q-controlled MCX (q >= 4) = 2(q-1)-1 Toffolis with clean ancillae.
    return 7 * lc["ccx"] + 21 * lc["c3x"] + 7 * lc.get("mcx_q", 0) * 0  # mcx_q handled below


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qmin", type=int, default=2)
    ap.add_argument("--qmax", type=int, default=8)
    ap.add_argument("--json", default=str(REPO / "paper/data_autonomous/primitive_resources.json"))
    ap.add_argument("--md", default=str(REPO / "paper/data_autonomous/primitive_resources.md"))
    args = ap.parse_args()

    rows = []
    for name, builder, cf in PRIMITIVES:
        for q in range(args.qmin, args.qmax + 1):
            qc, ndata, nanc = builder(q)
            lc = logical_counts(qc, q)
            pred = cf(q) if cf else None
            ok = None
            if pred:
                ok = all(lc.get(k, 0) == v for k, v in pred.items() if k in ("cx", "ccx", "c3x"))
            t0 = transpile(qc, basis_gates=BASIS, optimization_level=0)
            t3 = transpile(qc, basis_gates=BASIS, optimization_level=3)
            tof_equiv = lc["ccx"] + 3 * lc["c3x"] + (2 * q - 3) * lc["mcx_q"]
            rows.append(dict(
                primitive=name, q=q, qubits=qc.num_qubits, data=ndata, ancilla=nanc,
                x=lc["x"], cnot=lc["cx"], toffoli=lc["ccx"], c3x=lc["c3x"], mcx_q=lc["mcx_q"],
                closed_form_ok=ok, toffoli_equiv=tof_equiv, t_count=7 * tof_equiv,
                cx_o0=t0.count_ops().get("cx", 0), depth_o0=t0.depth(),
                cx_o3=t3.count_ops().get("cx", 0), depth_o3=t3.depth(),
            ))
            print(f"{name:22s} q={q}  qubits={qc.num_qubits:3d}  CNOT={lc['cx']:4d} Tof={lc['ccx']:4d} "
                  f"C3X={lc['c3x']:3d}  cf_ok={ok}  CX(o0)={rows[-1]['cx_o0']:5d} CX(o3)={rows[-1]['cx_o3']:5d} "
                  f"depth(o3)={rows[-1]['depth_o3']:5d}", flush=True)

    Path(args.json).write_text(json.dumps(rows, indent=2))

    # Markdown table
    lines = ["| Primitive | q | qubits (data+anc) | CNOT | Toffoli | C³X | Tof-equiv | T-count | CX {u,cx} o0 | CX o3 | depth o3 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| `{r['primitive']}` | {r['q']} | {r['qubits']} ({r['data']}+{r['ancilla']}) | {r['cnot']} | {r['toffoli']} | "
                     f"{r['c3x']} | {r['toffoli_equiv']} | {r['t_count']} | {r['cx_o0']} | {r['cx_o3']} | {r['depth_o3']} |")
    Path(args.md).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.json}\nwrote {args.md}")


if __name__ == "__main__":
    main()
