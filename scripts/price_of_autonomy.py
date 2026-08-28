"""What does autonomy cost? (§1, the central novelty, priced)

The paper's defining claim is that the ratio is computed reversibly inside
the circuit "with no classically pre-computed target loaded". The natural
question a reviewer asks next is what that buys and what it costs, and the
manuscript never states the second half: how many more gates and qubits
the autonomous pipeline needs than the parametric alternative that loads
an already-computed quotient image.

Both circuits are compared here on the same read-out -- a quotient
register in superposition over the position register, decodable by the
same routine:

  * **autonomous**: `dual_neqr_load` + reversible divider. Computes
    R = I_a // I_b on the device. This is what the paper builds.
  * **parametric**: a single NEQR load of the classically pre-computed R.
    Same final state on the position+quotient registers, no arithmetic,
    and no I_a / I_b registers at all.

The ratio of the two is the price of autonomy. It is not an argument
against the autonomous construction -- the parametric circuit cannot be
inverted into a QAE oracle over the *inputs*, and its target has already
been computed classically, which is the thing the paper set out not to do
-- but it is the number that frames every hardware result in §7, and it
belongs in the paper rather than in a reviewer's head.

Usage:
    python scripts/price_of_autonomy.py
    python scripts/price_of_autonomy.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from qiskit import transpile  # noqa: E402

from qimp.encoding.neqr import neqr_circuit  # noqa: E402
from qimp.processing.ratiometric_circuit import class_b_ratio  # noqa: E402
from qimp.runtime import ibm  # noqa: E402
from qimp.testing import _ensure_measured  # noqa: E402

CONFIGS = [
    ("canonical_shared", 1, 2),
    ("canonical", 1, 3),
    ("canonical", 2, 2),
    ("canonical", 3, 2),
]

BASIS = ["id", "u", "cx"]


def two_q(circ) -> int:
    return sum(n for name, n in circ.count_ops().items()
               if name in ("cz", "cx", "ecr", "rzz"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="price_of_autonomy", description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--hardware", action="store_true",
                    help="also transpile against a real backend Target, so the "
                         "comparison includes heavy-hex routing overhead")
    ap.add_argument("--backend", type=str, default="ibm_marrakesh")
    args = ap.parse_args(argv)

    target = None
    if args.hardware:
        service = ibm.get_service()
        target = ibm.pick_backend(service, min_qubits=24, name=args.backend).target
        print(f"Also transpiling against {args.backend}\n")

    from run_hardware_class_b_nonrestoring import load_dataset

    rows = []
    for dataset, n, q in CONFIGS:
        I_a, I_b = load_dataset(dataset, n, q)
        R = np.where(I_b > 0, I_a // np.maximum(I_b, 1), 0)

        auto, _ = class_b_ratio(I_a, I_b, q=q, divider="nonrestoring")
        para = neqr_circuit(R.astype(int), q=q)

        a_t = transpile(_ensure_measured(auto), basis_gates=BASIS,
                        optimization_level=3, seed_transpiler=1)
        p_t = transpile(_ensure_measured(para), basis_gates=BASIS,
                        optimization_level=3, seed_transpiler=1)

        rec = {
            "dataset": dataset, "n": n, "q": q,
            "autonomous_qubits": auto.num_qubits,
            "parametric_qubits": para.num_qubits,
            "autonomous_cx": two_q(a_t),
            "parametric_cx": two_q(p_t),
            "autonomous_depth": a_t.depth(),
            "parametric_depth": p_t.depth(),
        }
        if target is not None:
            a_h = transpile(_ensure_measured(auto), target=target,
                            optimization_level=3, seed_transpiler=1)
            p_h = transpile(_ensure_measured(para), target=target,
                            optimization_level=3, seed_transpiler=1)
            rec["autonomous_cx_hw"] = two_q(a_h)
            rec["parametric_cx_hw"] = two_q(p_h)
        rows.append(rec)

    hdr = (f"{'config':22s} {'qubits a/p':>13s} {'CX auto':>9s} {'CX param':>9s} "
           f"{'price':>8s}")
    if target is not None:
        hdr += f" {'price (hw)':>11s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cfg = f"{r['dataset']} n={r['n']} q={r['q']}"
        price = r["autonomous_cx"] / max(r["parametric_cx"], 1)
        line = (f"{cfg:22s} {r['autonomous_qubits']:5d}/{r['parametric_qubits']:<7d} "
                f"{r['autonomous_cx']:9d} {r['parametric_cx']:9d} {price:7.1f}x")
        if target is not None:
            line += f" {r['autonomous_cx_hw'] / max(r['parametric_cx_hw'], 1):10.1f}x"
        print(line)
    print(
        "\n  'price' is the two-qubit-gate cost of computing the ratio in the\n"
        "  circuit rather than loading it already computed. The parametric\n"
        "  circuit is not an alternative to the paper's construction -- its\n"
        "  target has already been computed classically, and it has no\n"
        "  input registers to invert a QAE oracle over -- but it is the\n"
        "  yardstick every §7 hardware number should be read against."
    )

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"\nWritten to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
