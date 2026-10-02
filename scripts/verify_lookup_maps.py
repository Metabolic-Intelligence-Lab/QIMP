"""Exactness of the synthesised per-pixel maps against the library's own pipelines.

For every operand pair at width q the Class-A and Class-C circuits built by
`class_a_gp_lookup` / `class_c_rogfp_lookup` are compared with `class_a_gp_full` /
`class_c_rogfp_full`, each decoded by the decoder the manuscript already uses. The
library circuits are simulated on the matrix-product-state backend (they are past the
statevector ceiling), the synthesised ones on statevector.

Usage: .venv/bin/python scripts/verify_lookup_maps.py [--q 2] [--q-frac 2] [--shots 4096]
"""
from __future__ import annotations
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from qiskit import transpile
from qiskit_aer import AerSimulator
from qimp.processing.ratiometric_circuit import (
    class_a_gp_full, class_a_gp_lookup, class_c_rogfp_full, class_c_rogfp_lookup,
    decode_class_a_full, decode_class_c_rogfp,
)

def _run(qc, method, shots, seed=7):
    """Transpile to the simulator basis only, with no backend, so the coupling-map
    width cap does not apply (as scripts/run_autonomous_class_{a,c}_*_mps.py do)."""
    m = qc.copy(); m.measure_all()
    t = transpile(m, basis_gates=["id", "u", "cx"], optimization_level=0)
    sim = AerSimulator(method=method, seed_simulator=seed)
    return sim.run(t, shots=shots).result().get_counts()

def images_covering(q):
    """2x2 images whose four pixels walk every (a, d) pair at width q."""
    pairs = [(a, b) for a in range(1 << q) for b in range(1 << q)]
    for k in range(0, len(pairs), 4):
        sel = pairs[k:k + 4]
        yield (np.array([[sel[0][0], sel[1][0]], [sel[2][0], sel[3][0]]]),
               np.array([[sel[0][1], sel[1][1]], [sel[2][1], sel[3][1]]]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=2); ap.add_argument("--q-frac", type=int, default=2)
    ap.add_argument("--shots", type=int, default=4096); ap.add_argument("--r-red-fp", type=int, default=1)
    a = ap.parse_args()
    q, qf = a.q, a.q_frac
    out = {"q": q, "q_frac": qf, "class_a": {"pairs": 0, "mismatch": 0}, "class_c": {"pairs": 0, "mismatch": 0}}
    for Ia, Ib in images_covering(q):
        # Class A
        ref_c, ref_l = class_a_gp_full(Ia, Ib, q=q, q_frac=qf)
        gp_ref, dz_ref = decode_class_a_full(_run(ref_c, "matrix_product_state", a.shots), 1, q, qf, ref_l, ref_c.num_qubits)
        new_c, new_l = class_a_gp_lookup(Ia, Ib, q=q, q_frac=qf)
        gp_new, dz_new = decode_class_a_full(_run(new_c, "statevector", a.shots), 1, q, qf, new_l, new_c.num_qubits)
        # Compare where the observable is defined. At I_a + I_b = 0 the library leaves the
        # magnitude register in whatever state the cascade left it and the synthesised map
        # writes zero; both set the flag, and the decoder masks those pixels.
        defined = ~dz_ref
        bad = int(np.sum(gp_ref[defined] != gp_new[defined]) + np.sum(dz_ref != dz_new))
        out["class_a"]["pairs"] += 4; out["class_a"]["mismatch"] += bad
        out["class_a"]["divzero_pixels"] = out["class_a"].get("divzero_pixels", 0) + int(dz_ref.sum())
        if bad:
            print(f"  Class A mismatch on I_a={Ia.tolist()} I_b={Ib.tolist()}: ref {gp_ref.tolist()} dz {dz_ref.tolist()} vs new {gp_new.tolist()} dz {dz_new.tolist()}")
        # Class C
        ref_c, ref_l = class_c_rogfp_full(Ia, Ib, q=q, q_frac=qf, R_red_fp=a.r_red_fp)
        rc_ref, dzc_ref = decode_class_c_rogfp(_run(ref_c, "matrix_product_state", a.shots), 1, q, qf, 0.0, 1.0, ref_l, ref_c.num_qubits)
        new_c, new_l = class_c_rogfp_lookup(Ia, Ib, q=q, q_frac=qf, R_red_fp=a.r_red_fp)
        rc_new, dzc_new = decode_class_c_rogfp(_run(new_c, "statevector", a.shots), 1, q, qf, 0.0, 1.0, new_l, new_c.num_qubits)
        definedc = ~dzc_ref
        badc = int(np.sum(rc_ref[definedc] != rc_new[definedc]) + np.sum(dzc_ref != dzc_new))
        out["class_c"]["pairs"] += 4; out["class_c"]["mismatch"] += badc
        if badc:
            print(f"  Class C mismatch on I_a={Ia.tolist()} I_b={Ib.tolist()}: ref {rc_ref.tolist()} dz {dzc_ref.tolist()} vs new {rc_new.tolist()} dz {dzc_new.tolist()}")
    for k in ("class_a", "class_c"):
        d = out[k]
        print(f"{k}: {d['pairs']} operand pairs ({d.get('divzero_pixels', 0)} of them div-zero, "
              f"compared on the flag only), {d['mismatch']} mismatches where the observable is defined "
              f"-> {'EXACT' if not d['mismatch'] else 'DIFFERS'}")
    json.dump(out, open(REPO / f"paper/data_autonomous/lookup_map_verification_q{q}.json", "w"), indent=2)

if __name__ == "__main__":
    main()
