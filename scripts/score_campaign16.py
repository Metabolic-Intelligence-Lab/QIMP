"""Score campaign 16 per paper/HW_CAMPAIGN_16_PROTOCOL.md (per-run rule and outcome bands).
Also rescored for reference: the original c7_d and c8_a runs at their full shots."""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from arithmetic_identity import analyse_run
from analyse_hw_signal import analyse
from run_hardware_class_b_nonrestoring import load_dataset

def flatfield(d: Path):
    m = json.load(open(d / "metadata.json")); c = json.load(open(d / "counts.json"))
    r = analyse(c, load_dataset(m["dataset"], m["n"], m["q"]), m["divider"], m["n"], m["q"])
    P = np.array([px["histogram"] for px in r["pixels"]]); t = [px["true"] for px in r["pixels"]]
    Q = P - P.mean(axis=0, keepdims=True); am = Q.argmax(axis=1)
    return int(sum(1 for a, tt in zip(am, t) if tt is not None and a == tt)), sum(1 for tt in t if tt is not None)

def score(pattern):
    rows = []
    for d in sorted(glob.glob(str(REPO / "data/output/ibm_hw/*/runs" / pattern))):
        d = Path(d); m, px, (const, const1) = analyse_run(d); p1 = [p for p in px if p["q_ge1"]]
        ex1 = float(np.nanmean([p["identity_rate"] - p["identity_cross"] for p in p1])); j1 = sum(p["joint_argmax_true"] for p in p1)
        ja = sum(p["joint_argmax_true"] for p in px); ff, nv = flatfield(d)
        rows.append(dict(label=m["label"], shots=m["shots"], cx=m.get("two_q_gate_count"), excess_q1=ex1, joint_q1=j1, n_q1=len(p1), const_q1=const1,
                         joint_all=ja, const_all=const, flatfield=ff, n_valid=nv, present=bool(ex1 > 0.005 and j1 >= 9),
                         d_true=sum(p["d_true"] for p in px), q_true=sum(p["q_true"] for p in px), r_true=sum(p["r_true"] for p in px)))
    return rows

def main():
    out = {}
    for name, pat in (("c7_d (2026-09-10)", "c7_d_random4_n3_ucry_kingston_r*_hw"), ("c8_a (2026-09-10)", "c8_a_random4_n3_ucry_kingston_r*_hw"),
                      ("c16 A = c7_d r1 circuit", "c16_A_kingston_r*_hw"), ("c16 B = c8_a r1 circuit", "c16_B_kingston_r*_hw")):
        rows = score(pat); out[name] = rows
        print(f"\n{name}")
        for r in rows:
            print(f"  {r['label']:36s} shots {r['shots']:6d} CX {r['cx']}  excess(q>=1) {r['excess_q1']:+.4f}  joint q>=1 {r['joint_q1']:2d}/{r['n_q1']} (null {r['const_q1']})  "
                  f"joint all {r['joint_all']:2d}/64 (null {r['const_all']})  flat-field {r['flatfield']}/{r['n_valid']}  d/q/r {r['d_true']}/{r['q_true']}/{r['r_true']}  -> {'PRESENT' if r['present'] else 'absent'}")
    A = out["c16 A = c7_d r1 circuit"]; B = out["c16 B = c8_a r1 circuit"]
    na, nb = sum(r["present"] for r in A), sum(r["present"] for r in B)
    print(f"\nA present {na}/{len(A)}, B present {nb}/{len(B)}")
    if len(A) >= 2 and len(B) >= 2:
        if na >= 2 and nb <= 1: band = "H-circuit"
        elif na >= 2 and nb >= 2: band = "reproducible now (H-time: campaign 8 was a drift episode)"
        elif na + nb <= 1: band = "absent now (H-time: campaign 7 was a transient device state)"
        else: band = "intermittent"
    else:
        band = "insufficient runs (stop rule)"
    print("band:", band)
    json.dump(dict(runs=out, band=band, a_present=na, b_present=nb), open(REPO / "paper/data_autonomous/campaign16_scores.json", "w"), indent=2)

if __name__ == "__main__":
    main()
