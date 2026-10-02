"""Score campaign 20 against the bands of paper/HW_CAMPAIGN_20_PROTOCOL.md.

Class A (Laurdan GP): the sign is scored only where the true GP is non-zero, because the sign
of zero is not defined; the exact-value accuracy is scored against the target's
constant-read-out null, the score a read-out that returns one value everywhere would get.
Class C (roGFP): exact value against the same null. Runs of the same label with different job
identifiers are counted as repeats (see the deviation log of the protocol).
"""
from __future__ import annotations
import collections, glob, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]

def main() -> None:
    runs = collections.defaultdict(list)
    for d in sorted(glob.glob(str(REPO / "data/output/ibm_hw/*/runs/c20_*_hw"))):
        m = json.load(open(Path(d) / "metadata.json"))
        runs[m["label"].rsplit("_r", 1)[0]].append(m)
    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from run_campaign_20 import classical
    from run_hardware_class_b_nonrestoring import load_dataset
    print(f"{'target':32s} {'2q':>4} {'runs':>4} {'valid':>5} {'null':>11} {'exact':>15} {'sign (GP != 0)':>16} {'MAE':>6}")
    out = {}
    for cfg, ms in sorted(runs.items()):
        m0 = ms[0]
        Ia, Ib = load_dataset(m0["dataset"], m0["n"], m0["q"])
        ref, dz = classical(m0["operator_class"], Ia, Ib)
        valid = ~dz; nv = int(valid.sum())
        vals, counts = np.unique(np.round(ref[valid], 6), return_counts=True)
        null = int(counts.max())
        ex = sum(m["match_count"] for m in ms); tot = nv * len(ms)
        sg = sgn = 0
        for m in ms:
            d = np.array(m["decoded"]); r = ref
            nz = valid & (r != 0)
            sg += int((np.sign(d[nz]) == np.sign(r[nz])).sum()); sgn += int(nz.sum())
        mae = float(np.mean([m["mean_abs_error"] for m in ms]))
        sign_txt = f"{sg}/{sgn} ({100*sg/sgn:3.0f} %)" if sgn else "        -       "
        print(f"{cfg.replace('c20_',''):32s} {ms[0]['two_q_gate_count']:4d} {len(ms):4d} {nv:5d} {null:4d}/{nv:<5d} {ex:5d}/{tot:<5d} ({100*ex/tot:3.0f} %) {sign_txt:>16} {mae:6.3f}")
        out[cfg] = dict(runs=len(ms), two_q=ms[0]["two_q_gate_count"], n_valid=nv, null=null, exact=ex, pooled=tot,
                        sign_ok=sg, sign_n=sgn, mae=mae, above_null=sum(1 for m in ms if m["match_count"] > null),
                        jobs=[m["job_id"] for m in ms])
    json.dump(out, open(REPO / "paper/data_autonomous/campaign20_scores.json", "w"), indent=2)

if __name__ == "__main__":
    main()
