"""Campaign 19 (paper/HW_CAMPAIGN_19_PROTOCOL.md): maximum-likelihood amplitude estimation
on hardware. The state preparation A uses the truth-table divider and the uniformly
controlled load, and the Grover reflection is restricted to the qubits A entangles
(`active_qubits` in scripts/qae_demo_class_b.py), which is what brings A·Q^k into the
device's gate budget.

Phase k=0 runs first. The protocol's stop rule forbids spending on k=1 unless k=0 clears
its band. Transpiled circuits are written with their SHA-256 before the first submission.

Usage:
  .venv/bin/python scripts/run_campaign_19.py --k 0 --rounds 3 [--backend ibm_kingston]
  .venv/bin/python scripts/run_campaign_19.py --k 1 --rounds 3
  .venv/bin/python scripts/run_campaign_19.py --dry-run --k 0 1
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, io, json, sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from qiskit import ClassicalRegister, qpy, transpile
from qimp.runtime import ibm
from qae_demo_class_b import apply_Q_once, build_A
from run_hardware_class_b_nonrestoring import load_dataset

OUT = REPO / "data" / "output" / "ibm_hw"
DATASET, N, Q, THRESHOLD = "fourvalue", 1, 2, 2
COST_PER_JOB = {0: 3, 1: 4, 2: 5}

def build_ak(Ia, Ib, k):
    qc, b, e = build_A(Ia, Ib, Q, THRESHOLD, divider="lookup", load="ucry")
    for _ in range(k):
        apply_Q_once(qc, Ia, Ib, Q, THRESHOLD, b, e, divider="lookup", load="ucry", reflection="active")
    cr = ClassicalRegister(1, "c_good"); qc.add_register(cr); qc.measure(e["good_corrected"], cr[0])
    return qc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ibm_kingston"); ap.add_argument("--k", type=int, nargs="+", default=[0])
    ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    Ia, Ib = load_dataset(DATASET, N, Q)
    R = np.where(Ib > 0, Ia // np.maximum(Ib, 1), 0)
    a_true = float(np.mean((R > THRESHOLD) & (Ib > 0)))
    theta = np.arcsin(np.sqrt(a_true))
    svc = ibm.get_service(); be = svc.backend(a.backend); tgt = be.target
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    pre = OUT / stamp / "transpiled_c19"; pre.mkdir(parents=True, exist_ok=True)
    circuits = {}
    print(f"dataset {DATASET} n={N} q={Q} threshold={THRESHOLD}: a_true = {a_true:.4f}")
    for k in a.k:
        qc = build_ak(Ia, Ib, k)
        best = None
        for s in range(8):
            t = transpile(qc, target=tgt, optimization_level=3, seed_transpiler=s)
            n2 = sum(t.count_ops().get(g, 0) for g in ("cz", "cx", "ecr"))
            if best is None or n2 < best[0]: best = (n2, s, t)
        buf = io.BytesIO(); qpy.dump(best[2], buf); raw = buf.getvalue()
        (pre / f"k{k}_{a.backend}.qpy").write_bytes(raw)
        circuits[k] = dict(circ=best[2], two_q=best[0], seed=best[1], sha256=hashlib.sha256(raw).hexdigest(),
                           ideal=float(np.sin((2 * k + 1) * theta) ** 2))
        print(f"  k={k}: {best[0]} two-qubit gates (seed {best[1]}), ideal P(good) = {circuits[k]['ideal']:.4f}, sha256 {circuits[k]['sha256'][:16]}")
    json.dump({str(k): {kk: v for kk, v in c.items() if kk != "circ"} for k, c in circuits.items()},
              open(pre / f"manifest_{a.backend}.json", "w"), indent=2)
    if a.dry_run:
        return
    Options = ibm._sampler_options_cls(); opts = Options(); opts.twirling.enable_measure = True
    Sampler = ibm._sampler_v2_cls(); sampler = Sampler(mode=be, options=opts)
    for r in range(1, a.rounds + 1):
        for k, c in circuits.items():
            remaining = svc.usage().get("usage_remaining_seconds", 0)
            if remaining < COST_PER_JOB.get(k, 5) + 5:
                print(f"[c19_k{k}_{a.backend[4:]}_r{r}] SKIP: {remaining} s remaining (stop rule)", flush=True); continue
            label = f"c19_k{k}_{a.backend[4:]}_r{r}"
            if list(OUT.glob(f"*/runs/{label}_hw/counts.json")):
                print(f"[{label}] already on disk, not resubmitted", flush=True); continue
            t0 = time.time(); job = sampler.run([c["circ"]], shots=a.shots); jid = job.job_id(); res = job.result()
            counts = dict(res[0].data.c_good.get_counts())
            p1 = counts.get("1", 0) / a.shots
            d = OUT / stamp / "runs" / f"{label}_hw"; d.mkdir(parents=True, exist_ok=True)
            json.dump(counts, open(d / "counts.json", "w"))
            meta = dict(label=label, campaign=19, dataset=DATASET, n=N, q=Q, threshold=THRESHOLD, grover_k=k,
                        divider="lookup", load="ucry", reflection="active", backend=a.backend, shots=a.shots,
                        mitigation="trex", job_id=jid, two_q_gate_count=c["two_q"], seed_transpiler=c["seed"],
                        transpiled_sha256=c["sha256"], a_true=a_true, p_ideal=c["ideal"], p_measured=p1,
                        round=r, remaining_before=remaining, wallclock_seconds=round(time.time() - t0, 1))
            json.dump(meta, open(d / "metadata.json", "w"), indent=2)
            print(f"[{label}] job {jid}  2q {c['two_q']}  P(good) measured {p1:.4f} vs ideal {c['ideal']:.4f}  remaining before {remaining} s", flush=True)
    print(f"campaign 19 (k = {a.k}) on {a.backend} finished", flush=True)

if __name__ == "__main__":
    main()
