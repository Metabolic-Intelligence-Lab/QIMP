"""Campaign 18 (paper/HW_CAMPAIGN_18_PROTOCOL.md): Class-B circuits with the truth-table
divider (divider="lookup", uniformly controlled load), TREX only, no decoupling.

Phase 1 transpiles every target for the backend (best of 16 seeds by two-qubit count), writes
the transpiled circuits and their SHA-256 to disk, and only then submits. Targets are
submitted in rounds (all targets once per round) so that each target's runs are spread in
time. A job is skipped if the remaining allowance is below its cost estimate plus 5 s.

Usage:
  .venv/bin/python scripts/run_campaign_17.py --backend ibm_kingston --rounds 3 --phase main
  .venv/bin/python scripts/run_campaign_17.py --backend ibm_marrakesh --rounds 3 --phase replication
  .venv/bin/python scripts/run_campaign_17.py --backend ibm_kingston --dry-run
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, io, json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from qiskit import qpy, transpile
from qimp.processing.ratiometric_circuit import class_b_ratio
from qimp.runtime import ibm
from run_hardware_class_b_nonrestoring import load_dataset

OUT = REPO / "data" / "output" / "ibm_hw"
MAIN = [("blocks", 4, 65536), ("tiled", 4, 65536), ("blocks", 3, 65536)]
REPLICATION = [("blocks", 4, 65536)]
COST = {65536: 22, 16384: 7, 4096: 3}   # estimated QPU seconds per job

def two_q(c):
    ops = c.count_ops(); return sum(ops.get(k, 0) for k in ("cz", "cx", "ecr"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ibm_kingston"); ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--phase", choices=["main", "replication"], default="main"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    svc = ibm.get_service(); be = svc.backend(a.backend); tgt = be.target
    targets = MAIN if a.phase == "main" else REPLICATION
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    pre = OUT / stamp / "transpiled_c18"; pre.mkdir(parents=True, exist_ok=True)
    circuits = {}
    for ds, n, shots in targets:
        Ia, Ib = load_dataset(ds, n, 2)
        qc, lay = class_b_ratio(Ia, Ib, q=2, divider="lookup", load="ucry"); qc.measure_all()
        best = None
        for s in range(16):
            t = transpile(qc, target=tgt, optimization_level=3, seed_transpiler=s)
            if best is None or two_q(t) < best[0]: best = (two_q(t), s, t)
        buf = io.BytesIO(); qpy.dump(best[2], buf); raw = buf.getvalue()
        key = f"{ds}_n{n}"; (pre / f"{key}_{a.backend}.qpy").write_bytes(raw)
        circuits[key] = dict(ds=ds, n=n, shots=shots, circ=best[2], two_q=best[0], seed=best[1], sha256=hashlib.sha256(raw).hexdigest(),
                             layout=sorted(best[2].layout.final_index_layout()))
        print(f"  {key:22s} {a.backend}: {best[0]} two-qubit gates (seed {best[1]}), sha256 {circuits[key]['sha256'][:16]}", flush=True)
    json.dump({k: {kk: v for kk, v in c.items() if kk != "circ"} for k, c in circuits.items()}, open(pre / f"manifest_{a.backend}.json", "w"), indent=2)
    print(f"transpiled circuits written to {pre} before submission", flush=True)
    if a.dry_run:
        return
    Options = ibm._sampler_options_cls(); opts = Options(); opts.twirling.enable_measure = True
    Sampler = ibm._sampler_v2_cls(); sampler = Sampler(mode=be, options=opts)
    for r in range(1, a.rounds + 1):
        for key, c in circuits.items():
            remaining = svc.usage().get("usage_remaining_seconds", 0)
            if remaining < COST[c["shots"]] + 5:
                print(f"[c18_{key}_lookup_{a.backend[4:]}_r{r}] SKIP: {remaining} s remaining (stop rule)", flush=True); continue
            label = f"c18_{key}_lookup_{a.backend[4:]}_r{r}"
            t0 = time.time(); job = sampler.run([c["circ"]], shots=c["shots"]); jid = job.job_id(); res = job.result()
            counts = dict(res[0].data.meas.get_counts())
            d = OUT / stamp / "runs" / f"{label}_hw"; d.mkdir(parents=True, exist_ok=True)
            json.dump(counts, open(d / "counts.json", "w"))
            buf = io.BytesIO(); qpy.dump(c["circ"], buf); (d / "transpiled.qpy").write_bytes(buf.getvalue())
            meta = dict(label=label, campaign=17, dataset=c["ds"], n=c["n"], q=2, divider="lookup", load="ucry", backend=a.backend, shots=c["shots"],
                        mitigation="trex", job_id=jid, two_q_gate_count=c["two_q"], seed_transpiler=c["seed"], transpiled_sha256=c["sha256"],
                        physical_qubits=c["layout"], round=r, remaining_before=remaining, wallclock_seconds=round(time.time() - t0, 1))
            json.dump(meta, open(d / "metadata.json", "w"), indent=2)
            print(f"[{label}] job {jid}  2q {c['two_q']}  remaining before {remaining} s", flush=True)
    print(f"campaign 18 {a.phase} on {a.backend} finished", flush=True)

if __name__ == "__main__":
    main()
