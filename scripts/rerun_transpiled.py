"""Campaign 16 (paper/HW_CAMPAIGN_16_PROTOCOL.md): re-execute archived, already-transpiled
hardware circuits unchanged (same layout, same routing), TREX only, no decoupling.

Usage:
  .venv/bin/python scripts/rerun_transpiled.py --source c7_d_random4_n3_ucry_kingston_r1_hw \
      --label c16_A_kingston_r1 --shots 40960 [--min-remaining 17]
"""
from __future__ import annotations
import argparse, datetime as dt, glob, json, shutil, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from qiskit import qpy
from qimp.runtime import ibm

OUT = REPO / "data" / "output" / "ibm_hw"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True); ap.add_argument("--label", required=True)
    ap.add_argument("--shots", type=int, default=40960); ap.add_argument("--min-remaining", type=int, default=17)
    a = ap.parse_args()
    src = sorted(glob.glob(str(OUT / "*" / "runs" / a.source)))
    assert len(src) == 1, src
    src = Path(src[0]); smeta = json.load(open(src / "metadata.json"))
    with open(src / "transpiled.qpy", "rb") as f:
        circ = qpy.load(f)[0]
    svc = ibm.get_service(); remaining = svc.usage().get("usage_remaining_seconds", 0)
    if remaining < a.min_remaining:
        print(f"[{a.label}] STOP: {remaining} s remaining < {a.min_remaining} (stop rule of the protocol)"); return
    be = svc.backend(smeta["backend"])
    Options = ibm._sampler_options_cls(); opts = Options(); opts.twirling.enable_measure = True
    Sampler = ibm._sampler_v2_cls(); sampler = Sampler(mode=be, options=opts)
    t0 = time.time(); job = sampler.run([circ], shots=a.shots); jid = job.job_id(); res = job.result()
    creg = circ.cregs[0].name
    counts = dict(getattr(res[0].data, creg).get_counts())
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    d = OUT / stamp / "runs" / f"{a.label}_hw"; d.mkdir(parents=True, exist_ok=True)
    json.dump(counts, open(d / "counts.json", "w")); shutil.copy(src / "transpiled.qpy", d / "transpiled.qpy")
    n2 = sum(v for k, v in circ.count_ops().items() if k in ("cz", "cx", "ecr", "rzz"))
    meta = {k: smeta[k] for k in ("dataset", "n", "q", "divider", "load", "backend", "n_pixels") if k in smeta}
    meta.update(label=a.label, campaign=16, source_run=a.source, source_job_id=smeta.get("job_id"), job_id=jid, shots=a.shots,
                mitigation="trex", two_q_gate_count=int(n2), wallclock_seconds=round(time.time() - t0, 1),
                remaining_before=remaining)
    json.dump(meta, open(d / "metadata.json", "w"), indent=2)
    print(f"[{a.label}] job {jid}  source {a.source}  CX {n2}  remaining before {remaining} s  wallclock {meta['wallclock_seconds']} s")

if __name__ == "__main__":
    main()
