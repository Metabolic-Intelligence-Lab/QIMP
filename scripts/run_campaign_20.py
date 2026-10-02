"""Campaign 20 (paper/HW_CAMPAIGN_20_PROTOCOL.md): the Class-A (Laurdan generalized
polarization) and Class-C (roGFP redox index) per-pixel operators on hardware, with the
synthesised per-pixel maps (`class_a_gp_lookup`, `class_c_rogfp_lookup`) and the uniformly
controlled load. Both are decoded by the decoders the manuscript already uses.

Transpiled circuits and their SHA-256 are written before the first submission; targets are
submitted in rounds so each target's runs are spread in time; a job is skipped when the
remaining allowance is below its estimate plus 5 s.

Usage:
  .venv/bin/python scripts/run_campaign_20.py --backend ibm_kingston --rounds 3
  .venv/bin/python scripts/run_campaign_20.py --dry-run
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, io, json, sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from qiskit import qpy, transpile
from qimp.processing.ratiometric_circuit import (
    class_a_gp_lookup, class_c_rogfp_lookup, decode_class_a_full, decode_class_c_rogfp,
)
from qimp.runtime import ibm
from run_hardware_class_b_nonrestoring import load_dataset

OUT = REPO / "data" / "output" / "ibm_hw"
Q, Q_FRAC, R_RED_FP, R_OX = 2, 2, 1, 1.0
# (class, dataset, n, shots)
TARGETS = [("a", "canonical_shared", 2, 16384), ("a", "random4", 2, 16384), ("a", "canonical_shared", 3, 65536),
           ("c", "canonical_shared", 2, 16384), ("c", "random4", 2, 16384), ("a", "fourvalue", 1, 4096),
           ("c", "fourvalue", 1, 4096)]
COST = {65536: 22, 16384: 7, 4096: 3}

def build(cls, Ia, Ib):
    if cls == "a":
        return class_a_gp_lookup(Ia, Ib, q=Q, q_frac=Q_FRAC, load="ucry")
    return class_c_rogfp_lookup(Ia, Ib, q=Q, q_frac=Q_FRAC, R_red_fp=R_RED_FP, load="ucry")

def classical(cls, Ia, Ib):
    """The reference the decoders reconstruct, and the div-zero mask."""
    Ia, Ib = Ia.astype(int), Ib.astype(int)
    if cls == "a":
        den = Ia + Ib; dz = den == 0
        mag = np.where(dz, 0, (np.abs(Ia - Ib) << Q_FRAC) // np.maximum(den, 1))
        return np.where(Ib > Ia, -1.0, 1.0) * mag / float(1 << Q_FRAC), dz
    dz = Ib == 0
    ratio = np.where(dz, (1 << (Q + Q_FRAC)) - 1, (Ia << Q_FRAC) // np.maximum(Ib, 1))
    return (ratio - R_RED_FP) / (R_OX * float(1 << Q_FRAC)), dz

def two_q(c):
    ops = c.count_ops(); return sum(ops.get(k, 0) for k in ("cz", "cx", "ecr"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ibm_kingston"); ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--start-round", type=int, default=1)
    a = ap.parse_args()
    svc = ibm.get_service(); be = svc.backend(a.backend); tgt = be.target
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    pre = OUT / stamp / "transpiled_c20"; pre.mkdir(parents=True, exist_ok=True)
    circuits = {}
    for cls, ds, n, shots in TARGETS:
        Ia, Ib = load_dataset(ds, n, Q)
        qc, lay = build(cls, Ia, Ib); qc.measure_all()
        best = None
        for s in range(16):
            t = transpile(qc, target=tgt, optimization_level=3, seed_transpiler=s)
            if best is None or two_q(t) < best[0]: best = (two_q(t), s, t)
        buf = io.BytesIO(); qpy.dump(best[2], buf); raw = buf.getvalue()
        key = f"{cls}_{ds}_n{n}"; (pre / f"{key}_{a.backend}.qpy").write_bytes(raw)
        ref, dz = classical(cls, Ia, Ib)
        circuits[key] = dict(cls=cls, ds=ds, n=n, shots=shots, circ=best[2], layout=lay, two_q=best[0], seed=best[1],
                             sha256=hashlib.sha256(raw).hexdigest(), n_valid=int((~dz).sum()), n_px=int(Ia.size))
        print(f"  {key:26s} {a.backend}: {qc.num_qubits} qubits, {best[0]} two-qubit gates (seed {best[1]}), "
              f"{circuits[key]['n_valid']}/{circuits[key]['n_px']} valid pixels, sha256 {circuits[key]['sha256'][:16]}", flush=True)
    json.dump({k: {kk: v for kk, v in c.items() if kk not in ("circ", "layout")} for k, c in circuits.items()},
              open(pre / f"manifest_{a.backend}.json", "w"), indent=2)
    print(f"transpiled circuits written to {pre} before submission", flush=True)
    if a.dry_run:
        return
    Options = ibm._sampler_options_cls(); opts = Options(); opts.twirling.enable_measure = True
    Sampler = ibm._sampler_v2_cls(); sampler = Sampler(mode=be, options=opts)
    for r in range(a.start_round, a.start_round + a.rounds):
        for key, c in circuits.items():
            remaining = svc.usage().get("usage_remaining_seconds", 0)
            if remaining < COST[c["shots"]] + 5:
                print(f"[c20_{key}_{a.backend[4:]}_r{r}] SKIP: {remaining} s remaining (stop rule)", flush=True); continue
            label = f"c20_{key}_{a.backend[4:]}_r{r}"
            if list(OUT.glob(f"*/runs/{label}_hw/counts.json")):
                print(f"[{label}] already on disk, not resubmitted", flush=True); continue
            Ia, Ib = load_dataset(c["ds"], c["n"], Q)
            t0 = time.time(); job = sampler.run([c["circ"]], shots=c["shots"]); jid = job.job_id(); res = job.result()
            counts = dict(res[0].data.meas.get_counts())
            total = max(len(k) for k in counts)  # measure_all: one clbit per logical qubit
            if c["cls"] == "a":
                img, dzq = decode_class_a_full(counts, c["n"], Q, Q_FRAC, c["layout"], total)
            else:
                img, dzq = decode_class_c_rogfp(counts, c["n"], Q, Q_FRAC, 0.0, R_OX, c["layout"], total)
            ref, dz = classical(c["cls"], Ia, Ib)
            valid = ~dz
            match = int(np.sum((img == ref) & valid)); mae = float(np.mean(np.abs(img[valid] - ref[valid]))) if valid.any() else float("nan")
            sign_ok = int(np.sum((np.sign(img) == np.sign(ref)) & valid)) if c["cls"] == "a" else None
            d = OUT / stamp / "runs" / f"{label}_hw"; d.mkdir(parents=True, exist_ok=True)
            json.dump(counts, open(d / "counts.json", "w"))
            meta = dict(label=label, campaign=20, operator_class=c["cls"], dataset=c["ds"], n=c["n"], q=Q, q_frac=Q_FRAC,
                        R_red_fp=R_RED_FP, load="ucry", divider="lookup", backend=a.backend, shots=c["shots"],
                        mitigation="trex", job_id=jid, two_q_gate_count=c["two_q"], seed_transpiler=c["seed"],
                        transpiled_sha256=c["sha256"], n_valid=c["n_valid"], match_count=match,
                        sign_match=sign_ok, mean_abs_error=mae, divzero_match=int(np.sum(dzq == dz)),
                        decoded=img.tolist(), classical=ref.tolist(), round=r, remaining_before=remaining,
                        wallclock_seconds=round(time.time() - t0, 1))
            json.dump(meta, open(d / "metadata.json", "w"), indent=2)
            extra = f" sign {sign_ok}/{c['n_valid']}" if sign_ok is not None else ""
            print(f"[{label}] job {jid}  2q {c['two_q']}  exact {match}/{c['n_valid']}{extra}  MAE {mae:.3f}  remaining before {remaining} s", flush=True)
    print(f"campaign 20 on {a.backend} finished", flush=True)

if __name__ == "__main__":
    main()
