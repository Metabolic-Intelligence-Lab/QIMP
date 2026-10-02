"""Arithmetic-consistency check on the hardware read-out (no new QPU time).

Every hardware job measures all qubits. On exit the non-restoring divider leaves
the remainder r in the I_a register and preserves the divisor d in I_b, so each
shot yields a triple (q, r, d) per pixel. The identity q*d + r = I_a(p) with r < d
is an arithmetic relation between three registers; a read-out offset acting on
each register cannot create it. Per run and per pixel we report:
  - per-shot identity rate against the product-of-marginals null (same per-register
    read-out statistics, registers independent);
  - the joint 6-bit argmax over (q, r, d) against the true triple (chance 1/64);
  - per-register argmax accuracy (d is load-and-read-out only, q and r carry the arithmetic).
Usage: .venv/bin/python scripts/arithmetic_identity.py [--prefix c14_] [--json out.json]
"""
from __future__ import annotations
import argparse, collections, glob, json, re, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from qimp.processing.ratiometric_circuit import class_b_ratio
from run_hardware_class_b_nonrestoring import load_dataset

_LAY = {}
def layout_for(dataset, n, q, divider, load):
    key = (dataset, n, q, divider, load)
    if key not in _LAY:
        I_a, I_b = load_dataset(dataset, n, q)
        qc, lay = class_b_ratio(I_a, I_b, q=q, divider=divider, load=load)
        _LAY[key] = (I_a, I_b, lay, qc.num_qubits)
    return _LAY[key]

def analyse_run(d: Path):
    m = json.load(open(d / "metadata.json")); c = json.load(open(d / "counts.json"))
    n, q = int(m["n"]), int(m["q"])
    divider = m.get("divider", "nonrestoring")
    I_a, I_b, lay, total = layout_for(m["dataset"], n, q, divider, m.get("load", "mcx"))
    pos, Ia, Ib, quo = lay["position"], lay["I_a"], lay["I_b"], lay["quotient"]
    V = 1 << q
    joint = collections.defaultdict(lambda: np.zeros((V, V, V)))
    for s, cnt in c.items():
        f = s.replace(" ", "")
        bit = lambda i: int(f[total - 1 - i])
        col = sum(bit(pos[i]) << i for i in range(n)); row = sum(bit(pos[n + i]) << i for i in range(n))
        qv = sum(bit(quo[i]) << i for i in range(q)); rv = sum(bit(Ia[i]) << i for i in range(q)); dv = sum(bit(Ib[i]) << i for i in range(q))
        joint[(row, col)][qv, rv, dv] += cnt
    Q, Rr, D = np.meshgrid(np.arange(V), np.arange(V), np.arange(V), indexing="ij")
    px = []; truths = []
    valid = [(rc, J / J.sum()) for rc, J in sorted(joint.items()) if int(I_b[rc]) > 0]
    for (row, col), P in valid:
        a, b = int(I_a[row, col]), int(I_b[row, col])
        J = joint[(row, col)]
        if divider == "lookup":
            # out-of-place divider: the dividend register keeps a. Identity: the measured dividend is the
            # pixel's and the quotient register holds the integer quotient of the measured operands.
            ident = (Rr == a) & (D > 0) & (Q == np.where(D > 0, Rr // np.maximum(D, 1), -1))
        else:
            ident = (Q * D + Rr == a) & (Rr < D)
        cross = float(np.mean([P2[ident].sum() for rc2, P2 in valid if rc2 != (row, col) and int(I_a[rc2]) != a])) if any(rc2 != (row, col) and int(I_a[rc2]) != a for rc2, _ in valid) else float('nan')
        pq, pr, pd = P.sum((1, 2)), P.sum((0, 2)), P.sum((0, 1))
        prod = pq[:, None, None] * pr[None, :, None] * pd[None, None, :]
        truth = (a // b, a if divider == "lookup" else a % b, b)
        truths.append(truth)
        top = np.unravel_index(P.argmax(), P.shape)
        px.append(dict(pixel=[row, col], true=list(truth), identity_rate=float(P[ident].sum()), identity_null=float(prod[ident].sum()),
                       identity_uniform=float(ident.mean()), identity_cross=cross, joint_true_share=float(P[truth]), joint_argmax_true=bool(tuple(top) == truth),
                       q_true=bool(pq.argmax() == truth[0]), q_ge1=bool(truth[0] >= 1), r_true=bool(pr.argmax() == truth[1]), d_true=bool(pd.argmax() == truth[2]),
                       q_share=float(pq[truth[0]]), r_share=float(pr[truth[1]]), d_share=float(pd[truth[2]]), shots=int(J.sum())))
    # constant-read-out null for the joint decode: the most common true triple over valid pixels
    const = max(truths.count(x) for x in set(truths)) if truths else 0
    t1 = [x for x in truths if x[0] >= 1]
    const1 = max(t1.count(x) for x in set(t1)) if t1 else 0
    return m, px, (const, const1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prefix", default="c"); ap.add_argument("--json", default=str(REPO / "paper/data_autonomous/arithmetic_identity.json"))
    a = ap.parse_args()
    per_cfg = collections.defaultdict(list)
    for d in sorted(glob.glob(str(REPO / "data/output/ibm_hw/*/runs" / f"{a.prefix}*_hw"))):
        d = Path(d)
        if not (d / "metadata.json").exists(): continue
        try:
            m, px, (const, const1) = analyse_run(d)
        except Exception as e:
            print(f"skip {d.name}: {e}", file=sys.stderr); continue
        cfg = re.sub(r"_r\d+$", "", m["label"])
        per_cfg[cfg].append(dict(label=m["label"], backend=m["backend"], pixels=px, const=const, const1=const1))
    out = {}
    print(f"{'config':44s} runs px  ident   null   cross  Δx/se  t_runs | q>=1: px  ident  cross  t_runs joint  q   | joint  const  q     r     d")
    for cfg, runs in per_cfg.items():
        P = [p for r in runs for p in r["pixels"]]
        if not P: continue
        ident = np.array([p["identity_rate"] for p in P]); null = np.array([p["identity_null"] for p in P]); unif = np.array([p["identity_uniform"] for p in P])
        cross = np.array([p["identity_cross"] for p in P]); ok = ~np.isnan(cross)
        # run-level statistic: mean over a run's pixels of (identity - cross), then mean / se over runs
        run_d = [np.nanmean([q['identity_rate'] - q['identity_cross'] for q in r['pixels']]) for r in runs]
        run_d1 = [np.nanmean([q['identity_rate'] - q['identity_cross'] for q in r['pixels'] if q['q_ge1']]) for r in runs if any(q['q_ge1'] for q in r['pixels'])]
        t_runs = float(np.mean(run_d) / (np.std(run_d, ddof=1) / np.sqrt(len(run_d)))) if len(run_d) > 1 and np.std(run_d, ddof=1) > 0 else float('nan')
        t_runs1 = float(np.mean(run_d1) / (np.std(run_d1, ddof=1) / np.sqrt(len(run_d1)))) if len(run_d1) > 1 and np.std(run_d1, ddof=1) > 0 else float('nan')
        P1 = [p for p in P if p['q_ge1']]
        cross_excess = float((ident[ok] - cross[ok]).mean()) if ok.any() else float('nan')
        cross_se = float(np.sqrt(((ident[ok] * (1 - ident[ok]) + cross[ok] * (1 - cross[ok])) / np.array([p['shots'] for p in P])[ok]).sum()) / ok.sum()) if ok.any() else float('nan')
        shots = np.array([p["shots"] for p in P]); se = np.sqrt((null * (1 - null) / shots).sum()) / len(P)
        row = dict(n_runs=len(runs), n_px=len(P), identity_rate=float(ident.mean()), identity_null=float(null.mean()), identity_uniform=float(unif.mean()),
                   excess_over_null_sigma=float((ident.mean() - null.mean()) / se), identity_cross=float(np.nanmean(cross)), cross_excess=cross_excess, t_runs=t_runs, n_runs_t=len(run_d), n_px_q1=len(P1), identity_q1=float(np.mean([p['identity_rate'] for p in P1])) if P1 else float('nan'), cross_q1=float(np.nanmean([p['identity_cross'] for p in P1])) if P1 else float('nan'), t_runs_q1=t_runs1, joint_q1=int(sum(p['joint_argmax_true'] for p in P1)), q_true_q1=int(sum(p['q_true'] for p in P1)), cross_excess_sigma=cross_excess / cross_se if cross_se and cross_se > 0 else float('nan'), joint_argmax_true=int(sum(p["joint_argmax_true"] for p in P)),
                   q_true=int(sum(p["q_true"] for p in P)), r_true=int(sum(p["r_true"] for p in P)), d_true=int(sum(p["d_true"] for p in P)),
                   joint_true_share=float(np.mean([p["joint_true_share"] for p in P])), backend=runs[0]["backend"],
                   q_share=float(np.mean([p['q_share'] for p in P])), r_share=float(np.mean([p['r_share'] for p in P])), d_share=float(np.mean([p['d_share'] for p in P])),
                   q_share_sd_runs=float(np.std([np.mean([p['q_share'] for p in r['pixels']]) for r in runs], ddof=1)) if len(runs) > 1 else 0.0,
                   d_share_sd_runs=float(np.std([np.mean([p['d_share'] for p in r['pixels']]) for r in runs], ddof=1)) if len(runs) > 1 else 0.0,
                   joint_const_null=int(sum(r["const"] for r in runs)), joint_const_null_q1=int(sum(r["const1"] for r in runs)), joint_chance=float(len(P) / 64 if P[0]['true'] is not None else 0))
        out[cfg] = row
        print(f"{cfg:44s} {row['n_runs']:3d} {row['n_px']:3d}  {row['identity_rate']:.3f}  {row['identity_null']:.3f}  {row['identity_cross']:.3f}  {row['cross_excess_sigma']:5.1f}  {row['t_runs']:5.1f} | {row['n_px_q1']:3d}  {row['identity_q1']:.3f}  {row['cross_q1']:.3f}  {row['t_runs_q1']:5.1f}  {row['joint_q1']:3d}/{row['n_px_q1']:<3d} ({row['joint_const_null_q1']:3d}) {row['q_true_q1']:3d} | {row['joint_argmax_true']:3d}/{row['n_px']:<3d} {row['joint_const_null']:4d}  {row['q_true']:3d}  {row['r_true']:3d}  {row['d_true']:3d}")
    json.dump(out, open(a.json, "w"), indent=2)

if __name__ == "__main__":
    main()
