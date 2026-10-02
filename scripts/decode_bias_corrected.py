"""Pre-registered bias-corrected decoders, scored on held-out runs (paper v2, §7.4).

PROTOCOL, fixed before any held-out number was looked at (9 Sep 2026).

Problem. On the balanced target (R = [[1,0],[0,1]], TREX only, non-restoring,
~650 CX) the read-out separates the two quotient values by 5.6 sigma, but a
global offset toward bin 1, larger than that separation, sends the per-pixel
argmax to 1 almost everywhere (match 2.27/4 against a 2/4 constant null).
Question: does a decoder that removes the common-mode offset return the
per-pixel ratio, when it is specified in advance and scored on runs it was
not fitted on?

Two decoders, both fixed here and not tuned afterwards:

  A. Flat-field (label-free, within-run). For each run, m = mean quotient
     histogram over the pixels of that run; pixel decode = argmax(p - m).
     Uses no truth labels and no other runs. Analogue of the flat-field
     correction of a microscopy frame. Undefined on a constant image, which is
     exactly the degenerate target the paper already excludes.

  B. Held-out calibrated offset (leave-one-run-out). For held-out run r,
     m_cal = mean pixel histogram over the other runs of the same
     configuration; decode = argmax(p - m_cal). Tests whether the offset is
     stable enough across jobs to be calibrated once and applied to a fresh
     run. Uses the calibration runs' target content only through their mean
     histogram, never the held-out run.

Scoring, per configuration:
  * match count per run under argmax (baseline), A, B; mean +/- sd over runs;
  * runs strictly above the constant-read-out null, with an exact binomial
    p-value against 1/2 (a sign test);
  * pooled per-pixel accuracy over all runs (chance = 1/4 for four bins);
  * separation and global bias after correction, same definition as
    analyse_balanced_target.py (d = p1 - p0, split by truth).

Configurations (all TREX only, non-restoring):
  balanced n=1  j16_/j17_/j17b_  (15 runs, null 2/4)
  fura2 n=1     j14_fura2_trexonly_  (5 runs, R=[[2,dz],[2,0]]: three valid pixels, two of them
                the high-order quotient 2, one divide-by-zero pixel; null 2/4)
  balanced n=2  j19_  (5 runs, 16 px, twelve valid: six 0s and six 1s, four divzero; null 6/16)
Restoring on balanced (j18_, 5 runs) is reported as a control: its separation
is 0.8 sigma, so a corrected decoder should NOT rescue it.

Whatever the outcome, it is reported as it comes.

Usage: .venv/bin/python scripts/decode_bias_corrected.py [--json out]
"""
from __future__ import annotations
import argparse, glob, json, sys
from math import comb
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from analyse_hw_signal import analyse, load_images  # noqa: E402
HW = REPO / "data" / "output" / "ibm_hw"

CONFIGS = {
    "balanced_n1_nonrestoring": dict(prefixes=("j16_", "j17_", "j17b_"), dataset="canonical_shared", n=1, q=2, divider="nonrestoring"),
    "fura2_n1_nonrestoring":    dict(prefixes=("j14_fura2_trexonly_",), dataset="fura2", n=1, q=2, divider="nonrestoring"),
    "balanced_n2_nonrestoring": dict(prefixes=("j19_",), dataset="canonical_shared", n=2, q=2, divider="nonrestoring"),
    "balanced_n1_restoring_control": dict(prefixes=("j18_",), dataset="canonical_shared", n=1, q=2, divider="restoring"),
}

def collect(prefixes):
    out = []
    for pref in prefixes:
        for d in sorted(glob.glob(str(HW / "*" / "runs" / f"{pref}*_hw"))):
            dp = Path(d)
            try:
                out.append((json.load(open(dp / "metadata.json")), json.load(open(dp / "counts.json"))))
            except OSError:
                pass
    return out

def sign_test_p(k, n):
    """One-sided exact binomial P[X >= k], X ~ Bin(n, 1/2)."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2**n

def run_matrices(runs, images, divider, n, q):
    """Per run: (P [px, bins], truth [px] with -1 for divzero)."""
    mats = []
    for meta, counts in runs:
        r = analyse(counts, images, divider, n, q)
        P = np.array([px["histogram"] for px in r["pixels"]], float)
        t = np.array([-1 if px["divzero"] else px["true"] for px in r["pixels"]])
        mats.append((meta["label"], P, t))
    return mats

def score(mats, decode, null):
    matches, correct, total, d1, d0 = [], 0, 0, [], []
    for i, (label, P, t) in enumerate(mats):
        Q = decode(i, P, mats)            # corrected matrix
        dec = Q.argmax(axis=1)
        valid = t >= 0
        m = int(((dec == t) & valid).sum()); matches.append(m)
        correct += m; total += int(valid.sum())
        for px in range(len(t)):
            if t[px] == 1: d1.append(Q[px, 1] - Q[px, 0])
            if t[px] == 0: d0.append(Q[px, 1] - Q[px, 0])
    m = np.array(matches, float)
    above = int((m > null).sum())
    out = dict(match_mean=round(m.mean(), 3), match_sd=round(m.std(ddof=1), 3) if len(m) > 1 else None,
               match_hist={str(int(v)): int((m == v).sum()) for v in sorted(set(m))},
               runs_above_null=above, n_runs=len(m), sign_test_p=round(sign_test_p(above, len(m)), 4),
               pixel_accuracy=round(correct / total, 3), pixels=total)
    if d1 and d0:
        d1, d0 = np.array(d1), np.array(d0)
        sep = d1.mean() - d0.mean(); se = np.sqrt(d1.var(ddof=1)/len(d1) + d0.var(ddof=1)/len(d0))
        out.update(separation=round(float(sep), 4), separation_sigma=round(float(sep/se), 2),
                   global_bias=round(float(np.concatenate([d1, d0]).mean()), 4))
    return out

def dec_argmax(i, P, mats): return P
def dec_flatfield(i, P, mats): return P - P.mean(axis=0, keepdims=True)
def dec_heldout(i, P, mats):
    others = [M for j, (_, M, _) in enumerate(mats) if j != i]
    m_cal = np.mean([M.mean(axis=0) for M in others], axis=0)
    return P - m_cal[None, :]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json", default=str(REPO / "paper/data_autonomous/bias_corrected_decoder.json"))
    a = ap.parse_args(); results = {}
    for name, cfg in CONFIGS.items():
        runs = collect(cfg["prefixes"])
        if not runs: print(name, "no runs"); continue
        images = load_images(cfg["dataset"], cfg["n"], cfg["q"])
        mats = run_matrices(runs, images, cfg["divider"], cfg["n"], cfg["q"])
        I_a, I_b = images
        R = np.where(I_b > 0, I_a // np.maximum(I_b, 1), 0); dz = I_b == 0
        null = max(max(int(((R == v) & ~dz).sum()) for v in range(4)), int(dz.sum()))
        res = {"n_runs": len(mats), "constant_readout_null": f"{null}/{R.size}", "cx_range": [min(m["two_q_gate_count"] for m, _ in runs), max(m["two_q_gate_count"] for m, _ in runs)]}
        for dname, fn in [("argmax", dec_argmax), ("A_flatfield", dec_flatfield), ("B_heldout_offset", dec_heldout)]:
            if dname == "B_heldout_offset" and len(mats) < 2: continue
            res[dname] = score(mats, fn, null)
        results[name] = res
        print(f"\n=== {name}: {len(mats)} runs, null {null}/{R.size}, CX {res['cx_range']}")
        for dname in ("argmax", "A_flatfield", "B_heldout_offset"):
            if dname in res:
                s = res[dname]
                print(f"  {dname:18s} match {s['match_mean']:.2f} ± {s['match_sd']}  hist {s['match_hist']}  above-null {s['runs_above_null']}/{s['n_runs']} (p={s['sign_test_p']})  "
                      f"pixel acc {s['pixel_accuracy']}  sep {s.get('separation')} ({s.get('separation_sigma')}σ)  bias {s.get('global_bias')}")
    json.dump(results, open(a.json, "w"), indent=2); print("\nwrote", a.json)

if __name__ == "__main__" and not any(f in sys.argv for f in ("--pooled", "--campaign3", "--campaign4", "--campaign5", "--campaign6", "--campaign6D", "--campaign7", "--campaign8")):
    main()


# ---------------------------------------------------------------------------
# Pooled-run extension (added 10 Sep 2026, after the per-run results above were
# recorded): pool the archived runs of a configuration as if they were one job
# with more shots, flat-field decode the pooled histograms, and trace pixel
# accuracy against shots per pixel over random pooling orders. Tests whether the
# balanced target is shot-noise limited.
# ---------------------------------------------------------------------------
def pooled_study(out_json: Path, out_png: Path, n_orders: int = 200, seed: int = 0):
    import collections
    rng = np.random.default_rng(seed)
    results = {}
    curves = {}
    for name, cfg in CONFIGS.items():
        runs = collect(cfg["prefixes"])
        if not runs:
            continue
        images = load_images(cfg["dataset"], cfg["n"], cfg["q"])
        N = len(runs)
        px_per_run = 4096 // (4 ** cfg["n"])

        def decode_pool(idx):
            c = collections.Counter()
            for i in idx:
                for k, v in runs[i][1].items():
                    c[k] += v
            r = analyse(dict(c), images, cfg["divider"], cfg["n"], cfg["q"])
            P = np.array([p["histogram"] for p in r["pixels"]])
            t = np.array([-1 if p["divzero"] else p["true"] for p in r["pixels"]])
            dec = (P - P.mean(axis=0, keepdims=True)).argmax(axis=1)
            valid = t >= 0
            return int(((dec == t) & valid).sum()), int(valid.sum()), dec.tolist(), t.tolist()

        m, v, dec, t = decode_pool(range(N))
        acc = {k: [] for k in range(1, N + 1)}
        for _ in range(n_orders):
            order = rng.permutation(N)
            for k in range(1, N + 1):
                mm, vv, _, _ = decode_pool(order[:k])
                acc[k].append(mm / vv)
        curve = [(k * px_per_run, float(np.mean(a))) for k, a in acc.items()]
        results[name] = dict(n_runs=N, pooled_shots=N * 4096, pooled_match=f"{m}/{v}", decoded=dec, true=t,
                             accuracy_vs_shots_per_pixel=curve)
        curves[name] = curve
        print(f"{name}: pooled {N} runs -> {m}/{v}; accuracy {curve[0][1]:.2f} -> {curve[-1][1]:.2f}")
    json.dump(results, open(out_json, "w"), indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        labels = {"balanced_n1_nonrestoring": "balanced, non-restoring, 650 CX",
                  "fura2_n1_nonrestoring": "Fura-2, non-restoring, 650 CX",
                  "balanced_n1_restoring_control": "balanced, restoring, 1100 CX (control)",
                  "balanced_n2_nonrestoring": "balanced n=2, non-restoring, 1250 CX"}
        for name, curve in curves.items():
            x, y = zip(*curve)
            ax.plot(x, y, marker="o", ms=3, label=labels.get(name, name))
        ax.axhline(0.25, ls=":", c="k", lw=0.8)
        ax.text(ax.get_xlim()[1] * 0.98, 0.26, "chance (4 bins)", ha="right", fontsize=7)
        ax.set_xlabel("shots per pixel (pooled runs)"); ax.set_ylabel("per-pixel accuracy, flat-field decoder")
        ax.set_xscale("log"); ax.set_ylim(0, 1.05); ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout(); fig.savefig(out_png, dpi=200)
        print("wrote", out_png)
    except Exception as e:  # pragma: no cover
        print("figure skipped:", e)


if __name__ == "__main__" and "--pooled" in sys.argv:
    pooled_study(REPO / "paper/data_autonomous/bias_corrected_decoder_pooled.json",
                 REPO / "paper/figures_autonomous/fig_flatfield_accuracy_vs_shots.png")


# ---------------------------------------------------------------------------
# Campaign 3 scoring (paper/HW_CAMPAIGN_3_PROTOCOL.md, fixed 10 Sep 2026 before
# any campaign-3 job was submitted). Same decoders and scoring as above, plus
# the 4x4 confusion matrix for the four-value target.
# ---------------------------------------------------------------------------
CONFIGS3 = {
    "c3_c1_balanced_16k_marrakesh": dict(prefixes=("c3_c1_",), dataset="canonical_shared", n=1, q=2, divider="nonrestoring"),
    "c3_c2_fourvalue_16k_marrakesh": dict(prefixes=("c3_c2_",), dataset="fourvalue", n=1, q=2, divider="nonrestoring"),
    "c3_c3_fourvalue_4k_kingston": dict(prefixes=("c3_c3_",), dataset="fourvalue", n=1, q=2, divider="nonrestoring"),
}


def campaign3(out_json: Path):
    results = {}
    for name, cfg in CONFIGS3.items():
        runs = collect(cfg["prefixes"])
        if not runs:
            print(name, ": no runs yet"); continue
        images = load_images(cfg["dataset"], cfg["n"], cfg["q"])
        mats = run_matrices(runs, images, cfg["divider"], cfg["n"], cfg["q"])
        I_a, I_b = images
        R = np.where(I_b > 0, I_a // np.maximum(I_b, 1), 0); dz = I_b == 0
        null = max(max(int(((R == v) & ~dz).sum()) for v in range(4)), int(dz.sum()))
        res = {"n_runs": len(mats), "constant_readout_null": f"{null}/{R.size}",
               "cx_range": [min(m["two_q_gate_count"] for m, _ in runs), max(m["two_q_gate_count"] for m, _ in runs)],
               "shots": sorted({m["shots"] for m, _ in runs})}
        for dname, fn in [("argmax", dec_argmax), ("A_flatfield", dec_flatfield)]:
            s = score(mats, fn, null)
            conf = np.zeros((4, 4), int)
            resid = []
            for i, (label, P, t) in enumerate(mats):
                Q = fn(i, P, mats); dec = Q.argmax(axis=1)
                for px in range(len(t)):
                    if t[px] >= 0:
                        conf[t[px], dec[px]] += 1
                        if dname == "A_flatfield":
                            resid.append(Q[px, t[px]])
            s["confusion_true_x_decoded"] = conf.tolist()
            if resid:
                s["residual_on_true_bin"] = [round(float(np.mean(resid)), 4), round(float(np.std(resid)), 4)]
            res[dname] = s
        results[name] = res
        print(f"\n=== {name}: {len(mats)} runs, shots {res['shots']}, null {null}/{R.size}, CX {res['cx_range']}")
        for dname in ("argmax", "A_flatfield"):
            s = res[dname]
            print(f"  {dname:12s} match {s['match_mean']} ± {s['match_sd']}  hist {s['match_hist']}  above-null {s['runs_above_null']}/{s['n_runs']} (p={s['sign_test_p']})  pixel acc {s['pixel_accuracy']}")
            print(f"               confusion (rows true 0..3, cols decoded 0..3): {s['confusion_true_x_decoded']}"
                  + (f"  residual on true bin {s.get('residual_on_true_bin')}" if 'residual_on_true_bin' in s else ""))
    json.dump(results, open(out_json, "w"), indent=2); print("\nwrote", out_json)


if __name__ == "__main__" and "--campaign3" in sys.argv:
    campaign3(REPO / "paper/data_autonomous/campaign3_decoders.json")


# ---------------------------------------------------------------------------
# Campaign 4 (paper/HW_CAMPAIGN_4_PROTOCOL.md, fixed 10 Sep 2026 before any
# campaign-4 job): configurations, and decoder C (per-position calibrated
# offset, cross-target validation).
# ---------------------------------------------------------------------------
CONFIGS4 = {
    "c4_a_fourvalue_16k_kingston": dict(prefixes=("c4_a_",), dataset="fourvalue", n=1, q=2, divider="nonrestoring"),
    "c4_b_fourvalue_4k_fez":       dict(prefixes=("c4_b_",), dataset="fourvalue", n=1, q=2, divider="nonrestoring"),
    "c4_c_fourvalue_n2_16k_kingston": dict(prefixes=("c4_c_",), dataset="fourvalue", n=2, q=2, divider="nonrestoring"),
    "c3+c4_fourvalue_kingston_pooled_runs": dict(prefixes=("c3_c3_", "c4_a_"), dataset="fourvalue", n=1, q=2, divider="nonrestoring"),
}


def per_position_offsets(cal_mats):
    """o_p = mean over calibration runs of (P[p] - onehot(true_p)), zero-mean over bins."""
    acc = None
    for _, P, t in cal_mats:
        O = P.copy()
        for px in range(len(t)):
            if t[px] >= 0:
                O[px, t[px]] -= 1.0
        acc = O if acc is None else acc + O
    o = acc / len(cal_mats)
    return o - o.mean(axis=1, keepdims=True)


def campaign4(out_json: Path):
    results = {}
    # --- hardware configurations, same scoring as campaign 3
    for name, cfg in CONFIGS4.items():
        runs = collect(cfg["prefixes"])
        if not runs:
            print(name, ": no runs yet"); continue
        images = load_images(cfg["dataset"], cfg["n"], cfg["q"])
        mats = run_matrices(runs, images, cfg["divider"], cfg["n"], cfg["q"])
        I_a, I_b = images
        R = np.where(I_b > 0, I_a // np.maximum(I_b, 1), 0); dz = I_b == 0
        null = max(max(int(((R == v) & ~dz).sum()) for v in range(4)), int(dz.sum()))
        res = {"n_runs": len(mats), "constant_readout_null": f"{null}/{R.size}",
               "cx_range": [min(m["two_q_gate_count"] for m, _ in runs), max(m["two_q_gate_count"] for m, _ in runs)],
               "shots": sorted({m["shots"] for m, _ in runs})}
        for dname, fn in [("argmax", dec_argmax), ("A_flatfield", dec_flatfield)]:
            s = score(mats, fn, null); conf = np.zeros((4, 4), int); resid = []
            for i, (label, P, t) in enumerate(mats):
                Q = fn(i, P, mats); dec = Q.argmax(axis=1)
                for px in range(len(t)):
                    if t[px] >= 0:
                        conf[t[px], dec[px]] += 1
                        if dname == "A_flatfield": resid.append(Q[px, t[px]])
            s["confusion_true_x_decoded"] = conf.tolist()
            if resid: s["residual_on_true_bin"] = [round(float(np.mean(resid)), 4), round(float(np.std(resid)), 4)]
            res[dname] = s
        results[name] = res
        print(f"\n=== {name}: {len(mats)} runs, shots {res['shots']}, null {null}/{R.size}, CX {res['cx_range']}")
        for dname in ("argmax", "A_flatfield"):
            s = res[dname]
            print(f"  {dname:12s} match {s['match_mean']} ± {s['match_sd']}  hist {s['match_hist']}  above-null {s['runs_above_null']}/{s['n_runs']} (p={s['sign_test_p']})  pixel acc {s['pixel_accuracy']}"
                  + (f"  residual {s.get('residual_on_true_bin')}" if 'residual_on_true_bin' in s else ""))
            print(f"               confusion: {s['confusion_true_x_decoded']}")
    # --- decoder C: calibrate on fourvalue@marrakesh (c3_c2_), score balanced@marrakesh (c3_c1_ + archived j16/j17/j17b)
    cal_runs = collect(("c3_c2_",))
    tgt_runs = collect(("c3_c1_",)) + collect(("j16_", "j17_", "j17b_"))
    if cal_runs and tgt_runs:
        cal = run_matrices(cal_runs, load_images("fourvalue", 1, 2), "nonrestoring", 1, 2)
        tgt = run_matrices(tgt_runs, load_images("canonical_shared", 1, 2), "nonrestoring", 1, 2)
        o = per_position_offsets(cal)
        def dec_C(i, P, mats): return P - o
        out = {}
        for dname, fn in [("argmax", dec_argmax), ("A_flatfield", dec_flatfield), ("C_per_position_calibrated", dec_C)]:
            out[dname] = score(tgt, fn, 2)
        results["decoder_C_balanced_marrakesh_calibrated_on_fourvalue_marrakesh"] = dict(
            n_calibration_runs=len(cal), n_target_runs=len(tgt), offsets=np.round(o, 4).tolist(), scores=out)
        print(f"\n=== decoder C: balanced@marrakesh ({len(tgt)} runs) calibrated on fourvalue@marrakesh ({len(cal)} runs)")
        for dname, s in out.items():
            print(f"  {dname:28s} match {s['match_mean']} ± {s['match_sd']}  hist {s['match_hist']}  above-null {s['runs_above_null']}/{s['n_runs']} (p={s['sign_test_p']})  pixel acc {s['pixel_accuracy']}")
    json.dump(results, open(out_json, "w"), indent=2); print("\nwrote", out_json)


if __name__ == "__main__" and "--campaign4" in sys.argv:
    campaign4(REPO / "paper/data_autonomous/campaign4_decoders.json")


# ---------------------------------------------------------------------------
# Campaign 5 (paper/HW_CAMPAIGN_5_PROTOCOL.md, fixed 10 Sep 2026 before any
# campaign-5 job): six pooled 4x4 runs on ibm_kingston (C4c + C5).
# ---------------------------------------------------------------------------
def campaign5(out_json: Path):
    import collections
    from math import comb
    cfg = dict(prefixes=("c4_c_", "c5_"), dataset="fourvalue", n=2, q=2, divider="nonrestoring")
    runs = collect(cfg["prefixes"])
    if not runs:
        print("no runs"); return
    images = load_images(cfg["dataset"], cfg["n"], cfg["q"])
    mats = run_matrices(runs, images, cfg["divider"], cfg["n"], cfg["q"])
    null = 4
    results = {"n_runs": len(mats), "labels": [m["label"] for m, _ in runs],
               "cx_range": [min(m["two_q_gate_count"] for m, _ in runs), max(m["two_q_gate_count"] for m, _ in runs)]}
    for dname, fn in [("argmax", dec_argmax), ("A_flatfield", dec_flatfield)]:
        s = score(mats, fn, null); conf = np.zeros((4, 4), int); resid = []; correct = 0; total = 0
        for i, (label, P, t) in enumerate(mats):
            Q = fn(i, P, mats); dec = Q.argmax(axis=1)
            for px in range(len(t)):
                if t[px] >= 0:
                    conf[t[px], dec[px]] += 1; total += 1; correct += int(dec[px] == t[px])
                    if dname == "A_flatfield": resid.append(Q[px, t[px]])
        # exact binomial P[X >= correct], X ~ Bin(total, 1/4), independence approximation
        pbin = sum(comb(total, k) * 0.25**k * 0.75**(total - k) for k in range(correct, total + 1))
        s.update(confusion_true_x_decoded=conf.tolist(), binomial_p_vs_chance=float(f"{pbin:.3g}"), correct=correct, total=total)
        if resid: s["residual_on_true_bin"] = [round(float(np.mean(resid)), 4), round(float(np.std(resid)), 4)]
        results[dname] = s
        print(f"\n  {dname:12s} match {s['match_mean']} ± {s['match_sd']}  per run {s['match_hist']}  above-null {s['runs_above_null']}/{s['n_runs']} (sign p={s['sign_test_p']})  pixel acc {s['pixel_accuracy']} ({correct}/{total}, binomial p={pbin:.2g})"
              + (f"  residual {s.get('residual_on_true_bin')}" if 'residual_on_true_bin' in s else ""))
        print(f"               confusion: {s['confusion_true_x_decoded']}")
    # pooled-shots curve
    rng = np.random.default_rng(0); N = len(mats); acc = {k: [] for k in range(1, N + 1)}
    def decode_pool(idx):
        c = collections.Counter()
        for i in idx:
            for k, v in runs[i][1].items(): c[k] += v
        r = analyse(dict(c), images, cfg["divider"], cfg["n"], cfg["q"])
        P = np.array([p["histogram"] for p in r["pixels"]]); t = np.array([-1 if p["divzero"] else p["true"] for p in r["pixels"]])
        dec = (P - P.mean(axis=0, keepdims=True)).argmax(axis=1); valid = t >= 0
        return int(((dec == t) & valid).sum()), int(valid.sum())
    for _ in range(100):
        order = rng.permutation(N)
        for k in range(1, N + 1):
            m, v = decode_pool(order[:k]); acc[k].append(m / v)
    curve = [(k * 16384 // 16, float(np.mean(a))) for k, a in acc.items()]
    m_all, v_all = decode_pool(range(N))
    results["accuracy_vs_shots_per_pixel"] = curve; results["pooled_all_runs_match"] = f"{m_all}/{v_all}"
    print(f"\n  pooled-shots curve (shots/px, accuracy): {[(a, round(b, 3)) for a, b in curve]}   all runs pooled: {m_all}/{v_all}")
    json.dump(results, open(out_json, "w"), indent=2); print("wrote", out_json)


if __name__ == "__main__" and "--campaign5" in sys.argv:
    campaign5(REPO / "paper/data_autonomous/campaign5_4x4.json")


# ---------------------------------------------------------------------------
# Campaign 6 (paper/HW_CAMPAIGN_6_PROTOCOL.md, fixed 10 Sep 2026 before any
# campaign-6 job): q = 3 target on kingston; cyclic per-position calibration
# on marrakesh validated on held-out targets (decoder C').
# ---------------------------------------------------------------------------
def _score_generic(mats, null, n_vals, decode):
    """Like score() but with an n_vals x n_vals confusion and no d-statistics."""
    matches, correct, total, resid = [], 0, 0, []
    conf = np.zeros((n_vals, n_vals), int)
    for i, (label, P, t) in enumerate(mats):
        Q = decode(i, P, mats); dec = Q.argmax(axis=1); valid = t >= 0
        m = int(((dec == t) & valid).sum()); matches.append(m); correct += m; total += int(valid.sum())
        for px in range(len(t)):
            if t[px] >= 0:
                conf[t[px], dec[px]] += 1; resid.append(Q[px, t[px]])
    m = np.array(matches, float); above = int((m > null).sum())
    return dict(match_mean=round(m.mean(), 3), match_sd=round(m.std(ddof=1), 3) if len(m) > 1 else None,
                match_hist={str(int(v)): int((m == v).sum()) for v in sorted(set(m))},
                runs_above_null=above, n_runs=len(m), sign_test_p=round(sign_test_p(above, len(m)), 4),
                pixel_accuracy=round(correct / total, 3), correct=correct, pixels=total,
                confusion_true_x_decoded=conf.tolist(),
                residual_on_true_bin=[round(float(np.mean(resid)), 4), round(float(np.std(resid)), 4)])


def campaign6(out_json: Path):
    results = {}
    # --- C6a: q = 3 on kingston
    runs = collect(("c6_a_",))
    if runs:
        images = load_images("fourvalue_q3", 1, 3)
        mats = run_matrices(runs, images, "nonrestoring", 1, 3)
        res = {"n_runs": len(mats), "constant_readout_null": "1/4", "chance": "1/8",
               "cx_range": [min(m["two_q_gate_count"] for m, _ in runs), max(m["two_q_gate_count"] for m, _ in runs)]}
        for dname, fn in [("argmax", dec_argmax), ("A_flatfield", dec_flatfield)]:
            res[dname] = _score_generic(mats, 1, 8, fn)
        results["c6_a_fourvalue_q3_16k_kingston"] = res
        print(f"\n=== C6a q=3 kingston: {len(mats)} runs, CX {res['cx_range']}, null 1/4, chance 1/8")
        for dname in ("argmax", "A_flatfield"):
            s = res[dname]
            print(f"  {dname:12s} match {s['match_mean']} ± {s['match_sd']}  per run {s['match_hist']}  above-null {s['runs_above_null']}/{s['n_runs']} (p={s['sign_test_p']})  pixel acc {s['pixel_accuracy']} ({s['correct']}/{s['pixels']})  residual {s['residual_on_true_bin']}")
            print(f"               confusion (8x8): {s['confusion_true_x_decoded']}")
    else:
        print("C6a: no runs yet")
    # --- C': cyclic calibration on marrakesh
    cal_sets = [("fourvalue", ("c3_c2_",)), ("fourvalue_p2", ("c6_b_fourvalue_p2_",)),
                ("fourvalue_p3", ("c6_b_fourvalue_p3_",)), ("fourvalue_p4", ("c6_b_fourvalue_p4_",))]
    cal_mats = []
    for ds, pref in cal_sets:
        r = collect(pref)
        if not r:
            print(f"C': calibration target {ds} has no runs yet"); cal_mats = None; break
        cal_mats += run_matrices(r, load_images(ds, 1, 2), "nonrestoring", 1, 2)
    if cal_mats:
        o = per_position_offsets(cal_mats)   # mean over targets and runs of (P - onehot), zero-meaned
        def dec_Cp(i, P, mats): return P - o
        out = {"offsets_per_position": np.round(o, 4).tolist(), "n_calibration_runs": len(cal_mats), "validation": {}}
        for vname, ds, pref, null in [("balanced_marrakesh", "canonical_shared", ("c3_c1_", "j16_", "j17_", "j17b_"), 2),
                                      ("fura2_marrakesh", "fura2", ("j14_fura2_trexonly_",), 2)]:
            vr = collect(pref)
            if not vr: continue
            vm = run_matrices(vr, load_images(ds, 1, 2), "nonrestoring", 1, 2)
            out["validation"][vname] = {dn: _score_generic(vm, null, 4, fn) for dn, fn in
                                        [("argmax", dec_argmax), ("A_flatfield", dec_flatfield), ("Cprime_cyclic_calibrated", dec_Cp)]}
            print(f"\n=== C' on {vname} ({len(vm)} held-out runs), offsets from {len(cal_mats)} calibration runs")
            for dn, s in out["validation"][vname].items():
                print(f"  {dn:26s} match {s['match_mean']} ± {s['match_sd']}  per run {s['match_hist']}  above-null {s['runs_above_null']}/{s['n_runs']} (p={s['sign_test_p']})  pixel acc {s['pixel_accuracy']}  residual {s['residual_on_true_bin']}")
        print("  estimated per-position offsets o_p (rows = pixels, cols = bins):", np.round(o, 3).tolist())
        results["cyclic_calibration_marrakesh"] = out
    json.dump(results, open(out_json, "w"), indent=2); print("\nwrote", out_json)


if __name__ == "__main__" and "--campaign6" in sys.argv:
    campaign6(REPO / "paper/data_autonomous/campaign6.json")


# ---------------------------------------------------------------------------
# Campaign 6 addendum (protocol addendum, 10 Sep 2026, written after C' was
# scored and before D was scored): decoder D, per-pixel template decoding.
# ---------------------------------------------------------------------------
def campaign6_D(out_json: Path):
    tg = [("fourvalue", ("c3_c2_",)), ("fourvalue_p2", ("c6_b_fourvalue_p2_",)),
          ("fourvalue_p3", ("c6_b_fourvalue_p3_",)), ("fourvalue_p4", ("c6_b_fourvalue_p4_",))]
    mats = {ds: run_matrices(collect(pref), load_images(ds, 1, 2), "nonrestoring", 1, 2) for ds, pref in tg}
    n_px, n_vals = 4, 4

    def templates(exclude=None):
        T = np.full((n_px, n_vals, n_vals), np.nan)
        for ds, ms in mats.items():
            if ds == exclude or not ms:
                continue
            P = np.mean([M for _, M, _ in ms], axis=0); t = ms[0][2]
            for px in range(n_px):
                T[px, t[px], :] = P[px]
        return T

    def make_dec(T):
        def dec(i, P, m):
            Q = np.full_like(P, -np.inf)
            for px in range(P.shape[0]):
                for v in range(n_vals):
                    if not np.isnan(T[px, v]).any():
                        Q[px, v] = float(np.sum(P[px] * np.log(np.clip(T[px, v], 1e-6, None))))
            return Q
        return dec

    results = {}
    T = templates()
    results["templates"] = np.round(T, 4).tolist()
    for vname, ds, pref, null in [("balanced_marrakesh", "canonical_shared", ("c3_c1_", "j16_", "j17_", "j17b_"), 2),
                                  ("fura2_marrakesh", "fura2", ("j14_fura2_trexonly_",), 2)]:
        vm = run_matrices(collect(pref), load_images(ds, 1, 2), "nonrestoring", 1, 2)
        results[vname] = {dn: _score_generic(vm, null, 4, fn) for dn, fn in
                          [("argmax", dec_argmax), ("A_flatfield", dec_flatfield), ("D_template", make_dec(T))]}
        print(f"\n=== D on {vname} ({len(vm)} held-out runs)")
        for dn, s in results[vname].items():
            print(f"  {dn:12s} match {s['match_mean']} ± {s['match_sd']}  per run {s['match_hist']}  above-null {s['runs_above_null']}/{s['n_runs']} (p={s['sign_test_p']})  pixel acc {s['pixel_accuracy']}  conf {s['confusion_true_x_decoded']}")
    print("\n=== D leave-one-target-out (secondary, biased: left-out value has no template)")
    loo = {}
    for ds in mats:
        Tx = templates(exclude=ds); s = _score_generic(mats[ds], 1, 4, make_dec(Tx)); loo[ds] = s
        print(f"  leave-{ds}-out: match {s['match_mean']} px acc {s['pixel_accuracy']}  conf {s['confusion_true_x_decoded']}")
    results["leave_one_target_out"] = loo
    json.dump(results, open(out_json, "w"), indent=2); print("\nwrote", out_json)


if __name__ == "__main__" and "--campaign6D" in sys.argv:
    campaign6_D(REPO / "paper/data_autonomous/campaign6_D.json")


# ---------------------------------------------------------------------------
# Campaign 7 (paper/HW_CAMPAIGN_7_PROTOCOL.md, fixed 10 Sep 2026 before any
# campaign-7 job): UCRY load, 4x4 and 8x8 on ibm_kingston.
# ---------------------------------------------------------------------------
CONFIGS7 = {
    "c7_a_fourvalue_n2_ucry_kingston": dict(prefixes=("c7_a_",), dataset="fourvalue", n=2, q=2, divider="nonrestoring"),
    "c7_b_random4_n2_ucry_kingston":   dict(prefixes=("c7_b_",), dataset="random4", n=2, q=2, divider="nonrestoring"),
    "c7_c_laurdan_n2_ucry_kingston":   dict(prefixes=("c7_c_",), dataset="canonical_shared", n=2, q=2, divider="nonrestoring"),
    "c7_d_random4_n3_ucry_kingston":   dict(prefixes=("c7_d_",), dataset="random4", n=3, q=2, divider="nonrestoring"),
    # Campaign 9 (paper/HW_CAMPAIGN_9_PROTOCOL.md, fixed 10 Sep 2026): 4x4 tiled on ibm_marrakesh.
    "c9_fourvalue_n2_ucry_marrakesh":  dict(prefixes=("c9_",), dataset="fourvalue", n=2, q=2, divider="nonrestoring"),
    # Campaign 10 (paper/HW_CAMPAIGN_10_PROTOCOL.md, fixed 11 Sep 2026).
    "c10_a_fourvalue_n2_ucry_fez":            dict(prefixes=("c10_a_",), dataset="fourvalue", n=2, q=2, divider="nonrestoring"),
    "c9+c10b_fourvalue_n2_ucry_marrakesh":    dict(prefixes=("c9_", "c10_b_"), dataset="fourvalue", n=2, q=2, divider="nonrestoring"),
    "c7c+c10c_laurdan_n2_ucry_kingston":      dict(prefixes=("c7_c_", "c10_c_"), dataset="canonical_shared", n=2, q=2, divider="nonrestoring"),
    "c10_d_laurdan_n3_ucry_kingston":         dict(prefixes=("c10_d_",), dataset="canonical_shared", n=3, q=2, divider="nonrestoring"),
    "c10_e_random4_n2_ucry_marrakesh":        dict(prefixes=("c10_e_",), dataset="random4", n=2, q=2, divider="nonrestoring"),
    # Campaign 12b (paper/HW_CAMPAIGN_12_PROTOCOL.md, fixed 11 Sep 2026): balanced 8x8 on ibm_kingston.
    "c12_b_balanced8_n3_ucry_kingston":       dict(prefixes=("c12_b_",), dataset="balanced8", n=3, q=2, divider="nonrestoring"),
}


def campaign7(out_json: Path):
    from math import comb
    results = {}
    for name, cfg in CONFIGS7.items():
        runs = collect(cfg["prefixes"])
        if not runs:
            print(name, ": no runs yet"); continue
        images = load_images(cfg["dataset"], cfg["n"], cfg["q"])
        mats = run_matrices(runs, images, cfg["divider"], cfg["n"], cfg["q"])
        I_a, I_b = images
        R = np.where(I_b > 0, I_a // np.maximum(I_b, 1), 0); dz = I_b == 0
        null = max(max(int(((R == v) & ~dz).sum()) for v in range(4)), int(dz.sum()))
        res = {"n_runs": len(mats), "constant_readout_null": f"{null}/{R.size}", "valid_pixels": int((~dz).sum()),
               "cx_range": [min(m["two_q_gate_count"] for m, _ in runs), max(m["two_q_gate_count"] for m, _ in runs)],
               "shots": sorted({m["shots"] for m, _ in runs})}
        for dname, fn in [("argmax", dec_argmax), ("A_flatfield", dec_flatfield)]:
            s = _score_generic(mats, null, 4, fn)
            c, t = s["correct"], s["pixels"]
            s["binomial_p_vs_chance"] = float(f"{sum(comb(t, k) * 0.25**k * 0.75**(t - k) for k in range(c, t + 1)):.3g}")
            res[dname] = s
        results[name] = res
        print(f"\n=== {name}: {len(mats)} runs, shots {res['shots']}, null {null}/{R.size}, valid {res['valid_pixels']}, CX {res['cx_range']}")
        for dname in ("argmax", "A_flatfield"):
            s = res[dname]
            print(f"  {dname:12s} match {s['match_mean']} ± {s['match_sd']}  per run {s['match_hist']}  above-null {s['runs_above_null']}/{s['n_runs']} (p={s['sign_test_p']})  pixel acc {s['pixel_accuracy']} ({s['correct']}/{s['pixels']}, binomial p={s['binomial_p_vs_chance']})  residual {s['residual_on_true_bin']}")
            print(f"               confusion: {s['confusion_true_x_decoded']}")
    json.dump(results, open(out_json, "w"), indent=2); print("\nwrote", out_json)


if __name__ == "__main__" and "--campaign7" in sys.argv:
    campaign7(REPO / "paper/data_autonomous/campaign7_ucry.json")


# ---------------------------------------------------------------------------
# Campaign 8 (paper/HW_CAMPAIGN_8_PROTOCOL.md, fixed 10 Sep 2026 before any
# campaign-8 job): 8x8 random (pooled with C7d) and 8x8 Fura-2 on ibm_kingston.
# ---------------------------------------------------------------------------
CONFIGS8 = {
    "c7d+c8a_random4_n3_ucry_kingston": dict(prefixes=("c7_d_", "c8_a_"), dataset="random4", n=3, q=2, divider="nonrestoring"),
    "c8_b_fura2_n3_ucry_kingston":      dict(prefixes=("c8_b_",), dataset="fura2", n=3, q=2, divider="nonrestoring"),
}


def campaign8(out_json: Path):
    from math import comb
    results = {}
    for name, cfg in CONFIGS8.items():
        runs = collect(cfg["prefixes"])
        if not runs:
            print(name, ": no runs yet"); continue
        images = load_images(cfg["dataset"], cfg["n"], cfg["q"])
        mats = run_matrices(runs, images, cfg["divider"], cfg["n"], cfg["q"])
        I_a, I_b = images
        R = np.where(I_b > 0, I_a // np.maximum(I_b, 1), 0); dz = I_b == 0
        null = max(max(int(((R == v) & ~dz).sum()) for v in range(4)), int(dz.sum()))
        res = {"n_runs": len(mats), "constant_readout_null": f"{null}/{R.size}", "valid_pixels": int((~dz).sum()),
               "cx_range": [min(m["two_q_gate_count"] for m, _ in runs), max(m["two_q_gate_count"] for m, _ in runs)],
               "shots": sorted({m["shots"] for m, _ in runs})}
        for dname, fn in [("argmax", dec_argmax), ("A_flatfield", dec_flatfield)]:
            s = _score_generic(mats, null, 4, fn); c, t = s["correct"], s["pixels"]
            s["binomial_p_vs_chance"] = float(f"{sum(comb(t, k) * 0.25**k * 0.75**(t - k) for k in range(c, t + 1)):.3g}")
            res[dname] = s
        results[name] = res
        print(f"\n=== {name}: {len(mats)} runs, shots {res['shots']}, null {null}/{R.size}, valid {res['valid_pixels']}, CX {res['cx_range']}")
        for dname in ("argmax", "A_flatfield"):
            s = res[dname]
            print(f"  {dname:12s} match {s['match_mean']} ± {s['match_sd']}  per run {s['match_hist']}  above-null {s['runs_above_null']}/{s['n_runs']} (p={s['sign_test_p']})  pixel acc {s['pixel_accuracy']} ({s['correct']}/{s['pixels']}, binomial p={s['binomial_p_vs_chance']})  residual {s['residual_on_true_bin']}")
            print(f"               confusion: {s['confusion_true_x_decoded']}")
    json.dump(results, open(out_json, "w"), indent=2); print("\nwrote", out_json)


if __name__ == "__main__" and "--campaign8" in sys.argv:
    campaign8(REPO / "paper/data_autonomous/campaign8_8x8.json")
