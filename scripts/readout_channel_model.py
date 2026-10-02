"""Value-dependent read-out of basis-encoded arithmetic registers: a per-bit
asymmetric-flip channel model, fitted on the cyclic calibration targets and
validated on held-out circuits (paper v2, §6.4 addendum / §8).

Model. For a q-bit output register read after a deep circuit, the observed
histogram for true value v is
    P(obs = w | v) = (1 - f) * prod_b K_b(w_b | v_b) + f / 2^q
with a per-bit binary channel K_b (flip rates e01_b = P(1|0), e10_b = P(1->0))
and a depolarising floor f. Parameters: 2q + 1 (shared over pixels) or per
pixel. Fitted by multinomial maximum likelihood on the four cyclic targets
(every pixel sees every value once), compared against (i) the free 4x4
template per pixel (12 free parameters per pixel) and (ii) a symmetric-flip
model (e01 = e10) by held-out log-likelihood and by decode accuracy on targets
never used in the fit (balanced and Fura-2 runs on ibm_marrakesh), and on a
different circuit (the restoring divider) and a different width (q = 3).

Usage: .venv/bin/python scripts/readout_channel_model.py
"""
from __future__ import annotations
import json, sys
from itertools import product
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from decode_bias_corrected import collect, run_matrices, dec_argmax, dec_flatfield, _score_generic
from analyse_hw_signal import load_images

def bits(v, q): return [(v >> b) & 1 for b in range(q)]

def channel_matrix(params, q):
    """params = [e01_0, e10_0, ..., e01_{q-1}, e10_{q-1}, f] -> T[v, w]."""
    e = params[:2 * q]; f = params[2 * q]
    T = np.zeros((2 ** q, 2 ** q))
    for v in range(2 ** q):
        for w in range(2 ** q):
            p = 1.0
            for b in range(q):
                vb, wb = (v >> b) & 1, (w >> b) & 1
                e01, e10 = e[2 * b], e[2 * b + 1]
                p *= (e01 if wb else 1 - e01) if vb == 0 else ((1 - e10) if wb else e10)
            T[v, w] = (1 - f) * p + f / 2 ** q
    return T

def fit(hists, truths, q, symmetric=False):
    """hists: list of (histogram over 2^q bins, shots) ; truths: true values."""
    def nll(x):
        if symmetric:
            e = np.repeat(x[:q], 2); params = np.concatenate([e, [x[q]]])
        else:
            params = x
        params = np.clip(params, 1e-4, 0.999)
        T = channel_matrix(params, q); ll = 0.0
        for (h, shots), v in zip(hists, truths):
            ll += shots * np.sum(h * np.log(np.clip(T[v], 1e-9, None)))
        return -ll
    x0 = np.full(q + 1, 0.15) if symmetric else np.full(2 * q + 1, 0.15)
    bnds = [(1e-4, 0.999)] * len(x0)
    r = minimize(nll, x0, bounds=bnds, method="L-BFGS-B")
    if symmetric:
        return np.concatenate([np.repeat(r.x[:q], 2), [r.x[q]]]), -r.fun
    return r.x, -r.fun

def dec_channel(T):
    def dec(i, P, mats):
        Q = np.zeros_like(P)
        for px in range(P.shape[0]):
            for v in range(T.shape[0]):
                Q[px, v] = np.sum(P[px] * np.log(np.clip(T[v], 1e-9, None)))
        return Q
    return dec

def main():
    out = {}
    q = 2
    # --- calibration data: four cyclic targets on ibm_marrakesh (campaign 3 C2 + campaign 6)
    cal = [("fourvalue", ("c3_c2_",)), ("fourvalue_p2", ("c6_b_fourvalue_p2_",)), ("fourvalue_p3", ("c6_b_fourvalue_p3_",)), ("fourvalue_p4", ("c6_b_fourvalue_p4_",))]
    hists, truths, per_px = [], [], {}
    for ds, pref in cal:
        runs = collect(pref); mats = run_matrices(runs, load_images(ds, 1, 2), "nonrestoring", 1, 2)
        for (label, P, t), (meta, _) in zip(mats, runs):
            shots_px = meta["shots"] / 4
            for px in range(4):
                hists.append((P[px], shots_px)); truths.append(int(t[px]))
                per_px.setdefault(px, ([], [])); per_px[px][0].append((P[px], shots_px)); per_px[px][1].append(int(t[px]))
    n_cal = len(hists)
    p_shared, ll_shared = fit(hists, truths, q); p_sym, ll_sym = fit(hists, truths, q, symmetric=True)
    px_params = {px: fit(h, t, q)[0] for px, (h, t) in per_px.items()}
    ll_px = sum(fit(h, t, q)[1] for px, (h, t) in per_px.items())
    # free template log-likelihood (upper bound): per pixel per value mean histogram
    ll_free = 0.0
    for px, (h, t) in per_px.items():
        for v in range(4):
            H = [hh for (hh, s), tt in zip(h, t) if tt == v]; S = [s for (hh, s), tt in zip(h, t) if tt == v]
            m = np.mean(H, axis=0)
            for hh, s in zip(H, S): ll_free += s * np.sum(hh * np.log(np.clip(m, 1e-9, None)))
    print(f"calibration: {n_cal} pixel-histograms on ibm_marrakesh")
    print(f"  shared asymmetric flip model (5 params): e01={p_shared[0::2][:q].round(3)} e10={p_shared[1::2][:q].round(3)} f={p_shared[4]:.3f}  logL={ll_shared:.0f}")
    print(f"  shared symmetric  flip model (3 params): e={p_sym[0::2][:q].round(3)} f={p_sym[4]:.3f}  logL={ll_sym:.0f}")
    print(f"  per-pixel asymmetric (20 params):        logL={ll_px:.0f}   | free templates (48 params): logL={ll_free:.0f}")
    for px, pp in px_params.items():
        print(f"    pixel {px}: e01={pp[0::2][:q].round(3)} e10={pp[1::2][:q].round(3)} f={pp[4]:.3f}")
    out["calibration"] = dict(n_hist=n_cal, shared=dict(params=p_shared.tolist(), logL=ll_shared), symmetric=dict(params=p_sym.tolist(), logL=ll_sym),
                              per_pixel=dict(params={k: v.tolist() for k, v in px_params.items()}, logL=ll_px), free_templates_logL=ll_free)
    # --- validation on held-out targets (same device, same circuit family)
    T_shared = channel_matrix(p_shared, q); T_sym = channel_matrix(p_sym, q)
    print("\nvalidation (held-out targets, ibm_marrakesh, non-restoring 2x2):")
    for vname, ds, pref, null in [("balanced (20 runs)", "canonical_shared", ("c3_c1_", "j16_", "j17_", "j17b_"), 2), ("Fura-2 (5 runs)", "fura2", ("j14_fura2_trexonly_",), 2)]:
        vm = run_matrices(collect(pref), load_images(ds, 1, 2), "nonrestoring", 1, 2)
        res = {}
        for dn, fn in [("argmax", dec_argmax), ("flat-field", dec_flatfield), ("channel shared", dec_channel(T_shared)), ("channel symmetric", dec_channel(T_sym))]:
            s = _score_generic(vm, null, 4, fn); res[dn] = dict(match=s["match_mean"], px=s["pixel_accuracy"], above=f"{s['runs_above_null']}/{s['n_runs']}")
            print(f"  {vname:18s} {dn:18s} match {s['match_mean']:.2f}  pixel acc {s['pixel_accuracy']:.3f}  above null {s['runs_above_null']}/{s['n_runs']}")
        out[f"validation_{vname}"] = res
    # --- transfer to a different circuit (restoring divider, balanced) and different width (q=3 kingston, fitted on its own runs? no calibration -> apply shared marrakesh model? different device) 
    print("\ntransfer to a different circuit (restoring divider, balanced target, ibm_marrakesh, 5 runs):")
    vm = run_matrices(collect(("j18_",)), load_images("canonical_shared", 1, 2), "restoring", 1, 2); res = {}
    for dn, fn in [("argmax", dec_argmax), ("flat-field", dec_flatfield), ("channel shared (non-restoring fit)", dec_channel(T_shared))]:
        s = _score_generic(vm, 2, 4, fn); res[dn] = dict(match=s["match_mean"], px=s["pixel_accuracy"])
        print(f"  {dn:34s} match {s['match_mean']:.2f}  pixel acc {s['pixel_accuracy']:.3f}")
    out["transfer_restoring"] = res
    # --- transfer to 4x4 on the same device (campaigns 9-10, marrakesh)
    print("\ntransfer to 4x4 (tiled + random) on ibm_marrakesh, uniformly controlled load, same 2x2-fitted channel:")
    for vname, ds, pref, null in [("4x4 tiled (6 runs)", "fourvalue", ("c9_", "c10_b_"), 4), ("4x4 random (3 runs)", "random4", ("c10_e_",), 7)]:
        vm = run_matrices(collect(pref), load_images(ds, 2, 2), "nonrestoring", 2, 2); res = {}
        for dn, fn in [("argmax", dec_argmax), ("flat-field", dec_flatfield), ("channel shared", dec_channel(T_shared))]:
            s = _score_generic(vm, null, 4, fn); res[dn] = dict(match=s["match_mean"], px=s["pixel_accuracy"])
            print(f"  {vname:20s} {dn:16s} match {s['match_mean']:.2f}  pixel acc {s['pixel_accuracy']:.3f}")
        out[f"transfer_{vname}"] = res
    json.dump(out, open(REPO / "paper/data_autonomous/readout_channel_model.json", "w"), indent=2, default=float)
    print("\nwrote paper/data_autonomous/readout_channel_model.json")

if __name__ == "__main__" and "--part2" not in sys.argv:
    main()


# ---------------------------------------------------------------------------
# Part 2 (11 Sep 2026): label-free EM decoder with the shared channel, and
# per-device channel parameters.
# ---------------------------------------------------------------------------
def em_decode(P, q, shots_px, n_iter=8, init=None):
    """Unsupervised: alternate (E) posterior over v per pixel under channel T,
    (M) refit the 2q+1 channel parameters on the soft labels. Returns decoded
    values and the fitted parameters. init: initial channel params."""
    n_vals = 2 ** q
    params = np.array(init) if init is not None else np.full(2 * q + 1, 0.15)
    for _ in range(n_iter):
        T = channel_matrix(np.clip(params, 1e-4, 0.999), q)
        logpost = np.array([[shots_px * np.sum(P[px] * np.log(np.clip(T[v], 1e-9, None))) for v in range(n_vals)] for px in range(P.shape[0])])
        logpost -= logpost.max(axis=1, keepdims=True); post = np.exp(logpost); post /= post.sum(axis=1, keepdims=True)
        def nll(x):
            Tx = channel_matrix(np.clip(x, 1e-4, 0.999), q); ll = 0.0
            for px in range(P.shape[0]):
                for v in range(n_vals):
                    if post[px, v] > 1e-6:
                        ll += post[px, v] * shots_px * np.sum(P[px] * np.log(np.clip(Tx[v], 1e-9, None)))
            return -ll
        r = minimize(nll, params, bounds=[(1e-4, 0.999)] * len(params), method="L-BFGS-B"); params = r.x
    T = channel_matrix(np.clip(params, 1e-4, 0.999), q)
    dec = np.array([np.argmax([np.sum(P[px] * np.log(np.clip(T[v], 1e-9, None))) for v in range(n_vals)]) for px in range(P.shape[0])])
    return dec, params


def part2():
    q = 2; out = {}
    print("\n=== per-device channel parameters (shared model, fitted on four-value 2x2 runs of each device)")
    dev = {"ibm_marrakesh": ("fourvalue", ("c3_c2_",)), "ibm_kingston": ("fourvalue", ("c3_c3_", "c4_a_")), "ibm_fez": ("fourvalue", ("c4_b_",))}
    for name, (ds, pref) in dev.items():
        runs = collect(pref); mats = run_matrices(runs, load_images(ds, 1, 2), "nonrestoring", 1, 2)
        hists, truths = [], []
        for (label, P, t), (meta, _) in zip(mats, runs):
            for px in range(4): hists.append((P[px], meta["shots"] / 4)); truths.append(int(t[px]))
        p, ll = fit(hists, truths, q)
        print(f"  {name:14s} ({len(mats)} runs): bit0 e01={p[0]:.3f} e10={p[1]:.3f} | bit1 e01={p[2]:.3f} e10={p[3]:.3f} | f={p[4]:.3f}")
        out[name] = p.tolist()
    print("\n=== label-free EM channel decoder vs flat-field (held-out runs)")
    tests = [("balanced 2x2 marrakesh (20)", "canonical_shared", ("c3_c1_", "j16_", "j17_", "j17b_"), 1, 2, "nonrestoring"),
             ("Fura-2 2x2 marrakesh (5)", "fura2", ("j14_fura2_trexonly_",), 1, 2, "nonrestoring"),
             ("four-value 2x2 kingston (8)", "fourvalue", ("c3_c3_", "c4_a_"), 1, 2, "nonrestoring"),
             ("four-value 2x2 fez (3)", "fourvalue", ("c4_b_",), 1, 2, "nonrestoring"),
             ("4x4 tiled marrakesh (6)", "fourvalue", ("c9_", "c10_b_"), 2, 2, "nonrestoring"),
             ("4x4 random marrakesh (3)", "random4", ("c10_e_",), 2, 2, "nonrestoring"),
             ("4x4 tiled fez (3)", "fourvalue", ("c10_a_",), 2, 2, "nonrestoring"),
             ("4x4 random kingston (4)", "random4", ("c7_b_",), 2, 2, "nonrestoring"),
             ("4x4 Laurdan kingston (6)", "canonical_shared", ("c7_c_", "c10_c_"), 2, 2, "nonrestoring"),
             ("8x8 random kingston (5)", "random4", ("c7_d_", "c8_a_"), 3, 2, "nonrestoring"),
             ("8x8 Laurdan kingston (3)", "canonical_shared", ("c10_d_",), 3, 2, "nonrestoring"),
             ("balanced 2x2 restoring marrakesh (5)", "canonical_shared", ("j18_",), 1, 2, "restoring")]
    for name, ds, pref, n, qq, div in tests:
        runs = collect(pref); mats = run_matrices(runs, load_images(ds, n, qq), div, n, qq)
        acc = {"argmax": [0, 0], "flat-field": [0, 0], "EM channel": [0, 0]}
        for (label, P, t), (meta, _) in zip(mats, runs):
            valid = t >= 0; shots_px = meta["shots"] / P.shape[0]
            d_arg = P.argmax(axis=1); d_ff = (P - P.mean(axis=0, keepdims=True)).argmax(axis=1)
            d_em, _ = em_decode(P, qq, shots_px, init=np.array([0.3, 0.3, 0.3, 0.3, 0.05]))
            for k, d in [("argmax", d_arg), ("flat-field", d_ff), ("EM channel", d_em)]:
                acc[k][0] += int(((d == t) & valid).sum()); acc[k][1] += int(valid.sum())
        line = "  ".join(f"{k} {a[0]}/{a[1]} ({a[0]/a[1]:.2f})" for k, a in acc.items())
        print(f"  {name:38s} {line}")
        out[f"em_{name}"] = {k: a for k, a in acc.items()}
    json.dump(out, open(REPO / "paper/data_autonomous/readout_channel_model_part2.json", "w"), indent=2)


if __name__ == "__main__" and "--part2" in sys.argv:
    part2()
