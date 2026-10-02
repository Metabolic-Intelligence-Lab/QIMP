"""Content-corrected flat-field decoder (label-free), specified 11 Sep 2026
before being scored on the archived runs.

Decoder A (flat-field) subtracts the mean histogram over pixels, which on an
image dominated by one value contains that value's own signal. Decoder E
removes the estimated content before averaging:
    d^(0) = flat-field decode
    repeat t = 1..T:
        m  = mean_p [ P_p - s * onehot(d^(t-1)_p) ]        (common mode with content removed)
        d^(t) = argmax_v ( P_p - m )_v
with s the fraction of signal on the true bin, estimated per run as the mean
over pixels of (P_p[d_p] - mean of the other bins) after the first pass, and
T = 5. No labels, no calibration runs. Scored beside argmax and A on every
archived configuration; the pre-specified success criterion is the constant
read-out null over valid pixels, as in Table 13.
Usage: .venv/bin/python scripts/decode_content_corrected.py
"""
from __future__ import annotations
import json, sys
from math import comb
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from decode_bias_corrected import collect, run_matrices, sign_test_p
from analyse_hw_signal import load_images

def dec_E(P, T=5):
    n, k = P.shape
    d = (P - P.mean(axis=0, keepdims=True)).argmax(axis=1)
    for _ in range(T):
        s = float(np.mean([P[p, d[p]] - np.mean(np.delete(P[p], d[p])) for p in range(n)]))
        s = max(s, 0.0)
        onehot = np.zeros_like(P); onehot[np.arange(n), d] = s
        m = (P - onehot).mean(axis=0, keepdims=True)
        d_new = (P - m).argmax(axis=1)
        if np.array_equal(d_new, d): break
        d = d_new
    return d

TESTS = [("2x2 balanced marrakesh (20)", "canonical_shared", ("c3_c1_", "j16_", "j17_", "j17b_"), 1, 2, "nonrestoring"),
         ("2x2 Fura-2 marrakesh (5)", "fura2", ("j14_fura2_trexonly_",), 1, 2, "nonrestoring"),
         ("2x2 four-value kingston (8)", "fourvalue", ("c3_c3_", "c4_a_"), 1, 2, "nonrestoring"),
         ("2x2 four-value marrakesh (5)", "fourvalue", ("c3_c2_",), 1, 2, "nonrestoring"),
         ("2x2 four-value fez (3)", "fourvalue", ("c4_b_",), 1, 2, "nonrestoring"),
         ("4x4 tiled kingston (4)", "fourvalue", ("c7_a_",), 2, 2, "nonrestoring"),
         ("4x4 random kingston (4)", "random4", ("c7_b_",), 2, 2, "nonrestoring"),
         ("4x4 Laurdan kingston (6)", "canonical_shared", ("c7_c_", "c10_c_"), 2, 2, "nonrestoring"),
         ("4x4 tiled marrakesh (6)", "fourvalue", ("c9_", "c10_b_"), 2, 2, "nonrestoring"),
         ("4x4 random marrakesh (3)", "random4", ("c10_e_",), 2, 2, "nonrestoring"),
         ("4x4 tiled fez (3)", "fourvalue", ("c10_a_",), 2, 2, "nonrestoring"),
         ("8x8 random kingston (5)", "random4", ("c7_d_", "c8_a_"), 3, 2, "nonrestoring"),
         ("8x8 Fura-2 kingston (3)", "fura2", ("c8_b_",), 3, 2, "nonrestoring"),
         ("8x8 Laurdan kingston (3)", "canonical_shared", ("c10_d_",), 3, 2, "nonrestoring")]

def main():
    out = {}
    print(f"{'configuration':30s} {'null':>7s} | {'argmax':>16s} | {'A flat-field':>16s} | {'E content-corrected':>20s}")
    for name, ds, pref, n, q, div in TESTS:
        runs = collect(pref); images = load_images(ds, n, q); I_a, I_b = images
        mats = run_matrices(runs, images, div, n, q)
        R = np.where(I_b > 0, I_a // np.maximum(I_b, 1), -1); valid_mask = R >= 0
        null = max(int((R == v).sum()) for v in range(4)); nvalid = int(valid_mask.sum())
        res = {}
        for dn, fn in [("argmax", lambda P: P.argmax(axis=1)), ("A", lambda P: (P - P.mean(axis=0, keepdims=True)).argmax(axis=1)), ("E", dec_E)]:
            matches = []; correct = 0; total = 0
            for label, P, t in mats:
                d = fn(P); v = t >= 0; m = int(((d == t) & v).sum()); matches.append(m); correct += m; total += int(v.sum())
            above = sum(1 for m in matches if m > null)
            res[dn] = dict(match_mean=float(np.mean(matches)), above=f"{above}/{len(matches)}", p=sign_test_p(above, len(matches)), acc=correct / total, per_run=matches)
        out[name] = dict(null=f"{null}/{nvalid}", **res)
        f = lambda r: f"{r['match_mean']:5.2f} {r['above']:>4s} {r['acc']:.2f}"
        print(f"{name:30s} {null:3d}/{nvalid:<3d} | {f(res['argmax']):>16s} | {f(res['A']):>16s} | {f(res['E']):>20s}")
    json.dump(out, open(REPO / "paper/data_autonomous/decoder_E_content_corrected.json", "w"), indent=2)

if __name__ == "__main__":
    main()
