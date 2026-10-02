"""Score campaign 19 against the bands of paper/HW_CAMPAIGN_19_PROTOCOL.md.

B1: P(good) at k = 0 below 0.46 in at least 2 of 3 runs, run mean within 0.10 of the ideal.
B2: mean P(good) at k = 1 above mean at k = 0 by at least 6 pooled standard errors.
B4: the maximum-likelihood estimate of the amplitude from the measured p_k, reported as is.
Secondary: the depolarised prediction 0.5 + F (P_ideal - 0.5), F = exp(-N_2q * epsilon).
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

def load():
    runs = {}
    for d in sorted(glob.glob(str(REPO / "data/output/ibm_hw/*/runs/c19_k*_hw"))):
        m = json.load(open(Path(d) / "metadata.json"))
        runs.setdefault(m["grover_k"], []).append(m)
    return runs

def mlqae(p: dict[int, float], shots: int) -> tuple[float, float, float]:
    """Grid maximum-likelihood over a, with the likelihood of independent binomials."""
    a = np.linspace(1e-4, 1 - 1e-4, 20001); theta = np.arcsin(np.sqrt(a))
    ll = np.zeros_like(a)
    for k, ph in p.items():
        pk = np.sin((2 * k + 1) * theta) ** 2
        pk = np.clip(pk, 1e-9, 1 - 1e-9)
        ll += shots * (ph * np.log(pk) + (1 - ph) * np.log(1 - pk))
    i = int(np.argmax(ll)); lo, hi = ll.max() - 0.5 * 3.84, None  # 95 % likelihood interval
    inside = a[ll >= lo]
    return float(a[i]), float(inside.min()), float(inside.max())

def main():
    runs = load()
    if not runs:
        print("no campaign-19 runs"); return
    shots = runs[min(runs)][0]["shots"]; eps = 2.5e-3
    out = {}
    print(f"{'k':>2} {'runs':>4} {'2q':>5} {'P(good) per run':>34} {'mean':>7} {'ideal':>7} {'depolarised':>12}")
    for k in sorted(runs):
        ms = runs[k]; ps = [m["p_measured"] for m in ms]; n2 = ms[0]["two_q_gate_count"]
        dep = 0.5 + np.exp(-n2 * eps) * (ms[0]["p_ideal"] - 0.5)
        out[k] = dict(runs=len(ps), two_q=n2, p=ps, mean=float(np.mean(ps)), ideal=ms[0]["p_ideal"], depolarised=float(dep),
                      accounts=sorted({m.get("job_id", "")[:4] for m in ms}))
        print(f"{k:2d} {len(ps):4d} {n2:5d} {str([round(x,4) for x in ps]):>34} {np.mean(ps):7.4f} {ms[0]['p_ideal']:7.4f} {dep:12.4f}")
    se = lambda ps: float(np.sqrt(sum(p * (1 - p) / shots for p in ps)) / len(ps))
    b1 = sum(1 for p in out[0]["p"] if p < 0.46) >= 2 and abs(out[0]["mean"] - out[0]["ideal"]) <= 0.10
    print(f"\nB1 (k=0 carries the amplitude): {'PASS' if b1 else 'FAIL'}")
    if 1 in out:
        d = out[1]["mean"] - out[0]["mean"]; sd = float(np.sqrt(se(out[0]["p"]) ** 2 + se(out[1]["p"]) ** 2))
        print(f"B2 (amplification visible): difference {d:+.4f} = {d/sd:.1f} standard errors -> {'PASS' if d/sd >= 6 else 'FAIL'}")
        out["B2_sigma"] = d / sd
    est, lo, hi = mlqae({k: out[k]["mean"] for k in out if isinstance(k, int)}, shots)
    print(f"B4 (estimate): a_hat = {est:.3f}, 95 % likelihood interval [{lo:.3f}, {hi:.3f}], a_true = 0.250")
    out["mlqae"] = dict(a_hat=est, lo=lo, hi=hi, a_true=0.25); out["B1"] = bool(b1)
    json.dump({str(k): v for k, v in out.items()}, open(REPO / "paper/data_autonomous/campaign19_scores.json", "w"), indent=2)

if __name__ == "__main__":
    main()
