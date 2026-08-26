"""
Figures for the extended sim sweep + extended QAE + extended HW table.

Produces:
  fig11_highshot_sweep.png   - Class B match% vs n with high shots (3 datasets)
  fig12_qae_extension.png    - MLQAE advantage vs total query budget at n=1..4
  fig13_hw_neqr_ceiling.png  - HW NEQR-only match% vs N_2q for J2/J5/J6
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_OUT = REPO / "paper" / "data_autonomous"
FIG_OUT = REPO / "paper" / "figures_autonomous"
HW_ROOT = REPO / "data" / "output" / "ibm_hw"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# fig11: high-shot Class B match% vs n
# ---------------------------------------------------------------------------

def build_fig11() -> None:
    sweep_path = DATA_OUT / "scaling_sweep.csv"
    highshot_path = DATA_OUT / "highshot_sweep.csv"
    if not sweep_path.exists() or not highshot_path.exists():
        print("fig11: missing CSV inputs; skipping")
        return

    def load_csv(p: Path) -> list[dict]:
        with open(p) as f:
            return list(csv.DictReader(f))

    base = load_csv(sweep_path)
    hi = load_csv(highshot_path)

    datasets = ["canonical", "fura2", "rogfp2"]
    labels = {"canonical": "Laurdan canonical",
              "fura2": "Fura-2 synthetic",
              "rogfp2": "roGFP2 synthetic"}
    colors = {"canonical": "C0", "fura2": "C1", "rogfp2": "C2"}

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.6), constrained_layout=True)
    for ds in datasets:
        rows_b = [r for r in base
                  if r.get("dataset") == ds and r.get("match_pct") not in (None, "")]
        rows_h = [r for r in hi
                  if r.get("dataset") == ds and r.get("match_pct") not in (None, "")]
        nb = [int(r["n"]) for r in rows_b]
        pb = [float(r["match_pct"]) for r in rows_b]
        ax.plot(nb, pb, "o--", color=colors[ds], alpha=0.45,
                label=f"{labels[ds]} (4 shots/px)")
        nh = [int(r["n"]) for r in rows_h]
        ph = [float(r["match_pct"]) for r in rows_h]
        ax.plot(nh, ph, "s-", color=colors[ds],
                label=f"{labels[ds]} (16 shots/px)")
    ax.axhline(100, color="0.5", ls=":", lw=0.8)
    ax.set_xlabel(r"spatial qubits $n$ (image side $= 2^n$)")
    ax.set_ylabel(r"pixelwise match vs classical (%)")
    ax.set_xticks(range(2, 7))
    ax.set_ylim(80, 102)
    ax.set_title("Class B Aer-MPS scaling sweep — shot-budget effect")
    ax.legend(fontsize=8, loc="lower left")
    out = FIG_OUT / "fig11_highshot_sweep.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# fig12: QAE empirical advantage vs n (all available datapoints)
# ---------------------------------------------------------------------------

def build_fig12() -> None:
    ext_path = DATA_OUT / "qae_extension.json"
    ext = []
    if ext_path.exists():
        try:
            ext = json.loads(ext_path.read_text())
        except Exception:
            ext = []

    # Hard-coded historical n=1, n=2 results from the paper (canonical Laurdan)
    historical = [
        {"n": 1, "side": 2, "a_true": 0.75, "a_hat": 0.7587, "total_queries": 256,
         "mc_stderr": 2.7e-2, "abs_err_qae": 1.8e-3,
         "label": "Laurdan n=1 (15x)"},
        {"n": 2, "side": 4, "a_true": 0.75, "a_hat": 0.7586, "total_queries": 256,
         "mc_stderr": 2.7e-2, "abs_err_qae": 8.6e-3,
         "label": "Laurdan n=2 (3x)"},
    ]
    rows = list(historical)
    for r in ext:
        if "a_true" not in r:
            continue
        rows.append({
            "n": r["n"], "side": r["side"], "a_true": r["a_true"],
            "a_hat": r["a_hat"], "total_queries": r["total_queries"],
            "mc_stderr": r["mc_stderr_same_budget"],
            "abs_err_qae": r["abs_err_qae"],
            "label": f"Laurdan n={r['n']} ({r['advantage_ratio']:.1f}x)",
        })

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)

    qrange = np.logspace(1, 5, 80)
    ax.plot(qrange, 1.0 / qrange, "r--", lw=1.4, label=r"QAE limit $\propto M^{-1}$")
    ax.plot(qrange, 1.0 / np.sqrt(qrange), "b--", lw=1.4,
            label=r"MC limit $\propto M^{-1/2}$")

    markers = ["o", "s", "^", "D", "P"]
    for i, r in enumerate(rows):
        m = markers[i % len(markers)]
        # QAE empirical point (red filled)
        ax.scatter(r["total_queries"], r["abs_err_qae"],
                   marker=m, s=70, color="red", edgecolor="k", zorder=5,
                   label=f"MLQAE {r['label']}")
        # MC stderr at same budget (blue ring)
        ax.scatter(r["total_queries"], r["mc_stderr"],
                   marker=m, s=70, facecolor="none", edgecolor="blue", lw=1.4,
                   zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"total queries $M$")
    ax.set_ylabel(r"$|\hat{a} - a|$  vs  classical MC stderr")
    ax.set_title("Empirical QAE quadratic advantage across n (Laurdan canonical)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, which="both", alpha=0.25)
    out = FIG_OUT / "fig12_qae_extension.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# fig13: HW NEQR-only ceiling (J2, J5, J6) — match% vs transpiled CX
# ---------------------------------------------------------------------------

def _load_hw_label(label: str) -> dict | None:
    for summary in HW_ROOT.glob("*/summary.json"):
        try:
            data = json.loads(summary.read_text())
        except Exception:
            continue
        for val in data.values():
            if val.get("label") == label:
                return val
    return None


def build_fig13() -> None:
    j2 = _load_hw_label("j2_neqr_only")
    j5 = _load_hw_label("j5_neqr_n2")
    j6 = _load_hw_label("j6_neqr_n3")
    if j2 is None:
        print("fig13: missing j2 summary; skipping")
        return

    points: list[dict] = []
    for j, n, label in ((j2, 1, "J2 n=1"), (j5, 2, "J5 n=2"), (j6, 3, "J6 n=3")):
        if j is None:
            continue
        n_pix = 1 << (2 * n)
        ma = j.get("match_i_a_4", j.get("match_i_a", 0))
        mb = j.get("match_i_b_4", j.get("match_i_b", 0))
        pct_a = 100.0 * ma / n_pix
        pct_b = 100.0 * mb / n_pix
        points.append({
            "label": label, "n": n,
            "N_2q": j.get("two_q_gate_count", float("nan")),
            "F_pred": np.exp(-j.get("two_q_gate_count", 0) * 4e-3),
            "match_a_pct": pct_a, "match_b_pct": pct_b,
            "backend": j.get("backend", "?"),
        })

    if not points:
        print("fig13: no HW points; skipping")
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)

    xs = np.array([p["N_2q"] for p in points])
    ya = np.array([p["match_a_pct"] for p in points])
    yb = np.array([p["match_b_pct"] for p in points])
    ax.plot(xs, ya, "o-", color="C0", label="I_a match (majority)", markersize=8)
    ax.plot(xs, yb, "s-", color="C2", label="I_b match (majority)", markersize=8)

    # gate-budget fidelity curve (relative scale on right axis)
    nb = np.linspace(50, max(xs.max(), 2200), 200)
    fb = np.exp(-nb * 4e-3) * 100
    ax2 = ax.twinx()
    ax2.plot(nb, fb, "k:", alpha=0.55, label=r"$\exp(-N_{2q}\epsilon) \times 100$")
    ax2.set_ylabel(r"predicted fidelity $F \times 100$", color="0.3")
    ax2.set_ylim(0, 110)

    # uniform-random floor over 2^q = 4 codomain per intensity register
    ax.axhline(25, color="0.5", ls="--", lw=0.8, label="uniform-random floor (25%)")

    for p in points:
        ax.annotate(f"  {p['label']}\n  {p['backend']}",
                    (p["N_2q"], p["match_a_pct"]),
                    fontsize=8, alpha=0.8,
                    xytext=(8, 4), textcoords="offset points")

    ax.set_xlabel(r"transpiled two-qubit gate count $N_{2q}$")
    ax.set_ylabel(r"per-pixel match vs classical (%)")
    ax.set_title("Dual-NEQR encoder on Heron r2 — gate-budget ceiling")
    ax.set_ylim(0, 110)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    out = FIG_OUT / "fig13_hw_neqr_ceiling.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    build_fig11()
    build_fig12()
    build_fig13()


if __name__ == "__main__":
    main()
