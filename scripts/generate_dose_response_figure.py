"""Fig. (dose-response): decoupling penalty against the inverted fraction f on ibm_kingston
(campaigns 14 and 15). Left: per-pixel separation and sigma over null vs f. Right: true-value
share of the divisor (load-only, 1-rich), quotient and remainder registers vs f.
Reads paper/data_autonomous/campaign11_spectators.json and arithmetic_identity.json."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parents[1]
S = json.load(open(REPO / "paper/data_autonomous/campaign11_spectators.json"))
A = json.load(open(REPO / "paper/data_autonomous/arithmetic_identity.json"))
ptsM = [(0.0, "ibm_marrakesh:dd_active_bunched", "c13_dd_active_bunched_marrakesh"), (0.25, "ibm_marrakesh:dd_active_f025", "c15_dd_active_f025_marrakesh"), (0.5, "ibm_marrakesh:dd_active_f050", "c15_dd_active_f050_marrakesh"), (0.75, "ibm_marrakesh:dd_active_f075", "c15_dd_active_f075_marrakesh"), (1.0, "ibm_marrakesh:dd_active_f100", "c15_dd_active_f100_marrakesh")]
symM = (0.5, "ibm_marrakesh:dd_active", "c11_dd_active_marrakesh"); refM = ("ibm_marrakesh:nodd", "c11_nodd_marrakesh")
sepM = np.array([S[p[1]]["separation"] for p in ptsM]); sigM = np.array([S[p[1]]["sigma_over_null"] for p in ptsM])
pts = [(0.0, "ibm_kingston:dd_active_bunched", "c14_dd_active_bunched_kingston"),
       (0.25, "ibm_kingston:dd_active_f025", "c15_dd_active_f025_kingston"),
       (0.5, "ibm_kingston:dd_active_f050", "c15_dd_active_f050_kingston"),
       (0.75, "ibm_kingston:dd_active_f075", "c15_dd_active_f075_kingston"),
       (1.0, "ibm_kingston:dd_active_f100", "c15_dd_active_f100_kingston")]
sym = (0.5, "ibm_kingston:dd_active", "c14_dd_active_kingston")
ref = ("ibm_kingston:nodd", "c14_nodd_kingston")
f = np.array([p[0] for p in pts]); sep = np.array([S[p[1]]["separation"] for p in pts]); sig = np.array([S[p[1]]["sigma_over_null"] for p in pts])
slope, icpt = np.polyfit(f, sep, 1); r2 = 1 - ((sep - (slope * f + icpt)) ** 2).sum() / ((sep - sep.mean()) ** 2).sum()
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
sl2, ic2 = np.polyfit(f, sig, 1); r2s = 1 - ((sig - (sl2 * f + ic2)) ** 2).sum() / ((sig - sig.mean()) ** 2).sum()
ax0 = ax[0]; ax0.axhline(S[ref[0]]["sigma_over_null"], color="0.4", ls="--", lw=1, label="no decoupling"); ax0.plot(f, sig, "o-", color="C0", label="ibm_kingston, first pulse at window start"); ax0.plot([sym[0]], [S[sym[1]]["sigma_over_null"]], "s", color="C1", label="XX symmetric (campaigns 11, 14)"); ax0.plot(f, sl2 * f + ic2, ":", color="C0", lw=1, label=f"kingston linear fit, $R^2$ = {r2s:.2f}"); ax0.plot(f, sigM, "o-", color="C4", label="ibm_marrakesh"); ax0.plot([symM[0]], [S[symM[1]]["sigma_over_null"]], "s", color="C4", mfc="none"); ax0.axhline(S[refM[0]]["sigma_over_null"], color="C4", ls="--", lw=0.8, alpha=0.6); ax0.set_xlabel("inverted fraction $f$ of each idle window"); ax0.set_ylabel("margin over uniform null ($\\sigma$)"); ax0.legend(fontsize=7); ax0.set_title("modal margin (dashed: no decoupling)", fontsize=10)
ax = ax[1:]
ax[0].axhline(S[ref[0]]["separation"], color="0.4", ls="--", lw=1, label="no decoupling")
ax[0].plot(f, sep, "o-", color="C0", label="ibm_kingston, first pulse at window start")
slM, icM = np.polyfit(f, sepM, 1); r2M = 1 - ((sepM - (slM * f + icM)) ** 2).sum() / ((sepM - sepM.mean()) ** 2).sum()
ax[0].plot(f, sepM, "o-", color="C4", label="ibm_marrakesh"); ax[0].plot([symM[0]], [S[symM[1]]["separation"]], "s", color="C4", mfc="none"); ax[0].axhline(S[refM[0]]["separation"], color="C4", ls="--", lw=0.8, alpha=0.6); ax[0].plot(f, slM * f + icM, ":", color="C4", lw=1, label=f"marrakesh linear fit, $R^2$ = {r2M:.2f}")
ax[0].plot([sym[0]], [S[sym[1]]["separation"]], "s", color="C1", label="XX symmetric (campaigns 11, 14)")
ax[0].plot(f, slope * f + icpt, ":", color="C0", lw=1, label=f"kingston linear fit, $R^2$ = {r2:.2f}")
ax[0].set_xlabel("inverted fraction $f$ of each idle window"); ax[0].set_ylabel("per-pixel separation $d_1 - d_0$"); ax[0].legend(fontsize=7); ax[0].set_title("quotient signal (dashed: no decoupling)", fontsize=10)
for key, lab, col in (("d_share", "divisor $d$ (preserved, 1-rich)", "C2"), ("q_share", "quotient $q$", "C0"), ("r_share", "remainder $r$", "C3")):
    y = [A[p[2]][key] for p in pts]; ax[1].plot(f, y, "o-", color=col, label=lab)
    yM = [A[p[2]][key] for p in ptsM]; ax[1].plot(f, yM, "o--", color=col, mfc="none", alpha=0.7)
    ax[1].axhline(A[ref[1]][key], color=col, ls="--", lw=0.8, alpha=0.6)
ax[1].set_xlabel("inverted fraction $f$"); ax[1].set_ylabel("mean true-value share of the register"); ax[1].legend(fontsize=7); ax[1].set_title("per-register read-out, kingston filled, marrakesh open", fontsize=10)
fig.tight_layout(); out = REPO / "paper/figures_autonomous/fig_dose_response.png"; fig.savefig(out, dpi=200); print("wrote", out, f"kingston sep R2 {r2:.3f} sigma R2 {r2s:.3f}; marrakesh sep slope {slM:+.3f} R2 {r2M:.3f}")
