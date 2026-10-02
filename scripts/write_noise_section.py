"""Write §6.9 of the manuscript (the calibrated noise model and the projections) from
paper/data_autonomous/noise_calibration.json, so that the text and the numbers cannot diverge.
Re-running replaces the section in place.

Usage: .venv/bin/python scripts/write_noise_section.py
"""
from __future__ import annotations
import json, math
import numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
MS = REPO / "paper/autonomous_qimp_paper_v2.md"
D = json.load(open(REPO / "paper/data_autonomous/noise_calibration.json"))
HEAD = "### 6.9 A noise model calibrated on the device, and 256 pixels"
EIGHT = [("random4_n3", "random"), ("canonical_shared_n3", "Laurdan"), ("fura2_n3", "Fura-2"), ("balanced8_n3", "balanced")]
SMALL = [("fourvalue_n1", "$2 \\times 2$ four-value"), ("canonical_shared_n1", "$2 \\times 2$ Laurdan"),
         ("fourvalue_n2", "$4 \\times 4$ tiled"), ("random4_n2", "$4 \\times 4$ random"), ("canonical_shared_n2", "$4 \\times 4$ Laurdan")]
AC = {"c20_a_fourvalue_n1_kingston": "GP, $2 \\times 2$", "c20_a_canonical_shared_n2_kingston": "GP, Laurdan $4 \\times 4$",
      "c20_a_random4_n2_kingston": "GP, random $4 \\times 4$", "c20_a_canonical_shared_n3_kingston": "GP, Laurdan $8 \\times 8$",
      "c20_c_fourvalue_n1_kingston": "roGFP, $2 \\times 2$", "c20_c_canonical_shared_n2_kingston": "roGFP, Laurdan $4 \\times 4$",
      "c20_c_random4_n2_kingston": "roGFP, random $4 \\times 4$"}

def pc(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.0f} per cent"

def rng(a, b, nd=3):
    return f"{a:.{nd}f}" if abs(a - b) < 0.5 * 10 ** -nd else f"{a:.{nd}f} to {b:.{nd}f}"

def f(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"

def main() -> None:
    p = D["p_fit"]; hw = D["hardware_c17"]; fit = D["fit"]
    pk = min(fit, key=lambda k: abs(float(k) - p))          # grid point nearest the fitted p
    f0 = fit["0.0000"]
    rows = ["| Circuit | Role | Quantity | Device model | Calibrated model | Hardware |", "|---|---|---|---:|---:|---:|"]
    for k, name in EIGHT:
        rows.append(f"| Class B, $8 \\times 8$ {name} | fit | true-quotient probability | {f(f0[k]['true_share'], 3)} | {f(fit[pk][k]['true_share'], 3)} | {f(hw[k]['true_share'], 3)} |")
    val = D.get("validation", {})
    for k, name in SMALL:
        v = val.get("classB", {}).get(k)
        if v:
            rows.append(f"| Class B, {name} | held-out | true-quotient probability | | {f(v['sim']['true_share'], 3)} | {f(v['hw']['true_share'], 3)} |")
    for k, name in AC.items():
        v = val.get("classAC", {}).get(k)
        if v and not v.get("skipped"):
            rows.append(f"| {name} | held-out | exact value | | {f(v['sim']['exact'])} | {f(v['hw']['exact'])} |")
            if "sign_marginal" in v["sim"] and not math.isnan(v["sim"].get("sign_marginal", float('nan'))):
                rows.append(f"| {name} | held-out | sign, own qubit | | {f(v['sim']['sign_marginal'])} | {f(v['hw']['sign_marginal'])} |")
    for k in ("0", "1"):
        v = val.get("qae", {}).get(k)
        if v and not v.get("skipped"):
            rows.append(f"| oracle, $k = {k}$ | held-out | $P(\\mathrm{{good}})$, ideal {f(v['ideal'])} | | {f(v['sim'], 3)} | {f(v['hw'], 3)} |")
    proj = D.get("projection", {})
    PJ = [("B_blocks_n4", "Class B, $16 \\times 16$ block-constant, 11 operand pairs"), ("B_tiled_n4", "Class B, $16 \\times 16$ tiled, 4 operand pairs"),
          ("B_blocks_n3", "Class B, $8 \\times 8$ block-constant, 7 operand pairs"), ("A_gp_balanced_n2", "GP, $4 \\times 4$ balanced over four values"),
          ("A_gp_balanced_n3", "GP, $8 \\times 8$ balanced over four values")]
    c18 = D.get("c18_sim", {})
    C18 = [("blocks_n4", "Campaign 18, $16 \\times 16$ block-constant, 11 operand pairs"),
           ("tiled_n4", "Campaign 18, $16 \\times 16$ tiled, 4 operand pairs"),
           ("blocks_n3", "Campaign 18, $8 \\times 8$ block-constant, 7 operand pairs")]
    def c18_pooled(k):
        v = c18.get(k, {}); r = v.get("runs", [])
        if not r:
            return None
        return dict(two_q=v["two_q"], runs=len(r), ff=float(np.mean([x["flatfield"] for x in r])),
                    am=float(np.mean([x["argmax"] for x in r])), above=sum(x["flatfield"] > 0.25 for x in r),
                    joint=sum(x["joint"] for x in r), valid=sum(x["valid"] for x in r), jnull=sum(x["joint_const_null"] for x in r),
                    exq1=[x["excess_q1"] for x in r])
    for k, name in C18:
        v = c18_pooled(k)
        if v:
            rows.append(f"| {name} | projection, {v['two_q']} 2q, {v['runs']} runs | flat-field decode (null 0.25); joint triple (null) | | "
                        f"{f(v['ff'])}; {v['joint']}/{v['valid']} ({v['jnull']}) | not run |")
    for k, name in PJ:
        v = proj.get(k)
        if not v or (k.startswith("B") and c18_pooled(k[2:])):
            continue
        m = v["mean"]
        if k.startswith("B"):
            rows.append(f"| {name} | projection, {v['two_q']} 2q | flat-field decode (null 0.25) | | {f(m['flatfield'])} | not run |")
        else:
            rows.append(f"| {name} | projection, {v['two_q']} 2q | exact value (null 0.25); sign, own qubit | | {f(m['exact'])}; {f(m.get('sign_marginal'))} | not run |")
    table = "\n".join(rows)

    b_err = [abs(fit[pk][k]["true_share"] - hw[k]["true_share"]) for k, _ in EIGHT]
    small = [val["classB"][k] for k, _ in SMALL if k in val.get("classB", {})]
    small_bias = [s["sim"]["true_share"] - s["hw"]["true_share"] for s in small]
    small_ok = all(s["sim"]["flatfield"] == s["hw"]["flatfield"] for s in small)
    blk = proj.get("B_blocks_n4", {}); til = proj.get("B_tiled_n4", {}); br = proj.get("B_blocks_n3", {})
    g2 = proj.get("A_gp_balanced_n2", {}); g3 = proj.get("A_gp_balanced_n3", {})
    acs = val.get("classAC", {})
    gp_lau = acs.get("c20_a_canonical_shared_n2_kingston")
    gp_big = acs.get("c20_a_canonical_shared_n3_kingston")
    ro = acs.get("c20_c_canonical_shared_n2_kingston")
    qae0, qae1 = val.get("qae", {}).get("0"), val.get("qae", {}).get("1")
    lo_fit, hi_fit = 564, 573          # routed CZ of the four fitted circuits (Table 16)
    def depth_clause(n2):
        if n2 is None:
            return ""
        if n2 <= hi_fit:
            return f"{n2} two-qubit gates, no deeper than the 564 to 573 at which the model was fitted"
        if n2 <= 728:
            return (f"{n2} two-qubit gates, deeper than the 564 to 573 of the fit and inside the 576 to 728 at which "
                    f"the device decoded 4- and 16-pixel images with the long-division divider (Tables 13 and 14)")
        return f"{n2} two-qubit gates, beyond the depths at which the device has decoded, so the projection extrapolates"
    shots_note = ""
    if gp_big and gp_big.get("skipped"):
        shots_note = (f" The Class-A $8 \\times 8$ circuit of campaign 20 was routed onto {gp_big.get('active_qubits')} qubits, where a"
                      f" trajectory costs about a second, and is not simulated.")
        gp_big = None
    elif gp_big and gp_big.get("sim_shots", 0) < 16384:
        shots_note = (f" The Class-A $8 \\times 8$ circuit of campaign 20 was routed onto {gp_big.get('active_qubits')} qubits and is"
                      f" simulated at {gp_big['sim_shots']} shots, the rest at the shots of the device runs up to 16 384.")
    pulled = ""
    if gp_lau:
        pulled = (f" On the Laurdan patch at 16 pixels it predicts {pc(gp_lau['sim']['exact'])} of the GP values exact, where the"
                  f" device returns its constant-read-out null, {pc(gp_lau['hw']['exact'])}, and it reads the sign right from its own qubit in"
                  f" {pc(gp_lau['sim'].get('sign_marginal'))} of the informative pixel-runs against {pc(gp_lau['hw'].get('sign_marginal'))} on the device.")
        if gp_big:
            pulled += (f" At 64 pixels the same comparison gives {pc(gp_big['sim']['exact'])} against {pc(gp_big['hw']['exact'])} exact and"
                       f" {pc(gp_big['sim'].get('sign_marginal'))} against {pc(gp_big['hw'].get('sign_marginal'))} for the sign.")
    ro_s = (f" For the roGFP index on the same patch it predicts {pc(ro['sim']['exact'])} exact against {pc(ro['hw']['exact'])} measured." if ro else "")
    qae_s = ""
    if qae0 and qae1 and qae1.get("skipped") and not qae0.get("skipped"):
        qae_s = (f" For the amplitude-estimation oracle of §7.7 it predicts a good-branch probability of {f(qae0['sim'], 3)} at"
                 f" $k = 0$, against {f(qae0['hw'], 3)} on the device and {f(qae0['ideal'], 3)} ideal, so it misses the pull toward"
                 f" the all-zero state there too. The $k = 1$ circuit was routed onto {qae1.get('active_qubits')} qubits and is not simulated.")
    elif qae0 and qae1:
        qae_s = (f" For the amplitude-estimation oracle of §7.7 it predicts a good-branch probability of {f(qae0['sim'], 3)} at"
                 f" $k = 0$ and {f(qae1['sim'], 3)} at $k = 1$, against {f(qae0['hw'], 3)} and {f(qae1['hw'], 3)} on the device.")

    B = c18.get("bands"); cb, ct, c3 = c18_pooled("blocks_n4"), c18_pooled("tiled_n4"), c18_pooled("blocks_n3")
    if B and cb:
        verdict = {"pass": "would pass", "partial": "would be partial", "fail": "would fail"}
        c18_par = (f"""The 256-pixel projection is campaign 18 as pre-registered, on the model. It uses the
compilations fixed for the device on 25 September, checked against their fingerprints, the
protocol's 65 536 shots and three runs. The block-constant $16 \\times 16$ image carries eleven
distinct operand pairs on 64 independent blocks, with the four quotient values equally frequent,
and routes to {depth_clause(cb['two_q'])}. Over the three runs the flat-field decoder returns
{pc(cb['ff'])} of its pixels against a null of 25 per cent, {cb['above']} of 3 runs above it, so band B1
{verdict[B['B1']['verdict']]}. Without a decoder the true triple is the most frequent one in {cb['joint']} of
{cb['valid']} pixel-runs against a constant null of {cb['jnull']}, and the identity excess on the pixels with
$q \\ge 1$ is {'positive in every run' if all(x > 0 for x in cb['exq1']) else 'positive in ' + str(sum(x > 0 for x in cb['exq1'])) + ' of 3 runs'}, so band B2 {verdict[B['B2']['verdict']]}. The tiled control returns
{pc(ct['ff']) if ct else '—'} at {ct['two_q'] if ct else '—'} gates, and the block-constant $8 \\times 8$ bridge {pc(c3['ff']) if c3 else '—'} at {c3['two_q'] if c3 else '—'} gates, so band B3
{verdict[B['B3']['verdict']]}. """)
    else:
        c18_par = (f"""The 256-pixel projection is a Class-B circuit. The block-constant $16 \\times 16$ image carries
eleven distinct operand pairs on 64 independent blocks, with the four quotient values equally
frequent. It routes to {depth_clause(blk.get('two_q'))}. The flat-field decoder
returns {f(blk.get('mean', {}).get('flatfield'))} of its pixels against a null of 0.25, and the plain argmax {f(blk.get('mean', {}).get('argmax'))}. The tiled control,
four operand pairs at {til.get('two_q', '—')} gates, returns {f(til.get('mean', {}).get('flatfield'))}, and the block-constant $8 \\times 8$ image
{f(br.get('mean', {}).get('flatfield'))} at {br.get('two_q', '—')} gates. """)

    e2, e3 = g2.get("mean", {}).get("exact"), g3.get("mean", {}).get("exact")
    if e2 is not None and e3 is not None and e3 < 0.25:
        gp_par = (f"""On the Class-A target balanced over four GP values, two of each sign, the model returns
{pc(e2)} of the values exact at 16 pixels and {g2.get('two_q')} gates, and {pc(e3)} at 64 pixels and {g3.get('two_q')} gates,
below the null of 25 per cent. The decoder is not at fault, since the same circuits without noise
return every value. The model is an upper bound for this map, so at 64 pixels the GP is out of
reach of the device too, and at 16 pixels the device run remains the test. """)
    else:
        gp_par = (f"""On the Class-A target balanced over four GP values, two of each sign, the model returns
{pc(e2)} of the values exact at 16 pixels and {pc(e3)} at 64. By the result above that is an upper
bound, and the device run remains the test. """)
    import datetime as _dt
    cal = D.get("calibration", {})
    try:
        cal_date = _dt.datetime.fromisoformat(cal["last_update"]).strftime("%-d %B %Y")
    except Exception:
        cal_date = "the day of the simulations"
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    n_broken = words.get(len(cal.get("broken_couplers", [])), str(len(cal.get("broken_couplers", []))))
    n_dead = words.get(len(cal.get("dead_qubits", [])), str(len(cal.get("dead_qubits", []))))
    n_touch = words.get(sum(any(q in cal.get("dead_qubits", []) for q in c) for c in cal.get("broken_couplers", [])), "none")
    old = D.get("superseded_oct1", {})
    prev_fit = (f" The same fit on the calibration of the previous day gave $p = {old['p_fit']:.4f}$." if "p_fit" in old else "")
    chan = (f" at $p = {p:.4f}$" if f"{float(pk):.4f}" == f"{p:.4f}"
            else f", at the grid point $p = {float(pk):.4f}$ for the fit rows and at $p = {p:.4f}$ elsewhere")
    text = f"""{HEAD}

The allowance did not return before submission, so two measurements planned for the device ran
on a noise model calibrated on the device instead. Campaign 18, at 256 pixels, was pre-registered
for `ibm_kingston` (`paper/HW_CAMPAIGN_18_PROTOCOL.md`), and a Class-A target balanced in sign
is the device test that §6.8 leaves open. Both are projected here and are labelled as projections
wherever they appear. The model is `NoiseModel.from_backend` of `ibm_kingston` at the calibration of {cal_date},
with gate errors, relaxation during gates and read-out errors, frozen to a file so that every
stage below uses the same calibration. The {n_broken} couplers and the {n_dead} qubit reported broken that day,
{n_touch} of the couplers touching that qubit, are given the device's median values. As it stands the model is
optimistic. On the four 64-pixel images of campaign 17 it puts the probability of the true
quotient at {f(min(f0[k]['true_share'] for k, _ in EIGHT))} to {f(max(f0[k]['true_share'] for k, _ in EIGHT))}, where the device gives {f(min(hw[k]['true_share'] for k, _ in EIGHT))} to {f(max(hw[k]['true_share'] for k, _ in EIGHT))}. One
parameter closes the gap, a two-qubit depolarizing channel of strength $p$ composed with every CZ.
It stands for what the device model omits, relaxation and dephasing in idle windows, crosstalk and
coherent error. Fitted on those four images (`scripts/calibrate_noise_model.py`), $p = {p:.4f}$
reproduces all four within {f(max(b_err), 3)}.{prev_fit} Table 95 gives the fit, the comparisons on
data the fit did not see and the projections.

{table}

**Table 95**. The calibrated noise model. *Fit* rows fix the one free parameter, *held-out* rows
test it on data it was not fitted on, and *projection* rows are circuits that did not run on the
device. The device model is the backend noise model alone. The calibrated model adds the fitted
two-qubit channel{chan}. True-quotient probability is the mean over valid pixels of the probability of the
correct quotient. Hardware values are the means over the runs of Tables 16 and 17 and §7.7.

The held-out rows say where the model can be trusted. For the Class-B quotient it is
quantitative. At 4 and 16 pixels it overstates the probability of the true quotient by
{rng(min(small_bias), max(small_bias))}, and it {"returns the device's flat-field decode on every target" if small_ok else "does not return the device's flat-field decode on every target"}. For the Class-A
map it is not quantitative.{pulled}{ro_s}{qae_s} The device model relaxes qubits only during gates and the fitted
channel is symmetric, so neither produces the pull toward the relaxed state of §6.8 and §7.7. That is
consistent with relaxation in idle windows as its cause, and it makes the model an upper bound
for any circuit whose read-out that pull decides.{shots_note}

{c18_par}{gp_par}None of this is a hardware result. It gives campaign 18 a prediction to be tested
against when the allowance returns.

"""
    t = MS.read_text()
    if HEAD in t:
        i = t.index(HEAD); j = t.index("\n---\n\n## 7", i)
        t = t[:i] + text.rstrip() + "\n" + t[j:]
    else:
        j = t.index("---\n\n## 7 Amplitude-estimation oracle")
        t = t[:j] + text + t[j:]
    t = t.replace("<!-- PROJECTION_PARAGRAPH -->\n\n", "").replace("<!-- PROJECTION_PARAGRAPH -->\n", "")
    MS.write_text(t)
    print("section 6.9 written:", len(rows) - 2, "table rows")

if __name__ == "__main__":
    main()
