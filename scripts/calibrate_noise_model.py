"""A noise model calibrated on the hardware runs, for the projections the allowance did not cover.

The device model (`NoiseModel.from_backend(ibm_kingston)`, gate errors, thermal relaxation
during gates, read-out errors) is optimistic for these circuits: it predicts every pixel of the
64-pixel images, where the device returns 78 to 88 per cent. It omits idle decoherence,
crosstalk and coherent error. One parameter absorbs them: a two-qubit depolarizing channel of
strength p composed with every CZ of the device model.

Stage `fit`: p is chosen on the four 8x8 Class-B targets of campaign 17, by matching the mean
probability of the true quotient over the valid pixels, simulating the archived transpiled
circuits that ran on the device.
Stage `validate`: at the fitted p the model predicts data it was not fitted on: the 2x2 and
4x4 targets of campaign 17, the Class-A and Class-C circuits of campaign 20 and the amplitude-
estimation circuits of campaign 19.
Stage `project`: at the fitted p, the circuits the allowance did not reach.
Stage `c18`: campaign 18 as pre-registered (fixed compilations, protocol shots, three runs, bands).

Usage:
  .venv/bin/python scripts/calibrate_noise_model.py fit [--grid 0,0.004,0.008,0.012,0.016,0.024]
  .venv/bin/python scripts/calibrate_noise_model.py validate
  .venv/bin/python scripts/calibrate_noise_model.py project
"""
from __future__ import annotations
import argparse, collections, glob, json, sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from qiskit import qpy
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qimp.runtime import ibm
from analyse_hw_signal import analyse
from run_hardware_class_b_nonrestoring import load_dataset

OUT = REPO / "paper/data_autonomous/noise_calibration.json"
C17_STAMP = "20260925T145232Z"
EIGHT = ["random4_n3", "canonical_shared_n3", "fura2_n3", "balanced8_n3"]
SMALL = ["fourvalue_n1", "canonical_shared_n1", "fourvalue_n2", "random4_n2", "canonical_shared_n2"]
SIM_SHOTS = {1: 4096, 2: 8192, 3: 16384, 4: 32768}

_BE = None
def backend():
    global _BE
    if _BE is None:
        _BE = ibm.get_service().backend("ibm_kingston")
    return _BE

SNAP = REPO / "data/output/noise_sim/kingston_noise_model_snapshot.pkl"

def calibration() -> str:
    """Date of the device calibration the model is built from (the snapshot's, once one exists)."""
    import pickle
    if SNAP.exists():
        return pickle.load(open(SNAP, "rb"))["last_update"]
    return str(backend().properties().last_update_date)

def model(p_extra: float) -> AerSimulator:
    """The frozen device model (see `_device_model`) with a depolarizing channel of strength p
    composed with every CZ. The first call saves the device model to SNAP, and every later call,
    in any stage, loads it from there, so that the fit, the validation and the projections share
    one calibration even if the provider recalibrates the device in between."""
    import copy, pickle
    if SNAP.exists():
        nm = pickle.load(open(SNAP, "rb"))["noise_model"]
    else:
        nm = _device_model(); SNAP.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump({"noise_model": nm, "last_update": str(backend().properties().last_update_date)}, open(SNAP, "wb"))
    nm = copy.deepcopy(nm)
    local = nm._local_quantum_errors.get("cz", {})
    if p_extra > 0:
        dep = depolarizing_error(p_extra, 2)
        for qubits, err in list(local.items()):
            local[qubits] = err.compose(dep)
    sim = AerSimulator.from_backend(backend())
    sim.set_options(noise_model=nm)
    return sim

def _device_model() -> NoiseModel:
    """The device noise model at the current calibration, with broken elements given median values.

    Couplers that today's calibration reports as broken (error >= 0.5) were healthy when the
    archived circuits ran on 25 to 27 September, and a channel at error 1.0 is not simulable.
    Their error is replaced by that of the median coupler of the device, which is stated in the
    manuscript; no archived circuit is re-routed."""
    nm = NoiseModel.from_backend(backend())
    cz_props = backend().target["cz"]
    healthy = sorted((props.error, q) for q, props in cz_props.items() if props is not None and props.error is not None and props.error < 0.5)
    median_pair = healthy[len(healthy) // 2][1]
    local = nm._local_quantum_errors.get("cz", {})
    for q, props in cz_props.items():
        if props is not None and (props.error is None or props.error >= 0.5) and median_pair in local:
            local[q] = local[median_pair]
    # qubits reported dead today (no T1/T2, or read-out error >= 0.5): give them the errors of the
    # median qubit, for every single-qubit instruction and for the read-out
    tgt = backend().target
    def dead(q):
        qp = tgt.qubit_properties[q] if tgt.qubit_properties else None
        mp = tgt["measure"].get((q,)) if "measure" in tgt else None
        return (qp is None or not qp.t1 or not qp.t2) or (mp is not None and (mp.error or 0) >= 0.5)
    alive = [q for q in range(backend().num_qubits) if not dead(q)]
    t1s = sorted((tgt.qubit_properties[q].t1, q) for q in alive)
    q_med = t1s[len(t1s) // 2][1]
    # couplers touching a dead qubit carry its missing T1/T2 in their relaxation term: give them
    # the median coupler too
    for qubits in list(local):
        if any(dead(q) for q in qubits):
            local[qubits] = local[median_pair]
    for instr, per in nm._local_quantum_errors.items():
        if instr == "cz" or (q_med,) not in per:
            continue
        for q in range(backend().num_qubits):
            if dead(q) and (q,) in per:
                per[(q,)] = per[(q_med,)]
    for q in range(backend().num_qubits):
        if dead(q) and (q,) in nm._local_readout_errors and (q_med,) in nm._local_readout_errors:
            nm._local_readout_errors[(q,)] = nm._local_readout_errors[(q_med,)]
    return nm

def load_circuit(stamp: str, sub: str, key: str):
    with open(REPO / "data/output/ibm_hw" / stamp / sub / f"{key}_ibm_kingston.qpy", "rb") as f:
        return qpy.load(f)[0]

def classb_metrics(counts: dict, dataset: str, n: int) -> dict:
    """Mean probability of the true quotient over valid pixels, flat-field and argmax accuracy."""
    Ia, Ib = load_dataset(dataset, n, 2)
    r = analyse(counts, (Ia, Ib), "lookup", n, 2)
    P = np.array([px["histogram"] for px in r["pixels"]]); truth = [px["true"] for px in r["pixels"]]
    valid = [i for i, x in enumerate(truth) if x is not None]
    ff = (P - P.mean(axis=0, keepdims=True)).argmax(axis=1); am = P.argmax(axis=1)
    return dict(true_share=float(np.mean([P[i, truth[i]] for i in valid])),
                flatfield=float(np.mean([ff[i] == truth[i] for i in valid])),
                argmax=float(np.mean([am[i] == truth[i] for i in valid])))

def hardware_classb() -> dict:
    out = {}
    for key in EIGHT + SMALL:
        ds, n = key.rsplit("_n", 1); n = int(n)
        ms = [classb_metrics(json.load(open(Path(d) / "counts.json")), ds, n)
              for d in sorted(glob.glob(str(REPO / f"data/output/ibm_hw/*/runs/c17_{key}_lookup_kingston_r*_hw")))]
        out[key] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]} | {"runs": len(ms)}
    return out

def simulate_classb(sim, key: str, seed: int = 11) -> dict:
    ds, n = key.rsplit("_n", 1); n = int(n)
    circ = load_circuit(C17_STAMP, "transpiled_c17", key)
    counts = sim.run(circ, shots=SIM_SHOTS[n], seed_simulator=seed).result().get_counts()
    return classb_metrics(counts, ds, n)

def stage_fit(grid: list[float]) -> None:
    hw = hardware_classb()
    res = json.load(open(OUT)) if OUT.exists() else {}
    res["hardware_c17"] = hw; res.setdefault("calibration", {})["last_update"] = calibration()
    res.setdefault("fit", {})
    for p in grid:
        if f"{p:.4f}" in res["fit"]:
            continue
        t0 = time.time(); sim = model(p)
        res["fit"][f"{p:.4f}"] = {k: simulate_classb(sim, k) for k in EIGHT}
        json.dump(res, open(OUT, "w"), indent=2)
        line = "  ".join(f"{k.split('_n')[0][:8]} {res['fit'][f'{p:.4f}'][k]['true_share']:.3f}" for k in EIGHT)
        print(f"p = {p:.4f}  {line}   ({time.time() - t0:.0f} s)", flush=True)
    target = np.array([hw[k]["true_share"] for k in EIGHT])
    ps = sorted(float(p) for p in res["fit"])
    err = [float(np.sum((np.array([res["fit"][f'{p:.4f}'][k]["true_share"] for k in EIGHT]) - target) ** 2)) for p in ps]
    # quadratic refinement around the grid minimum
    i = int(np.argmin(err))
    if 0 < i < len(ps) - 1:
        a, b, c = np.polyfit(ps[i - 1:i + 2], err[i - 1:i + 2], 2); best = float(-b / (2 * a)) if a > 0 else ps[i]
    else:
        best = ps[i]
    res["p_fit"] = best; res["fit_error"] = dict(zip([f"{p:.4f}" for p in ps], err))
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"hardware true-quotient share at 8x8: " + "  ".join(f"{k.split('_n')[0][:8]} {hw[k]['true_share']:.3f}" for k in EIGHT))
    print(f"fitted p = {best:.4f}")

# ---------------------------------------------------------------------------------------------
# validation on data the fit did not see
# ---------------------------------------------------------------------------------------------
def _find_archive(sub: str, key: str, sha: str) -> Path | None:
    for d in sorted(glob.glob(str(REPO / "data/output/ibm_hw/*" / sub))):
        man = Path(d) / "manifest_ibm_kingston.json"
        if man.exists() and json.load(open(man)).get(key, {}).get("sha256") == sha:
            return Path(d)
    return None

def classa_metrics(counts: dict, cls: str, dataset: str, n: int) -> dict:
    """Exact-value accuracy by majority vote (the pre-registered decoder) and, for Class A, the
    sign read from its own qubit's marginal on the pixels whose true GP is non-zero."""
    from run_campaign_20 import build, classical, Q, Q_FRAC, R_OX
    from qimp.processing.ratiometric_circuit import decode_class_a_full, decode_class_c_rogfp
    Ia, Ib = load_dataset(dataset, n, Q); _, lay = build(cls, Ia, Ib); ref, dz = classical(cls, Ia, Ib)
    width = max(len(k) for k in counts)
    img, _ = (decode_class_a_full(counts, n, Q, Q_FRAC, lay, width) if cls == "a"
              else decode_class_c_rogfp(counts, n, Q, Q_FRAC, 0.0, R_OX, lay, width))
    valid = ~dz
    out = dict(exact=float(np.mean(img[valid] == ref[valid])))
    if cls == "a":
        acc = collections.defaultdict(lambda: np.zeros(2))
        for s, w in counts.items():
            bit = lambda i: int(s[width - 1 - i])
            col = sum(bit(lay["position"][i]) << i for i in range(n)); row = sum(bit(lay["position"][n + i]) << i for i in range(n))
            acc[(row, col)] += (w, w * bit(lay["sign"]))
        ok = tot = 0
        for (r, c), (w, s1) in acc.items():
            if dz[r, c] or ref[r, c] == 0:
                continue
            tot += 1; ok += int((s1 / w > 0.5) == bool(ref[r, c] < 0))
        out["sign_marginal"] = ok / tot if tot else float("nan")
    return out

BIG_ACTIVE, BIG_SHOTS = 18, 2048   # trajectory budget: circuits on more active qubits get fewer shots

def _active(circ) -> int:
    return len({circ.find_bit(q).index for ins in circ.data for q in ins.qubits if ins.operation.name != "barrier"})

def stage_validate() -> None:
    res = json.load(open(OUT)); p = res["p_fit"]; sim = model(p); hw = res["hardware_c17"]
    val = res.get("validation") or {"classB": {}, "classAC": {}, "qae": {}}
    save = lambda: (res.__setitem__("validation", val), json.dump(res, open(OUT, "w"), indent=2))
    print(f"validation at p = {p:.4f}\n-- Class B, campaign 17 targets not used in the fit")
    for key in SMALL:
        if key in val["classB"]:
            continue
        s = simulate_classb(sim, key)
        val["classB"][key] = dict(sim=s, hw=hw[key]); save()
        print(f"   {key:22s} true-quotient share  sim {s['true_share']:.3f}  hw {hw[key]['true_share']:.3f}   flat-field sim {s['flatfield']:.2f} hw {hw[key]['flatfield']:.2f}", flush=True)
    print("-- Class A and C, campaign 20 (pre-registered decoder: exact value by majority vote)")
    runs = collections.defaultdict(list)
    for d in sorted(glob.glob(str(REPO / "data/output/ibm_hw/*/runs/c20_*_hw"))):
        m = json.load(open(Path(d) / "metadata.json")); runs[m["label"].rsplit("_r", 1)[0]].append((m, json.load(open(Path(d) / "counts.json"))))
    for cfg, items in sorted(runs.items()):
        m0 = items[0][0]; key = cfg.replace("c20_", "").replace("_kingston", "")
        if cfg in val["classAC"]:
            continue
        arch = _find_archive("transpiled_c20", key, m0["transpiled_sha256"])
        if arch is None:
            print(f"   {cfg}: archived circuit not found"); continue
        circ = load_circuit(arch.parent.name, "transpiled_c20", key)
        hwm = [classa_metrics(c, m0["operator_class"], m0["dataset"], m0["n"]) for _, c in items]
        hwx = {k: float(np.nanmean([h.get(k, np.nan) for h in hwm])) for k in hwm[0]}
        nact = _active(circ)
        if nact > BIG_ACTIVE:
            # about one second per trajectory at 21 active qubits: not simulated, and stated as such
            val["classAC"][cfg] = dict(skipped=True, active_qubits=nact, hw=hwx, runs=len(items)); save()
            print(f"   {key:26s} not simulated ({nact} active qubits)", flush=True)
            continue
        shots = min(m0["shots"], 16384); method = "automatic"
        try:
            cnt = sim.run(circ, shots=shots, seed_simulator=7).result().get_counts()
        except Exception as exc:
            # Aer's trajectory sampler can stop on a numerically empty Kraus set; the density matrix
            # is exact and samples no trajectories
            if "Kraus is empty" not in str(exc):
                raise
            method = "density_matrix"
            cnt = sim.run(circ, shots=shots, seed_simulator=7, method="density_matrix").result().get_counts()
        sm = classa_metrics(cnt, m0["operator_class"], m0["dataset"], m0["n"])
        val["classAC"][cfg] = dict(sim=sm, hw=hwx, runs=len(items), sim_shots=shots, active_qubits=nact, method=method); save()
        extra = f"   sign (marginal) sim {sm['sign_marginal']:.2f} hw {hwx['sign_marginal']:.2f}" if "sign_marginal" in sm else ""
        print(f"   {key:26s} exact sim {sm['exact']:.2f} hw {hwx['exact']:.2f}{extra}   ({nact} active qubits, {shots} shots)", flush=True)
    print("-- amplitude estimation, campaign 19")
    for k in (0, 1):
        if str(k) in val["qae"]:
            continue
        ms = [json.load(open(Path(d) / "metadata.json")) for d in sorted(glob.glob(str(REPO / f"data/output/ibm_hw/*/runs/c19_k{k}_kingston_r*_hw")))]
        arch = _find_archive("transpiled_c19", str(k), ms[0]["transpiled_sha256"])
        circ = None
        for d in sorted(glob.glob(str(REPO / "data/output/ibm_hw/*/transpiled_c19"))):
            f = Path(d) / f"k{k}_ibm_kingston.qpy"
            if f.exists():
                import hashlib
                if hashlib.sha256(f.read_bytes()).hexdigest() == ms[0]["transpiled_sha256"]:
                    with open(f, "rb") as fh: circ = qpy.load(fh)[0]
                    break
        if circ is None:
            print(f"   k={k}: archived circuit not found"); continue
        if _active(circ) > BIG_ACTIVE:
            val["qae"][str(k)] = dict(skipped=True, active_qubits=_active(circ), hw=float(np.mean([m["p_measured"] for m in ms])),
                                      ideal=ms[0]["p_ideal"]); save()
            print(f"   k={k}: not simulated ({_active(circ)} active qubits)", flush=True); continue
        cnt = sim.run(circ, shots=4096, seed_simulator=7).result().get_counts()
        ps = cnt.get("1", 0) / 4096; ph = float(np.mean([m["p_measured"] for m in ms]))
        val["qae"][str(k)] = dict(sim=ps, hw=ph, ideal=ms[0]["p_ideal"]); save()
        print(f"   k={k}: P(good) sim {ps:.3f}  hw {ph:.3f}  ideal {ms[0]['p_ideal']:.3f}", flush=True)
    res["validation"] = val; json.dump(res, open(OUT, "w"), indent=2)

# ---------------------------------------------------------------------------------------------
# projection of what the allowance did not reach
# ---------------------------------------------------------------------------------------------
def stage_project(runs: int = 1) -> None:
    from qiskit import transpile
    from qimp.processing.ratiometric_circuit import class_b_ratio
    from run_campaign_20 import build
    res = json.load(open(OUT)); p = res["p_fit"]; sim = model(p); tgt = backend().target
    proj = res.get("projection", {})
    # shots per pixel: 128 at 16x16, 256 at 8x8 and 1024 at 4x4, enough for a per-pixel decode
    # the Class-B projections are campaign 18 as pre-registered (stage c18)
    jobs = [("A", "gp_balanced", 2, 16384), ("A", "gp_balanced", 3, 16384)]
    print(f"projection at p = {p:.4f}")
    for cls, ds, n, shots in jobs:
        key = f"{cls}_{ds}_n{n}"
        if key in proj:
            continue
        Ia, Ib = load_dataset(ds, n, 2)
        qc, _ = (class_b_ratio(Ia, Ib, q=2, divider="lookup", load="ucry") if cls == "B" else build("a", Ia, Ib))
        qc.measure_all()
        best = min(((transpile(qc, target=tgt, optimization_level=3, seed_transpiler=s), s) for s in range(8)),
                   key=lambda x: sum(x[0].count_ops().get(g, 0) for g in ("cz", "cx", "ecr")))
        n2 = sum(best[0].count_ops().get(g, 0) for g in ("cz", "cx", "ecr"))
        out = []
        # 15 active qubits: the density matrix in single precision (8 GB) beats trajectories
        kw = dict(method="density_matrix", precision="single") if _active(best[0]) == 15 else {}
        for r in range(runs):
            cnt = sim.run(best[0], shots=shots, seed_simulator=100 + r, **kw).result().get_counts()
            out.append(classb_metrics(cnt, ds, n) if cls == "B" else classa_metrics(cnt, "a", ds, n))
        mean = {k: float(np.nanmean([o[k] for o in out])) for k in out[0]}
        proj[key] = dict(two_q=n2, seed=best[1], shots=shots, runs=runs, per_run=out, mean=mean)
        res["projection"] = proj; json.dump(res, open(OUT, "w"), indent=2)
        print(f"   {key:22s} {n2:4d} two-qubit gates  " + "  ".join(f"{k} {v:.3f}" for k, v in mean.items()), flush=True)

# ---------------------------------------------------------------------------------------------
# campaign 18 as pre-registered, on the calibrated model
# ---------------------------------------------------------------------------------------------
C18_DIR = REPO / "data/output/ibm_hw/20260925T153234Z/transpiled_c18"
SIM_OUT = REPO / "data/output/noise_sim"

def stage_c18(runs: int = 3) -> None:
    """The compilations fixed for campaign 18 on 25 September (sha256-checked against its manifest),
    the protocol's shots and three runs, scored with the protocol's bands B1 to B3. Each simulated
    run is archived like a hardware run, marked simulated, so the hardware scorers read it."""
    import hashlib, io
    from arithmetic_identity import analyse_run
    res = json.load(open(OUT)); p = res["p_fit"]; sim = model(p)
    man = json.load(open(C18_DIR / "manifest_ibm_kingston.json"))
    c18 = res.get("c18_sim", {})
    print(f"campaign 18 on the calibrated model, p = {p:.4f}", flush=True)
    for key in ("blocks_n3", "tiled_n4", "blocks_n4"):
        e = man[key]; raw = (C18_DIR / f"{key}_ibm_kingston.qpy").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == e["sha256"], f"{key}: archived circuit does not match its manifest"
        circ = qpy.load(io.BytesIO(raw))[0]
        per = c18.get(key, {}).get("runs", [])
        todo = runs - len(per)
        if todo <= 0:
            continue
        # One density-matrix simulation in single precision (8 GB at 15 qubits) gives the final state
        # once, and Aer samples every shot independently from it, read-out errors included. The
        # remaining runs are consecutive blocks of that one job's per-shot memory, so each block is a
        # set of independent shots from the model, as a separate run would be. (Trajectories cost
        # about 0.8 s per shot on this circuit.)
        t0 = time.time(); seed = 1800 + len(per)
        mem = sim.run(circ, shots=e["shots"] * todo, seed_simulator=seed, method="density_matrix",
                      precision="single", memory=True).result().get_memory()
        block_seconds = round((time.time() - t0) / todo)
        for r in range(len(per), runs):
            j = r - (runs - todo)
            cnt = dict(collections.Counter(mem[j * e["shots"]:(j + 1) * e["shots"]]))
            label = f"c18sim_{key}_kingston_r{r + 1}"
            d = SIM_OUT / "c18" / f"{label}_sim"; d.mkdir(parents=True, exist_ok=True)
            meta = dict(label=label, backend=f"ibm_kingston noise model (calibration of {calibration()}) + fitted CZ depolarizing",
                        simulated=True, p_extra=p, dataset=e["ds"], n=e["n"], q=2, divider="lookup", load="ucry",
                        shots=e["shots"], two_q_gate_count=e["two_q"], transpiled_sha256=e["sha256"], seed_simulator=seed,
                        method=f"density_matrix (single precision), block {j + 1} of {todo} of one {e['shots'] * todo}-shot job")
            json.dump(meta, open(d / "metadata.json", "w"), indent=2); json.dump(cnt, open(d / "counts.json", "w"))
            m = classb_metrics(cnt, e["ds"], e["n"])
            _, px, (const, _) = analyse_run(d)
            q1 = [x for x in px if x["q_ge1"]]
            per.append(dict(m, valid=len(px), joint=int(sum(x["joint_argmax_true"] for x in px)), joint_const_null=int(const),
                            excess_q1=float(np.nanmean([x["identity_rate"] - x["identity_cross"] for x in q1])) if q1 else float("nan"),
                            seconds=block_seconds, method=f"density_matrix (single precision), block {j + 1} of {todo} of one job"))
            c18[key] = dict(two_q=e["two_q"], shots=e["shots"], runs=per); res["c18_sim"] = c18
            json.dump(res, open(OUT, "w"), indent=2)
            x = per[-1]
            print(f"   {label:30s} flat-field {x['flatfield']:.3f}  argmax {x['argmax']:.3f}  true share {x['true_share']:.3f}  "
                  f"joint {x['joint']}/{x['valid']} (null {x['joint_const_null']})  identity excess q>=1 {x['excess_q1']:+.4f}  ({x['seconds']} s)", flush=True)
    # the protocol's bands
    bl = c18["blocks_n4"]["runs"]; br = c18["blocks_n3"]["runs"]
    pooled = float(np.mean([x["flatfield"] for x in bl])); above = sum(x["flatfield"] > 0.25 for x in bl)
    b1 = "pass" if pooled >= 0.75 and above >= 2 else ("partial" if pooled >= 0.40 else "fail")
    b2n = sum(x["joint"] > x["joint_const_null"] and x["excess_q1"] > 0 for x in bl); b2 = "pass" if b2n >= 2 else "fail"
    b3p = float(np.mean([x["flatfield"] for x in br])); b3 = "pass" if b3p >= 0.85 else "fail"
    c18["bands"] = dict(B1=dict(pooled_flatfield=pooled, runs_above_null=above, verdict=b1),
                        B2=dict(runs_passing=b2n, verdict=b2), B3=dict(pooled_flatfield=b3p, verdict=b3))
    res["c18_sim"] = c18; json.dump(res, open(OUT, "w"), indent=2)
    print(f"bands on the model: B1 {b1} (pooled {pooled:.3f}, {above}/3 above null)  B2 {b2} ({b2n}/3)  B3 {b3} ({b3p:.3f})", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["fit", "validate", "project", "c18"])
    ap.add_argument("--grid", default="0,0.004,0.008,0.012,0.016,0.024")
    a = ap.parse_args()
    if a.stage == "fit":
        stage_fit([float(x) for x in a.grid.split(",")])
    elif a.stage == "validate":
        stage_validate()
    elif a.stage == "project":
        stage_project()
    else:
        stage_c18()
