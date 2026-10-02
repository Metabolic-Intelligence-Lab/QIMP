"""Campaign 11: spectator-qubit test of the dynamical-decoupling penalty
(paper/HW_CAMPAIGN_11_PROTOCOL.md).

Runs the balanced 2x2 Class-B circuit (non-restoring, uniformly controlled
load) on ibm_marrakesh with readout twirling and NO runtime decoupling, and
pads XX dynamical decoupling ourselves (the target has no native Y; campaign 6 found XX and XY4 indistinguishable in effect) with a qubit-selective transpiler pass,
so that four conditions can be compared on one fixed layout:

  nodd        no decoupling anywhere
  dd_active   XY4 on the active (circuit) qubits only
  dd_spect    XY4 on idle spectator qubits only (neighbours of the active set)
  dd_all      XY4 on active and spectator qubits

Spectator qubits are physical neighbours of the active set that the circuit
never touches; they are measured at the end into a separate register. Their
excitation P(1) is the crosstalk observable; the active read-out is analysed
with the usual null-referenced metric.

Usage:
  .venv/bin/python scripts/run_spectator_experiment.py --dry-run
  .venv/bin/python scripts/run_spectator_experiment.py --condition nodd --repeat 4
"""
from __future__ import annotations
import argparse, datetime as dt, json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from qiskit import transpile, ClassicalRegister
from qiskit.circuit.library import XGate, YGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling
from qimp.processing.ratiometric_circuit import class_b_ratio
from qimp.runtime import ibm
from run_hardware_class_b_nonrestoring import load_dataset

OUT = REPO / "data" / "output" / "ibm_hw"
CONDITIONS = ("nodd", "dd_active", "dd_spect", "dd_all", "dd_spect_matched", "dd_active_bunched", "dd_spect_matched_bunched")
# campaign 15 (dose-response): XX on the active qubits, first pulse at the start of each idle
# window, second at fraction f of the window, so the qubit is inverted for a fraction f.
DOSE = {"dd_active_f025": 0.25, "dd_active_f050": 0.50, "dd_active_f075": 0.75, "dd_active_f100": 1.00}
CONDITIONS = CONDITIONS + tuple(DOSE)


def build(backend_name: str, n_spect: int = 6, seed: int = 0):
    svc = ibm.get_service(); be = svc.backend(backend_name); tgt = be.target
    I_a, I_b = load_dataset("canonical_shared", 1, 2)
    qc, layout = class_b_ratio(I_a, I_b, q=2, divider="nonrestoring", load="ucry")
    qc.measure_all()
    tqc = transpile(qc, target=tgt, optimization_level=3, seed_transpiler=seed)
    active = sorted(set(tqc.layout.final_index_layout()))
    cmap = tgt.build_coupling_map()
    neigh = {}
    for a in active:
        for b in cmap.neighbors(a):
            if b not in active:
                neigh[b] = neigh.get(b, 0) + 1
    spect = [q for q, _ in sorted(neigh.items(), key=lambda kv: (-kv[1], kv[0]))][:n_spect]
    spec_reg = ClassicalRegister(len(spect), "spec")
    tqc.add_register(spec_reg)
    for i, q in enumerate(spect):
        tqc.measure(q, spec_reg[i])
    return be, tgt, tqc, active, spect, (I_a, I_b)


def pad_spectators_matched(tqc, tgt, spect, pairs=12, bunched=False):
    """Campaign 12: put `pairs` XX pairs on each spectator, spread over the whole
    circuit duration with explicit delays, so that each spectator receives about
    as many pulses as an active qubit does under dd_active (562/24 ~ 23)."""
    from qiskit.circuit import Delay
    c = tqc.copy()
    dur = int(round(c.estimate_duration(tgt, unit="dt")))
    x_dur = int(round(tgt["x"][(spect[0],)].duration / tgt.dt)) if tgt["x"][(spect[0],)].duration else 0
    slot = max((dur - 2 * pairs * x_dur) // (2 * pairs), 1)
    # remove the final spectator measurements, insert the pulse train, re-measure
    body = c.copy_empty_like()
    meas = []
    for inst in c.data:
        if inst.operation.name == "measure" and c.find_bit(inst.qubits[0]).index in spect:
            meas.append(inst); continue
        body.append(inst.operation, inst.qubits, inst.clbits)
    for q in spect:
        if bunched:
            # campaign 13: 24 consecutive X (12 pairs) then one delay of the same total length
            for _ in range(pairs):
                body.x(q); body.x(q)
            body.append(Delay(2 * pairs * slot, "dt"), [q])
        else:
            for _ in range(pairs):
                body.append(Delay(slot, "dt"), [q]); body.x(q); body.append(Delay(slot, "dt"), [q]); body.x(q)
    for inst in meas:
        body.append(inst.operation, inst.qubits, inst.clbits)
    return body


def pad(tqc, tgt, qubits, spacing=None):
    """XX padding; spacing=[0, 0, 1] puts the two pulses back to back at the start of
    each idle window (campaign 13: same pulse count, no time spent inverted)."""
    kw = dict(dd_sequence=[XGate(), XGate()], qubits=list(qubits), target=tgt)
    if spacing is not None:
        kw["spacing"] = spacing
    pm = PassManager([ALAPScheduleAnalysis(target=tgt), PadDynamicalDecoupling(**kw)])
    return pm.run(tqc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ibm_marrakesh")
    ap.add_argument("--condition", choices=CONDITIONS)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    be, tgt, tqc, active, spect, images = build(a.backend)
    print(f"backend {a.backend}: active qubits {active}\n  spectators {spect}")
    circuits = {}
    for cond in CONDITIONS:
        if cond == "nodd": c = tqc
        elif cond == "dd_active": c = pad(tqc, tgt, active)
        elif cond == "dd_spect": c = pad(tqc, tgt, spect)
        elif cond == "dd_all": c = pad(tqc, tgt, active + spect)
        elif cond == "dd_spect_matched": c = pad_spectators_matched(tqc, tgt, spect)
        elif cond == "dd_active_bunched": c = pad(tqc, tgt, active, spacing=[0.0, 0.0, 1.0])
        elif cond in DOSE: c = pad(tqc, tgt, active, spacing=[0.0, DOSE[cond], 1.0 - DOSE[cond]])
        else: c = pad_spectators_matched(tqc, tgt, spect, bunched=True)
        n2 = sum(v for k, v in c.count_ops().items() if k in ("cz", "cx", "ecr", "rzz"))
        nx = c.count_ops().get("x", 0) + c.count_ops().get("y", 0)
        circuits[cond] = (c, n2, nx)
        print(f"  {cond:10s} two-qubit {n2}  single-qubit x/y {nx}  depth {c.depth()}")
    if a.dry_run or not a.condition:
        return
    c, n2, nx = circuits[a.condition]
    Options = ibm._sampler_options_cls(); opts = Options(); opts.twirling.enable_measure = True
    Sampler = ibm._sampler_v2_cls(); sampler = Sampler(mode=be, options=opts)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    for rep in range(1, a.repeat + 1):
        camp = {"dd_spect_matched": 12, "dd_active_bunched": 13, "dd_spect_matched_bunched": 13}.get(a.condition, 11)
        if a.condition in DOSE:
            camp = 15
        elif a.backend != "ibm_marrakesh":
            camp = 14
        label = f"c{camp}_{a.condition}_{a.backend.replace('ibm_', '')}_r{rep}"
        t0 = time.time(); job = sampler.run([c], shots=a.shots); jid = job.job_id(); res = job.result()
        counts = dict(res[0].data.meas.get_counts()); spec_counts = dict(res[0].data.spec.get_counts())
        d = OUT / stamp / "runs" / f"{label}_hw"; d.mkdir(parents=True, exist_ok=True)
        json.dump(counts, open(d / "counts.json", "w")); json.dump(spec_counts, open(d / "spectator_counts.json", "w"))
        p1 = []
        for i in range(len(spect)):
            ones = sum(v for k, v in spec_counts.items() if k[::-1][i] == "1"); p1.append(ones / a.shots)
        meta = dict(label=label, campaign=camp, condition=a.condition, dataset="canonical_shared", n=1, q=2, divider="nonrestoring",
                    load="ucry", backend=a.backend, shots=a.shots, mitigation="trex(manual-dd)", job_id=jid,
                    two_q_gate_count=int(n2), single_q_xy=int(nx), active_qubits=active, spectator_qubits=spect,
                    spectator_p1=p1, wallclock_seconds=round(time.time() - t0, 1))
        json.dump(meta, open(d / "metadata.json", "w"), indent=2)
        print(f"[{label}] job {jid}  spectator P(1) = {[round(x, 4) for x in p1]}")


if __name__ == "__main__":
    main()
