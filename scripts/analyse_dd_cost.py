"""Why does XY4 decoupling *cost* signal on this circuit? (§7.4 mechanism)

§7.4 measures a factor of ~3 in signal margin between XY4-on and XY4-off
and states the natural reading -- that a circuit with few idle windows
gains little dephasing suppression while paying for every inserted pulse
-- but does not test it. That hypothesis is quantitative and can be
checked without a QPU, because both halves of the trade are computable
from the scheduled circuit and the backend's calibration data:

  * **what XY4 pays**: the number of pulses the padding pass inserts,
    times the device's single-qubit error rate.
  * **what XY4 buys**: the free-induction dephasing accumulated in the
    idle windows it fills, set by the idle time against T2.

The ratio of the two is the prediction. Where it exceeds one, decoupling
is a net loss, and the pass is measuring exactly the quantity §7.4 leaves
as future work: the penalty as a function of two-qubit gate count.

Scheduling is done locally with the real backend Target, because the
runtime inserts DD server-side -- the transpiled circuits archived under
`data/output/ibm_hw/` do not contain the pulses.

Usage:
    python scripts/analyse_dd_cost.py                      # default sweep
    python scripts/analyse_dd_cost.py --backend ibm_fez
    python scripts/analyse_dd_cost.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from qiskit import transpile  # noqa: E402
from qiskit.circuit.library import XGate  # noqa: E402
from qiskit.transpiler import PassManager  # noqa: E402
from qiskit.transpiler.passes import (  # noqa: E402
    ALAPScheduleAnalysis,
    ASAPScheduleAnalysis,
    PadDelay,
    PadDynamicalDecoupling,
)

from qimp.processing.ratiometric_circuit import class_b_ratio  # noqa: E402
from qimp.runtime import ibm  # noqa: E402
from qimp.testing import _ensure_measured  # noqa: E402

# The configurations §7 actually ran, ordered by two-qubit gate count, so
# the DD penalty can be read against the depth axis §7.4 leaves untested.
CONFIGS = [
    ("canonical_shared", 1, 2),
    ("canonical", 1, 2),
    ("canonical", 1, 3),
    ("canonical", 2, 2),
]


def load_images(dataset: str, n: int, q: int) -> tuple[np.ndarray, np.ndarray]:
    from run_hardware_class_b_nonrestoring import load_dataset

    return load_dataset(dataset, n, q)


def device_rates(backend) -> dict[str, float]:
    """Median single-qubit error, two-qubit error, T1 and T2 of the device."""
    target = backend.target
    one_q, two_q = [], []
    for name in ("x", "sx", "rz"):
        if name not in target:
            continue
        for props in target[name].values():
            if props is not None and props.error is not None:
                one_q.append(props.error)
    for name in ("cz", "ecr", "cx"):
        if name not in target:
            continue
        for props in target[name].values():
            if props is not None and props.error is not None:
                two_q.append(props.error)
    t1 = [p.t1 for p in target.qubit_properties if p is not None and p.t1]
    t2 = [p.t2 for p in target.qubit_properties if p is not None and p.t2]
    return {
        "e_1q": float(np.median(one_q)) if one_q else float("nan"),
        "e_2q": float(np.median(two_q)) if two_q else float("nan"),
        "t1_s": float(np.median(t1)) if t1 else float("nan"),
        "t2_s": float(np.median(t2)) if t2 else float("nan"),
    }


def active_qubits(circ) -> list[int]:
    """Physical qubits carrying at least one real operation.

    The transpiled circuit spans all 156 physical qubits, but only the ~24
    in the layout do anything. Counting idle time on the 132 spectators
    would inflate both sides of the trade by an order of magnitude and
    measure the device's width rather than the circuit's schedule.
    """
    idx = {circ.find_bit(q).index for inst in circ.data
           for q in inst.qubits
           if inst.operation.name not in ("delay", "barrier")}
    return sorted(idx)


def idle_profile(scheduled, active: list[int], dt: float) -> dict:
    """Per-qubit idle time, split at each qubit's first real operation.

    Only the idle that follows a qubit's first gate can dephase: before it
    the qubit is still in |0>, an eigenstate of the noise channel, which is
    why `PadDynamicalDecoupling` defaults to `skip_reset_qubits=True` and
    leaves that leading window unpadded. Counting it on the benefit side
    would credit DD with protecting a state that cannot decay.
    """
    active_set = set(active)
    first_op: dict[int, int] = {}
    for inst, start in zip(scheduled.data, scheduled.op_start_times):
        if inst.operation.name in ("delay", "barrier"):
            continue
        for qb in inst.qubits:
            i = scheduled.find_bit(qb).index
            if i in active_set and i not in first_op:
                first_op[i] = start

    live: dict[int, float] = {i: 0.0 for i in active}
    leading = 0.0
    n_live_windows = 0
    for inst, start in zip(scheduled.data, scheduled.op_start_times):
        if inst.operation.name != "delay":
            continue
        i = scheduled.find_bit(inst.qubits[0]).index
        if i not in active_set:
            continue
        seconds = inst.operation.duration * dt
        if start < first_op.get(i, 0):
            leading += seconds
        else:
            live[i] += seconds
            n_live_windows += 1

    duration = (max(scheduled.op_start_times) if scheduled.op_start_times else 0) * dt
    total_live = float(sum(live.values()))
    return {
        "n_active_qubits": len(active),
        "n_live_idle_windows": n_live_windows,
        "idle_live_s": total_live,
        "idle_leading_s": float(leading),
        "circuit_duration_s": float(duration),
        # Share of the active qubit-time budget spent idling *in a state
        # that can dephase*. This is what DD exists to protect.
        "live_idle_fraction": (
            total_live / (len(active) * duration) if duration and active else 0.0
        ),
        "_per_qubit_live": live,
    }


def count_dd_pulses(scheduled_dd, scheduled_plain, active: list[int]) -> int:
    """Single-qubit pulses the padding added, on active qubits only."""
    active_set = set(active)

    def one_q_ops(circ) -> int:
        return sum(
            1
            for inst in circ.data
            if len(inst.qubits) == 1
            and inst.operation.name not in ("delay", "barrier", "measure")
            and circ.find_bit(inst.qubits[0]).index in active_set
        )

    return one_q_ops(scheduled_dd) - one_q_ops(scheduled_plain)


def _schedule(transpiled, backend, active, order: str):
    Sched = ASAPScheduleAnalysis if order == "asap" else ALAPScheduleAnalysis
    plain = PassManager(
        [Sched(target=backend.target), PadDelay(target=backend.target)]
    ).run(transpiled)
    with_dd = PassManager(
        [
            Sched(target=backend.target),
            # XY4 in the Heron basis {rz, sx, x, cz}: the Y pulses are X
            # pulses in a rotated frame, and the frame change is a virtual
            # Rz -- zero duration, zero error. XY4 and XX4 therefore have the
            # same pulse count and the same gate-error cost, which is the
            # quantity computed here; they differ in decoupling performance,
            # which enters on the benefit side.
            PadDynamicalDecoupling(
                target=backend.target,
                dd_sequence=[XGate()] * 4,
                qubits=active,
            ),
        ]
    ).run(transpiled)
    return plain, with_dd


def analyse_one(dataset: str, n: int, q: int, backend, rates: dict) -> dict:
    I_a, I_b = load_images(dataset, n, q)
    qc, _ = class_b_ratio(I_a, I_b, q=q, divider="nonrestoring")
    measured = _ensure_measured(qc)
    transpiled = transpile(measured, target=backend.target, optimization_level=3)

    dt = backend.target.dt or 0.5e-9
    active = active_qubits(transpiled)
    summary = ibm._transpile_summary(transpiled)

    out = {
        "dataset": dataset,
        "n": n,
        "q": q,
        "logical_qubits": qc.num_qubits,
        "two_q_gate_count": summary["two_q_gate_count"],
        "depth": summary["depth"],
    }
    # The runtime does not document which scheduling it applies before
    # inserting DD, and the two bracket the answer: ALAP front-loads the
    # idle (mostly harmless, qubit still in |0>), ASAP back-loads it
    # (fully exposed). Reporting both bounds the estimate honestly.
    for order in ("alap", "asap"):
        plain, with_dd = _schedule(transpiled, backend, active, order)
        prof = idle_profile(plain, active, dt)
        per_qubit = prof.pop("_per_qubit_live")
        n_pulses = count_dd_pulses(with_dd, plain, active)

        # Both sides are expected error counts summed over the active
        # register, so they are directly comparable.
        cost = n_pulses * rates["e_1q"]
        buys = float(sum(1.0 - np.exp(-t / rates["t2_s"]) for t in per_qubit.values()))
        out[order] = {
            "dd_pulses": n_pulses,
            "dd_error_cost": cost,
            "dephasing_recoverable": buys,
            "cost_benefit_ratio": cost / buys if buys > 0 else float("inf"),
            **prof,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="analyse_dd_cost", description=__doc__)
    ap.add_argument("--backend", type=str, default="ibm_marrakesh")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    service = ibm.get_service()
    backend = ibm.pick_backend(service, min_qubits=24, name=args.backend)
    rates = device_rates(backend)
    print(f"Backend: {backend.name}")
    print(
        f"  median e_1q = {rates['e_1q']:.2e}, e_2q = {rates['e_2q']:.2e}, "
        f"T1 = {rates['t1_s'] * 1e6:.0f} us, T2 = {rates['t2_s'] * 1e6:.0f} us\n"
    )

    rows = []
    for dataset, n, q in CONFIGS:
        try:
            rows.append(analyse_one(dataset, n, q, backend, rates))
        except Exception as exc:  # a missing derived patch should not kill the sweep
            print(f"  !! {dataset} n={n} q={q}: {exc}")

    hdr = (
        f"{'config':22s} {'CX':>6s} {'sched':>6s} {'live idle%':>11s} "
        f"{'windows':>8s} {'pulses':>8s} {'pays':>8s} {'buys':>8s} {'pays/buys':>10s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cfg = f"{r['dataset']} n={r['n']} q={r['q']}"
        for order in ("alap", "asap"):
            d = r[order]
            print(
                f"{cfg if order == 'alap' else '':22s} "
                f"{r['two_q_gate_count'] if order == 'alap' else '':>6} "
                f"{order:>6s} {100 * d['live_idle_fraction']:10.1f}% "
                f"{d['n_live_idle_windows']:8d} {d['dd_pulses']:8d} "
                f"{d['dd_error_cost']:8.3f} {d['dephasing_recoverable']:8.3f} "
                f"{d['cost_benefit_ratio']:10.2f}"
            )
    print(
        "\n  'live idle' is idle time after a qubit's first gate -- the only\n"
        "  idle that can dephase, and the only idle DD pads.\n"
        "  'pays' = inserted pulses x median single-qubit error.\n"
        "  'buys' = sum over active qubits of 1 - exp(-live idle / T2), the\n"
        "           upper bound on what perfect decoupling could recover.\n"
        "  A ratio above 1 means decoupling is a net loss under the\n"
        "  incoherent-error model -- the reading §7.4 offers untested."
    )

    if args.json:
        args.json.write_text(json.dumps({"backend": backend.name, "rates": rates,
                                         "rows": rows}, indent=2))
        print(f"\nWritten to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
