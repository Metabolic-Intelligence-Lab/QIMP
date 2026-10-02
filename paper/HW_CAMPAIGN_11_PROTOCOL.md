# Hardware campaign 11: spectator-qubit test of the decoupling penalty

Written 11 Sep 2026 before any campaign-11 job. Fixed; deviations logged below.

## Purpose

§6.3 leaves the mechanism of the dynamical-decoupling penalty open between two
candidates: pulse-induced crosstalk on the two dozen simultaneously driven
qubits, and the scheduling rearrangement that padding forces. This campaign
measures idle spectator qubits, physical neighbours of the active register
that the circuit never touches, under four padding conditions on one fixed
layout, with readout twirling and the runtime's own decoupling disabled.
Padding is applied by `PadDynamicalDecoupling` with the XX sequence (the
target has no native Y; campaign 6 found XX and XY4 indistinguishable).

Circuit: balanced 2 x 2 Class-B, non-restoring, uniformly controlled load,
`ibm_marrakesh`, seed 0, 537 routed two-qubit gates, 24 active qubits
[69, 78, 84–92, 97, 98, 105–111, 117, 118, 125, 126], six spectators
[68, 70, 77, 83, 93, 104].

| Condition | XX padding on | Pulses added | Runs | Shots |
|---|---|---:|---:|---:|
| nodd | nothing | 0 | 4 | 4096 |
| dd_active | the 24 active qubits | 562 | 4 | 4096 |
| dd_spect | the 6 spectators only | 24 | 4 | 4096 |
| dd_all | active and spectators | 586 | 4 | 4096 |

## Observables (fixed)

- Active read-out: modal weight and sigma over the uniform null, flat-field
  match and per-pixel separation, exactly as Table 11.
- Spectators: P(1) per spectator qubit per run, with binomial error; pooled
  mean per condition.

## Predictions

- If manual XX padding on the active qubits reproduces the penalty (sigma
  over null falls by about 3x from nodd to dd_active), the penalty is a
  property of the pulses, not of the runtime's scheduling.
- If dd_active raises spectator P(1) above nodd by more than 3 binomial
  sigma pooled, pulse-induced excitation of neighbours is demonstrated.
- If dd_spect degrades the active read-out relative to nodd, crosstalk from
  neighbours' pulses into the active register is demonstrated directly
  (24 pulses only: a weak test, stated as such).
- If dd_active reproduces the penalty and neither spectator effect appears,
  the mechanism is internal to the padded qubits and crosstalk is excluded.
- If manual padding does not reproduce the penalty, the penalty lies in the
  runtime's decoupling implementation and not in the pulses as such.

## Commands

    for c in nodd dd_active dd_spect dd_all; do .venv/bin/python scripts/run_spectator_experiment.py --condition $c --repeat 4; done
    .venv/bin/python scripts/analyse_spectators.py

## Deviation log

(empty)
