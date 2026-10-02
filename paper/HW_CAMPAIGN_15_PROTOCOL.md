# Hardware campaign 15 — dose-response of the decoupling penalty in the inverted fraction f

Pre-registered 2026-09-11, before any job of this campaign was submitted. Written after
campaigns 11–14 (Table S15) and after the arithmetic-identity analysis of
`scripts/arithmetic_identity.py` on the existing runs, and before campaign 15 data existed.

## Question

Campaigns 13 and 14 showed that the same 562 XX pulses cost ~80 % less when placed back to
back (inverted for one pulse duration) than when spread (inverted for half of each idle
window). If the penalty is T1 relaxation during the inverted time, it must scale with the
fraction f of each idle window that the qubit spends inverted, and it must be signed by the
state held: a qubit holding |0> is exposed for a fraction f of its idle time (it decays only
while inverted), a qubit holding |1> is exposed for a fraction 1 - f (it is protected while
inverted). Pulse-driven crosstalk or scheduling effects do not depend on f.

## Design

Balanced 2x2 Class-B circuit, non-restoring divider, uniformly controlled load, `ibm_kingston`,
same fixed layout as campaign 14 (active qubits 48–51, 58, 59, 68–75, 78, 79, 89–95, 98;
spectators 38, 47, 52, 55, 67, 88; 537 routed two-qubit gates; 562 XX pulses on the active
qubits in every padded condition; TREX on; runtime decoupling off). XX with the first pulse at
the start of each idle window and the second at fraction f of the window
(`PadDynamicalDecoupling` spacing [0, f, 1 - f]), f in {0.25, 0.50, 0.75, 1.00}, four runs of
4096 shots each (16 jobs, ~48 s QPU). Together with campaign 14 (f = 0 bunched; f = 1/2 with
symmetric placement [1/4, 1/2, 1/4]; unpadded) this gives a five-point dose-response.

Dataset `canonical_shared`: I_a = [[1, 2], [0, 2]], I_b = [[1, 3], [1, 2]] (divisor bits are
1-rich: 5 of 8 bits set), quotients [[1, 0], [0, 1]], remainders [[0, 2], [0, 0]].

## Predictions (falsifiable)

P1. The per-pixel separation d1 - d0 of the quotient read-out decreases monotonically with f,
    and a linear fit over the five points has R^2 > 0.9. Crosstalk or scheduling predicts no
    dependence on f (all padded points equal within run scatter, sd ~0.02).
P2. The loss at f = 1 exceeds the loss at f = 1/2 (spread, campaign 14) because the register
    is |0>-dominated during idle time (ancillae, work and quotient bits before they are set).
P3. Signed by state: the divisor register I_b, which holds 1-rich values for the whole circuit,
    does not lose read-out accuracy with f (its true-value share stays within run scatter of
    the unpadded value or rises), while the quotient and remainder registers lose accuracy
    with f. This is the T1-inversion signature and no pulse-count mechanism produces it.
P4. Spectator excitation on the register's neighbours changes by less than 0.02 across f
    (pulse count is fixed), except qubit 55, which campaign 14 showed responds to the spacing.

## Scoring

`scripts/analyse_spectators.py` (separation, sigma over null, flat-field match, spectators) and
`scripts/arithmetic_identity.py --prefix c15_` (per-register true-value shares q, r, d and the
identity rate). Points at f = 0 and f = 1/2 (symmetric) and the unpadded reference come from
campaign 14 on the same layout. Linear fit of separation on f with the five padded points.

## Commands

    for c in dd_active_f025 dd_active_f050 dd_active_f075 dd_active_f100; do .venv/bin/python scripts/run_spectator_experiment.py --backend ibm_kingston --condition $c --repeat 4; done

## Deviation log

- 2026-09-11 11:49 UTC, after the jobs were submitted and before any result was read: the dataset
  values line had been typed from memory and was wrong (fingerprint 5bc2303a...). Corrected to
  the values printed by `load_dataset`; the predictions P1–P4 are unchanged (I_b is still 1-rich).
  Second fingerprint recorded below the first in `campaign15_protocol.sha256`.

## Addendum (pre-registered 2026-09-11 after the kingston runs were scored, before any marrakesh job)

Replication of the dose-response on `ibm_marrakesh`, on the fixed layout of campaigns 11–13
(active qubits 69, 78, 84–92, 97, 98, 105–111, 117, 118, 125, 126; spectators 68, 70, 77, 83,
93, 104; 537 routed two-qubit gates; 562 XX pulses), f in {0.25, 0.50, 0.75, 1.00}, four runs
each (16 jobs, ~48 s). Together with campaign 13 (f = 0 bunched), campaign 11 (symmetric f = 1/2
and unpadded) this gives the same five-point dose on a second device. Predictions P1–P3 apply
unchanged; P4 applies to all six marrakesh spectators (no neighbour was singled out there).
Kingston outcome, known at the time of writing: margin linear in f (R^2 = 0.96), separation
monotone with the largest step at f = 0 to 0.25, f = 1 worst, divisor register flat through
f = 0.75. Prediction P5 for marrakesh: the same ordering, with a linear fit R^2 > 0.9 on the
margin.

Command: for c in dd_active_f025 dd_active_f050 dd_active_f075 dd_active_f100; do .venv/bin/python scripts/run_spectator_experiment.py --backend ibm_marrakesh --condition $c --repeat 4; done
