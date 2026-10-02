# Hardware campaign 13: population inversion as the mechanism of the decoupling penalty

Written 11 Sep 2026 after campaign 12, before any campaign-13 job. Fixed.

## Hypothesis

A decoupling sequence holds a qubit in the inverted state for about half of
each padded idle window. During that time T1 relaxation converts |1> to |0>,
and the closing pulse maps the relaxed fraction to |1>: dephasing protection
is bought with T1-induced bit flips. For a circuit that computes in the
computational basis and measures in it, dephasing between branches does not
change the measured statistics, so the trade is a pure loss. Quantitatively,
for the matched spectators of campaign 12 (12 XX pairs, 1.65 µs per slot,
circuit duration 40.6 µs) the T1-only prediction is 0.20 on qubit 68 (T1 =
79 µs; measured 0.194), 0.08 on qubit 93 (T1 = 243 µs; measured 0.093) and
0.10–0.12 on the others (measured 0.18–0.30), so T1 inversion is the leading
term. For the active register, about 65 % of the 40.6 µs is idle; padding
inverts half of it, 13 µs, giving a T1 bit-flip probability of 6–13 % per
data qubit at the layout's T1 of 94–219 µs.

## Test

Same circuit, layout and spectators as campaigns 11–12. Two conditions, four
runs of 4096 shots each, TREX on, runtime decoupling off:

| Condition | Pulses | Time inverted per window |
|---|---:|---|
| dd_active_bunched | 562 on the active register (XX, spacing [0, 0, 1], both pulses at the start of each idle window) | one X duration (36 ns) |
| dd_spect_matched_bunched | 24 consecutive X on each spectator, then one delay | one X duration per pair |

## Predictions (fixed)

- If the mechanism is population inversion: dd_active_bunched shows no
  penalty (sigma over null >= 15, separation >= +0.11, flat-field match
  >= 3.5/4, against 4.8 sigma, +0.030 and 1.75/4 for the spread XX of
  campaign 11), and dd_spect_matched_bunched brings the spectators back to
  within 0.02 of their no-pulse baseline (0.038) against 0.198 when spread.
- If the mechanism is the pulses themselves (crosstalk or per-pulse error):
  dd_active_bunched shows the same penalty as dd_active, and the bunched
  spectators stay excited.
- Intermediate: both contribute; the ratio of the bunched to the spread
  penalty measures the share.

## Commands

    for c in dd_active_bunched dd_spect_matched_bunched; do .venv/bin/python scripts/run_spectator_experiment.py --condition $c --repeat 4; done
    .venv/bin/python scripts/analyse_spectators.py

## Deviation log

(empty)
