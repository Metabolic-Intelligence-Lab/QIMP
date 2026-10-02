# Hardware campaign 3: pre-registered protocol

Written 10 Sep 2026, before any job of this campaign was submitted. Nothing below
is to be changed after the first job runs; deviations, if any, are to be logged
at the bottom with the reason.

## Purpose

Two open questions from §6.4 of the v2 manuscript (Table 12, Fig. 7):

1. On the balanced target, is the flat-field decoder's shortfall shot noise or a
   position-dependent read-out offset? Pooling the archived 4096-shot runs put
   accuracy at 75 % and flat; a single job at four times the shots tests that
   directly.
2. Does the flat-field decoder return the **whole q = 2 codomain**? Every target
   so far had at most two distinct quotient values. A target with R = {0,1,2,3},
   one value per pixel, has a constant-read-out null of 1/4 and leaves no room
   for a biased argmax to score.

## Configurations

All: n = 1, q = 2, non-restoring divider, `--mitigation trex` (readout twirling,
no dynamical decoupling), `optimization_level=3`, one fresh transpile per job
(routing-seed variation is part of the measured scatter, as in campaign 2).

| Id | Dataset | Target R | Null | Shots | Repeats | Backend |
|---|---|---|---:|---:|---:|---|
| C1 | `canonical_shared` | [[1,0],[0,1]] | 2/4 | 16 384 | 5 | ibm_marrakesh |
| C2 | `fourvalue` | [[0,1],[2,3]] | 1/4 | 16 384 | 5 | ibm_marrakesh |
| C3 | `fourvalue` | [[0,1],[2,3]] | 1/4 | 4 096 | 3 | ibm_kingston |

13 jobs. Expected QPU charge: about 16 s per 16 384-shot job and 4 s per
4096-shot job (from campaign-2 metrics), roughly 170 s in total, inside the
600 s monthly open-plan allowance.

`fourvalue` is I_a = [[1,3],[2,3]], I_b = [[2,3],[1,1]] (added to
`scripts/run_hardware_class_b_nonrestoring.py` on 10 Sep 2026); it decodes 4/4
on the MPS simulator at 279 basis CX, every division non-trivial, no
divide-by-zero pixel.

## Decoders (fixed)

- **argmax** (baseline): mode of each pixel's quotient histogram.
- **A, flat-field** (`scripts/decode_bias_corrected.py`, unchanged): subtract
  the run's mean histogram over pixels, then argmax.

No other decoder will be applied to these runs before the paper is revised. No
threshold, weight or calibration is fitted on these runs.

## Scoring (fixed)

Per configuration, exactly as Table 12: match per run against the constant
null, runs strictly above the null with the one-sided sign test, pooled
per-pixel accuracy, modal weight and sigma over the uniform null, residual on
the true bin after flat-field correction (as §6.4). For C2/C3 additionally the
4 x 4 confusion matrix of decoded vs true quotient, pooled over runs.

## Predictions and what each outcome means

- C1, accuracy at 16 384 shots per run:
  - >= 3/4 in >= 4/5 runs with the same missed pixel as the pooled archive:
    the residual is position-dependent, shot noise was not the limit; the
    paper says so and names per-position calibration as the fix.
  - 4/4 in >= 4/5 runs: the pooled-archive plateau was job-to-job drift, not a
    fixed offset; the balanced target is recovered at 4x shots.
  - <= 2/4 typical: the decoder does not generalise to a fresh job on this
    target; report as such.
- C2, flat-field decode:
  - 4/4 in >= 4/5 runs: full codomain decoded; this becomes the headline
    hardware result and the Fura-2 result its replication.
  - 3/4 typical with one value systematically confused: report the confusion.
  - argmax is expected at or near the 1/4 null throughout.
- C3: same decoder on a second device; agreement with C2 within one pixel
  per run counts as replication.

## Commands

    bash scripts/run_campaign_3.sh            # pre-flight quota check, then all 13 jobs
    .venv/bin/python scripts/decode_bias_corrected.py --campaign3   # scoring

## Deviation log

- 10 Sep 2026, after C1 (5/5 runs archived, 35 s QPU): the launcher aborted at C2 because `fourvalue` had been added to `load_dataset` but not to the `--dataset` argparse choices. Fixed the choices list (no change to the target, decoders or scoring) and relaunched C2 and C3 only; C1 was not repeated.
