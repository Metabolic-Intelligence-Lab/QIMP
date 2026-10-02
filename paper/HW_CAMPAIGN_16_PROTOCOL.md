# Hardware campaign 16 — does the 8x8 pixel-specific arithmetic reproduce, and is it the circuit or the time?

Pre-registered 2026-09-11, before any job of this campaign was submitted.

## Background (known when this was written)

Table 14 (`scripts/arithmetic_identity.py`) shows pixel-specific arithmetic on the random 8x8 image
(`random4`, n = 3, q = 2, non-restoring, uniformly controlled load) in both runs of campaign 7
(`c7_d`, 2026-09-10 17:46 UTC: identity excess over the cross null on q >= 1 pixels +0.013 and
+0.012, true triple the most frequent one in 14 and 15 of 29 q >= 1 pixels, per-run constant null 7)
and in none of the three runs of campaign 8 (`c8_a`, 18:36 UTC: excess -0.002 to +0.001, 7 to 8 of
29). Routed CX: c7 1086 and 1234, c8 1227 to 1256. Layouts: c7 r1 and all c8 runs use overlapping
regions (qubits ~0-69), c7 r2 a disjoint region (83-137). Neither the CX count nor the region
separates the runs.

## Design

Two archived, already-transpiled circuits are re-executed unchanged (same layout, same routing)
on `ibm_kingston` with `scripts/rerun_transpiled.py`, TREX only, no decoupling, 40 960 shots each
(640 per pixel):

- A = `c7_d_random4_n3_ucry_kingston_r1_hw` (1086 CX; signal present on 2026-09-10)
- B = `c8_a_random4_n3_ucry_kingston_r1_hw` (1227 CX; signal absent on 2026-09-10)

Order A, B, A, B, A, B (labels `c16_A_kingston_r1..3`, `c16_B_kingston_r1..3`). Budget: ~16 s
per job from the campaign-7/8 accounting (65 536 shots ~ 25 s); 109 s available. Stop rule: a job
is not submitted if fewer than 17 s remain; the campaign is then scored on the jobs that ran.

## Per-run criterion (fixed before data)

A run carries pixel-specific arithmetic if both hold on its q >= 1 pixels: identity excess over
the cross null > 0.005, and the true triple is the most frequent one in >= 9 of 29 pixels.
Calibration by subsampling the campaign-7/8 runs to 40 960 shots (30 draws each): the rule
classifies campaign-7 runs as present in 92 % of draws and campaign-8 runs in 0 %.

## Hypotheses and outcome bands

- H-circuit (the compiled circuit carries it): A present, B absent. Band: >= 2 of 3 A present and
  <= 1 of 3 B present, with A - B difference.
- H-time (device state on 2026-09-10 changed between 17:46 and 18:36): A and B alike.
  Band "reproducible now": >= 2 of 3 A and >= 2 of 3 B present -> 64-pixel arithmetic reproduces
  on the current device state, and campaign 8 was a drift episode.
  Band "absent now": <= 1 of 6 present -> campaign 7 was a transient device state.
- Anything else: intermittent; report the frequency of present runs with its binomial interval.

Secondary (reported, not used for the bands): flat-field decode accuracy against the constant null
35/64; joint argmax over all pixels against the constant null 9/64 per run; per-register d/q/r.

## Commands

    for k in 1 2 3; do
      .venv/bin/python scripts/rerun_transpiled.py --source c7_d_random4_n3_ucry_kingston_r1_hw --label c16_A_kingston_r$k --shots 40960
      .venv/bin/python scripts/rerun_transpiled.py --source c8_a_random4_n3_ucry_kingston_r1_hw --label c16_B_kingston_r$k --shots 40960
    done
    .venv/bin/python scripts/arithmetic_identity.py --prefix c16_

## Deviation log

(empty)
