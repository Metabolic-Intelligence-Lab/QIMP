# Hardware campaign 17 — 64-pixel division with the truth-table divider

Pre-registered 2026-09-11, before any job of this campaign was submitted.

## Rationale (known when this was written)

With the long-division divider (`q_div_nonrestoring`, 189 CNOTs and 24 qubits at q = 2) the
Class-B circuit decodes at 2x2 and 4x4 (537 to 724 routed two-qubit gates) and not at 8x8
(987 to 1256), and campaign 16 showed that the 8x8 pixel-specific arithmetic of campaign 7 does
not reproduce. `q_div_lookup` (src/qimp/processing/arithmetic.py) computes the same function,
a // d with the all-ones quotient and flag at d = 0, from its truth table by fixed-polarity
Reed-Muller synthesis: 48 CNOTs, no ancilla, both operands preserved, self-inverse, exact on
every operand pair (tests/unit/test_div_lookup.py, q = 1, 2, 3; bit-exact Class-B circuits at
n = 1, 2). Routed on `ibm_kingston` (best of 16 seeds): 8x8 564-573, 4x4 116-206, 2x2 111
two-qubit gates, against 987-1158, 577-719 and 537-579 with the long-division divider. The
8x8 circuit is thereby brought to the depth at which the 4x4 circuit decoded (flat-field 98 %
tiled, 84 % random over four runs each, Table 13).

## Design

`scripts/run_campaign_17.py`. Divider `lookup`, uniformly controlled load, q = 2, TREX only,
no decoupling, `optimization_level=3`, best of 16 transpiler seeds per target by two-qubit
count, fixed and written to disk with SHA-256 before the first submission.

Phase main, `ibm_kingston`, three rounds, each round submitting every target once in this order:
8x8 random (`random4`), 8x8 Laurdan (`canonical_shared`, 35 valid pixels), 8x8 Fura-2 (54 valid),
8x8 balanced (`balanced8`), 4x4 tiled four-value, 4x4 random, 4x4 Laurdan (12 valid), 2x2
four-value, 2x2 Laurdan. Shots 65 536 / 16 384 / 4 096 (1024 per pixel at 8x8 and 4x4). 27 jobs,
~290 s estimated.
Phase replication, `ibm_marrakesh`, three rounds of 8x8 random, 4x4 random, 2x2 four-value.
9 jobs, ~96 s estimated. Run only after phase main.
Stop rule: a job is skipped if the remaining allowance is below its estimate plus 5 s; the
campaign is scored on the jobs that ran and every run is reported.

## Analysis (fixed before data)

`scripts/arithmetic_identity.py` (identity event for the lookup divider: the measured dividend
equals the pixel's dividend and the quotient register equals the integer quotient of the
measured dividend and divisor; cross and independent nulls as in Table 14), the flat-field and
argmax decoders of `scripts/analyse_hw_signal.py` / Table 13, constant-read-out nulls over the
valid pixels.

## Predictions and bands

P1 (8x8, decoder-free, per run): the number of pixels whose most frequent (q, a, d) triple is
the true one exceeds the run's constant-read-out null, and the own-pixel identity excess over
the cross null on the pixels with q >= 1 is positive.
- "reproducible 64-pixel arithmetic": P1 in all three runs for at least 3 of the 4 targets;
- "partial": P1 in at least 6 of the 12 runs;
- otherwise "not reproduced".

P2 (8x8 decode): pooled flat-field accuracy over three runs >= 75 % on the random and balanced
targets -> "decoded"; 55-75 % -> "partial"; below -> "not decoded". Laurdan and Fura-2 violate
the balanced-value assumption of the flat-field decoder; for them the argmax is scored against
the constant null and counts if above it in at least 2 of 3 runs. Reference with the
long-division divider on the same images: random 50 %, balanced 38 % (Table 13).

P3 (4x4 and 2x2): pooled argmax accuracy >= 90 % and flat-field >= 95 % on every target.

P4 (replication, `ibm_marrakesh`): P1 on the 8x8 random target in at least 2 of 3 runs.

Secondary, recorded and not used for the bands: the `AerSimulator.from_backend` noise model of
`ibm_kingston` predicts 100 % argmax at every size (`scripts/predict_lookup_noise.py`,
`paper/data_autonomous/lookup_noise_prediction_ibm_kingston.json`); earlier campaigns show this
model to be optimistic, and it is recorded to document that.

## Commands

    .venv/bin/python scripts/run_campaign_17.py --backend ibm_kingston --rounds 3 --phase main
    .venv/bin/python scripts/run_campaign_17.py --backend ibm_marrakesh --rounds 3 --phase replication
    .venv/bin/python scripts/arithmetic_identity.py --prefix c17_

## Deviation log

(empty)

## Deviation log (continued)

- 2026-09-25 18:44 UTC: the driver lost its HTTPS connection to the runtime while polling the
  ninth job of round 1 (`RequestsApiError`, `Max retries exceeded`) and exited. The job itself
  had run on the device; its result was fetched afterwards from the job id and persisted with
  the `recovered` field set in its metadata, so round 1 is complete on all nine targets. No job
  was resubmitted and no allowance was spent twice.
- Rounds 2 and 3 were relaunched with `--reuse` on the transpiled circuits of round 1, whose
  SHA-256 the driver re-checks before submitting, and with `--start-round 2`. Re-using the
  round-1 circuits makes the rounds a test of reproducibility in time at a fixed compilation,
  which is stricter than the original design, where each launch re-transpiled. The driver also
  skips a label whose run directory already exists, so an interrupted round can be resumed
  without resubmitting.
- 2026-09-25 19:05 UTC: the first relaunch of rounds 2 and 3 was pointed at the wrong archive
  directory and re-executed the 11 September transpilation of the `random4` 8x8 target before it
  was stopped. That job ran on the device. Its result is persisted under the label
  `c17x_random4_n3_lookup_kingston_altcompile_r1`, is reported as a second compilation of the
  same circuit, and is excluded from the bands. The relaunch was repeated against the round-1
  archive, whose fingerprints the driver verifies before submitting.
