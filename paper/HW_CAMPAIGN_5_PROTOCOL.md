# Hardware campaign 5: pre-registered protocol

Written 10 Sep 2026 after campaign 4 was scored and before any campaign-5 job
was submitted. Fixed; deviations logged at the bottom.

## Purpose

Campaign 4 C4c: the four-value pattern tiled to 4 x 4 (n = 2, 26 qubits,
1616–1701 routed CX) on `ibm_kingston`, two runs at 16 384 shots, flat-field
decode 12/16 and 5/16 (17/32 pixels, chance 8/32). Two runs are a probe.
Four more runs, same configuration, make six; the question is whether the
pooled per-pixel accuracy stays above chance with a defensible test.

## Configuration

| Id | Dataset | n | Shots | Repeats | Backend | Expected QPU |
|---|---|---:|---:|---:|---|---:|
| C5 | `fourvalue` | 2 | 16 384 | 4 | ibm_kingston | ~100 s |

TREX only, non-restoring, q = 2, optimization_level=3, fresh transpile per job.

## Decoders and scoring

argmax and A flat-field, unchanged (Tables 12–13). Scoring on the six pooled
runs (C4c + C5): match per run against the null 4/16; runs above null with the
sign test; pooled pixel accuracy with an exact binomial test against 1/4
treating the 96 pixel decodes as independent (an approximation, stated as
such: pixels within a run share a calibration); the 4 x 4 confusion matrix;
residual on the true bin. Also the accuracy-vs-shots curve of Fig. 7 style by
pooling the six runs in random orders (`--pooled` logic, n = 2).

## Predictions

- Pooled accuracy >= 45 % over six runs (>= 43/96): the 4 x 4 decode is a
  claim: "first per-pixel signal above chance on a 16-pixel image", goes into
  the abstract with its numbers.
- 30–45 %: signal present but weak; stays a probe, reported with the numbers.
- <= 30 %: the two campaign-4 runs were a fluctuation; reported as such.
- Per-run match: >= 8/16 in >= 3/6 runs would mean the majority of runs decode
  half the image or better.

## Commands

    bash scripts/run_campaign_5.sh
    .venv/bin/python scripts/decode_bias_corrected.py --campaign5

## Deviation log

(empty)
