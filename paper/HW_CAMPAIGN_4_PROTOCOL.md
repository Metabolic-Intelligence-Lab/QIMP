# Hardware campaign 4: pre-registered protocol

Written 10 Sep 2026 after campaign 3 was scored (Table 13) and before any
campaign-4 job was submitted. Fixed; deviations logged at the bottom.

## Purpose

1. Statistical strength of the campaign-3 headline (four-value target decoded
   4/4 in 3/3 runs on `ibm_kingston`): three runs cap the sign test at
   p = 0.125. Five more runs at 16 384 shots bring the pooled test to eight
   runs and test whether the residual grows with shots on this device.
2. A third device (`ibm_fez`).
3. Scale: the same four-value pattern tiled to 4 x 4 (n = 2, 16 pixels, each
   quotient value four times, constant null 4/16; 26 qubits, 1005 basis CX,
   16/16 on the MPS simulator). Campaign 2 found no per-pixel separation at
   n = 2 on `ibm_marrakesh` (1250 routed CX). This probes `ibm_kingston`.
4. Without device time: a **per-position calibrated decoder** (C, below),
   the remedy named in §6.4 for the position-dependent residual on
   `ibm_marrakesh`, validated across targets.

## Configurations (all TREX only, non-restoring, q = 2, optimization_level=3)

| Id | Dataset | n | Shots | Repeats | Backend | Expected QPU |
|---|---|---:|---:|---:|---|---:|
| C4a | `fourvalue` | 1 | 16 384 | 5 | ibm_kingston | ~40 s |
| C4b | `fourvalue` | 1 | 4 096 | 3 | ibm_fez | ~12 s |
| C4c | `fourvalue` | 2 | 16 384 | 2 | ibm_kingston | ~50 s |

## Decoders

- argmax and A flat-field: exactly as Tables 12 and 13.
- **C, per-position calibrated offset** (new, fixed here). For device D, the
  calibration set is every archived TREX-only, non-restoring, n = 1 run on D of
  a target with known content (campaign 3 `fourvalue` runs on D; for
  `ibm_kingston` also C4a). For each calibration run and pixel, the offset is
  the histogram minus the one-hot of the true quotient; the per-position
  offset vector o_p is the mean over calibration runs of that difference at
  pixel p, with the mean over its four bins subtracted so that o_p sums to
  zero. Decode a held-out run as argmax(p_pixel - o_pixel). Held-out means: a
  run of a DIFFERENT target on the same device (balanced runs on
  `ibm_marrakesh`, calibrated from `fourvalue` on `ibm_marrakesh`), so that the
  calibration never sees the target it is scored on. Note o_p contains the
  calibration target's decoherence term on its own true bin; that is why
  cross-target validation is required and why the decoder may fail.

## Scoring

As Table 13, per configuration and decoder; for C4c the 4 x 4 confusion
matrix; for decoder C, the balanced `ibm_marrakesh` runs (campaign 3 C1 and the
15 archived TREX-only runs) scored with and without the calibration.

## Predictions

- C4a: flat-field 4/4 in >= 4/5 runs replicates the headline; pooled eight
  runs give the sign test p <= 0.03. Residual expected ~0.10 as at 4096 shots
  (device-set), not higher.
- C4b: 4/4 in >= 2/3 replicates on a third device; 2/4 or less says the
  effect is device-specific and the paper says so.
- C4c: most likely outcome is no per-pixel separation (as at n = 2 on
  `ibm_marrakesh`); any pixel accuracy above 40 % (chance 25 %, null 25 %)
  would be the first signal at 4 x 4 and is reported as such with its
  confusion matrix. Two runs only: a scale probe, not a claim.
- Decoder C on `ibm_marrakesh` balanced runs: >= 3.5/4 mean would show the
  residual is a stable per-position property of the device; <= 2.5/4 says it
  drifts between jobs and per-position calibration must be in-run.

## Commands

    bash scripts/run_campaign_4.sh
    .venv/bin/python scripts/decode_bias_corrected.py --campaign4

## Deviation log

(empty)
