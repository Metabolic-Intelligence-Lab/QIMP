# Hardware campaign 7: pre-registered protocol

Written 10 Sep 2026 after campaign 6 and before any campaign-7 job. Fixed;
deviations logged at the bottom.

## Purpose

The 4 x 4 and 8 x 8 circuits were depth-limited by the NEQR load: one
multi-controlled X per set pixel bit, 1005 basis CX at n = 2 and 5100–5500 at
n = 3. Loading each bit-plane with one uniformly controlled Ry (Möttönen et
al. 2004; `load="ucry"` in `class_b_ratio`, added 10 Sep 2026) costs 2^(2n)
CNOTs per plane and gives the same state (statevector overlap 1.0000, MPS
decode 16/16 and 64/64). Measured on the `ibm_kingston` target, best of four
routing seeds: 4 x 4 tiled 577 CX, 4 x 4 random 676 CX, 8 x 8 tiled 580 CX,
8 x 8 random 1125 CX, against 1557 / 1518 / 7416 / 6051 with the old load.
The 4 x 4 now costs what the 2 x 2 cost when it decoded 8/8.

## Configurations (all TREX only, non-restoring, q = 2, `--load ucry`, ibm_kingston)

| Id | Dataset | n | Pixels | Null | Shots | Repeats | Expected QPU |
|---|---|---:|---:|---:|---:|---:|---:|
| C7a | `fourvalue` (tiled) | 2 | 16 | 4/16 | 16 384 | 4 | ~30 s |
| C7b | `random4` (seed 1, non-periodic) | 2 | 16 | 5/16 | 16 384 | 4 | ~30 s |
| C7c | `canonical_shared` (real Laurdan patch) | 2 | 12 valid | 6/16 | 16 384 | 3 | ~22 s |
| C7d | `random4` (seed 1) | 3 | 64 | see scorer | 65 536 | 2 | ~60 s |

C7a/C7b/C7c are the claim; C7d is a probe at 1125 CX with 1024 shots per pixel.

## Decoders and scoring

argmax and A flat-field, unchanged. Per configuration: match per run against
the null, runs above null with the sign test, pooled pixel accuracy with the
exact binomial against 1/4 (independence approximation, stated), confusion
matrix, residual on the true bin.

## Predictions

- C7a or C7b: flat-field >= 14/16 in >= 3/4 runs: "16-pixel image divided on
  hardware with the quotients decoded", into the abstract. 10–14/16 typical:
  partial, into the results with numbers. <= 8/16: the load was not the only
  limit at n = 2.
- C7b vs C7a: the non-periodic target is the one that counts; the tiled one is
  reported beside it.
- C7c: >= 10/12 valid pixels in >= 2/3 runs: a real microscopy 4 x 4 patch
  decoded on hardware.
- C7d: pixel accuracy >= 50 % over the two runs would be signal at 64 pixels;
  reported as a probe.

## Commands

    bash scripts/run_campaign_7.sh
    .venv/bin/python scripts/decode_bias_corrected.py --campaign7

## Deviation log

- 10 Sep 2026, before the first job returned: the null for `random4` at n = 2 in the configuration table was written as 5/16; the seeded target has quotient values {0: 6, 1: 7, 3: 3}, so the constant-read-out null is 7/16. The scorer computes the null from the target and was never wrong; the table entry is corrected here. Nothing else changed.
