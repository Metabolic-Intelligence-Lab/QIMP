# Hardware campaign 9: pre-registered protocol

Written 10 Sep 2026 after campaign 8 and before any campaign-9 job. Fixed.

## Purpose

Every 4 x 4 and 8 x 8 result (campaigns 7–8) is on `ibm_kingston`. The
remaining monthly allowance is 32 s. Three 4 x 4 jobs at 16 384 shots cost about
21 s. This campaign tests whether the 16-pixel decode transfers to the weaker
device, `ibm_marrakesh`, where the 2 x 2 four-value target decoded at 70 % of
pixels with the flat-field decoder.

## Configuration (TREX only, non-restoring, q = 2, `--load ucry`)

| Id | Dataset | n | Null | Shots | Repeats | Backend |
|---|---|---:|---:|---:|---:|---|
| C9 | `fourvalue` (tiled) | 2 | 4/16 | 16 384 | 3 | ibm_marrakesh |

## Decoders and scoring

argmax and A flat-field, unchanged; scoring as Table 15.

## Predictions

- flat-field >= 14/16 in >= 2/3 runs: the 4 x 4 decode is not device-specific.
- 10–14/16: partial, consistent with the 70 % this device gives at 2 x 2.
- <= 8/16: device-specific; reported as such.

## Deviation log

- 11 Sep 2026: the campaign-9 entry had been appended to the scorer file after its dispatch block, so `--campaign7` did not see it; the entry was moved into the CONFIGS7 dictionary. No change to decoders or scoring.
