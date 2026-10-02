# Hardware campaign 10: pre-registered protocol

Written 11 Sep 2026 after campaign 9 and before any campaign-10 job. Fixed;
deviations logged at the bottom. Runs on a fresh open-plan allowance.

## Purpose

1. Device generality of the 4 x 4 decode: `ibm_fez` has no 4 x 4 run;
   `ibm_marrakesh` has three (65 %, one at 16/16).
2. Statistics of the real-microscopy 4 x 4 result: the Laurdan patch has three
   runs on `ibm_kingston` (12/12, 11/12, 7/12).
3. A real-microscopy 8 x 8: the Laurdan frame block-meaned to 8 x 8 on the
   shared photometric scale (`canonical_shared`, n = 3, offset (8,12),
   selected by `scripts/select_shared_scale_patch.py --n 3 --block 4`):
   35 valid pixels with quotients 0 and 1, 29 divide-by-zero pixels (dark
   background), constant null 29/64.

## Configurations (TREX only, non-restoring, q = 2, `--load ucry`)

| Id | Dataset | n | Shots | Repeats | Backend | Expected QPU |
|---|---|---:|---:|---:|---|---:|
| C10a | `fourvalue` (tiled) | 2 | 16 384 | 3 | ibm_fez | ~21 s |
| C10b | `fourvalue` (tiled) | 2 | 16 384 | 3 | ibm_marrakesh | ~21 s |
| C10c | `canonical_shared` (Laurdan 4 x 4) | 2 | 16 384 | 3 | ibm_kingston | ~21 s |
| C10d | `canonical_shared` (Laurdan 8 x 8) | 3 | 65 536 | 3 | ibm_kingston | ~75 s |
| C10e | `random4` | 2 | 16 384 | 3 | ibm_marrakesh | ~21 s |

## Decoders and scoring

argmax and A flat-field, unchanged; scoring as Table 13. C10b is pooled with
campaign 9 (six runs); C10c with campaign 7 C7c (six runs). Divide-by-zero
pixels are scored on the flag, as everywhere.

## Predictions

- C10a: >= 10/16 in >= 2/3 runs: the decode generalises to the weakest device
  in part; <= 6/16: device-specific, reported.
- C10b pooled (6 runs): mean >= 11/16 keeps `ibm_marrakesh` at "partial";
  >= 14/16 in >= 4/6 would move it to the claim band.
- C10c pooled (6 runs): >= 10/12 in >= 4/6 runs makes the Laurdan 4 x 4 a
  claim rather than a partial result.
- C10d: >= 70 % of the 35 valid pixels in >= 2/3 runs: a 64-pixel microscopy
  patch decoded on hardware; 45–70 %: partial; below: reported.
- C10e: as C10b for the random image.

## Deviation log

(empty)
