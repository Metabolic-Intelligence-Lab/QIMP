# Hardware campaign 8: pre-registered protocol

Written 10 Sep 2026 after campaign 7 was scored and before any campaign-8 job.
Fixed; deviations logged at the bottom.

## Purpose

Campaign 7 C7d: 8 x 8 random image, uniformly controlled load, two runs,
flat-field 74/128 pixels (58 %). Two runs are a probe. Three more runs make
five. A second 64-pixel image with biophysical content, the synthetic Fura-2
calcium image at n = 3 (`fura2`, seed 1: 54 valid pixels, quotient 0 on 39 and
2 on 15, ten divide-by-zero pixels, a ring-shaped cell), tests the high
quotient bit at this size. The 8 x 8 Laurdan patch quantised per channel
(`canonical`, n = 3) is degenerate (34 divide-by-zero pixels, 28 of 30 valid
pixels at quotient 1) and is not run.

## Configurations (TREX only, non-restoring, q = 2, `--load ucry`, ibm_kingston, 65 536 shots)

| Id | Dataset | n | Valid px | Null | Repeats | Routed CX (measured) | Expected QPU |
|---|---|---:|---:|---:|---:|---:|---:|
| C8a | `random4` (seed 1) | 3 | 64 | 35/64 | 3 | 1086–1234 | ~75 s |
| C8b | `fura2` (seed 1) | 3 | 54 | 39/64 | 3 | ~1113 | ~75 s |

## Decoders and scoring

argmax and A flat-field, unchanged; scoring as Table 15, pooled over the five
random runs (C7d + C8a) and over the three Fura-2 runs; confusion matrices;
binomial p against chance 1/4 under independence (stated).

## Predictions

- C8a pooled (5 runs): pixel accuracy >= 55 %: the 64-pixel result stands as
  reported (a probe with signal); >= 70 %: it becomes a partial claim;
  <= 40 %: the two campaign-7 runs were high draws.
- C8b: >= 70 % of the 54 valid pixels in >= 2/3 runs: a 64-pixel biophysical
  image decoded with the high quotient bit; 40–70 %: partial; <= 40 %: no.

## Commands

    bash scripts/run_campaign_8.sh
    .venv/bin/python scripts/decode_bias_corrected.py --campaign8

## Deviation log

(empty)
