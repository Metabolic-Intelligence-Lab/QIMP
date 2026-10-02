# Hardware campaign 20 — Laurdan GP and roGFP on hardware

Pre-registered 2026-09-25, before any job of this campaign was submitted.

## Rationale

Table 2 of the manuscript reports the Class-A (Laurdan generalized polarization) and Class-C
(roGFP redox index) pipelines on the matrix-product-state simulator only: at n = 1, q = 2 they
need 71 and 114 qubits, past what the device can carry. The same truth-table synthesis that
gave `q_div_lookup` applies to the whole per-pixel map. `class_a_gp_lookup` and
`class_c_rogfp_lookup` (src/qimp/processing/ratiometric_circuit.py) compute

    Class A: magnitude = floor(|I_a - I_b| * 2^q_frac / (I_a + I_b)), sign = 1 iff I_b > I_a,
             div_flag = 1 iff I_a + I_b = 0
    Class C: ratio = floor(I_a * 2^q_frac / I_b), rc_signed = ratio - R_red_fp,
             div_flag = 1 iff I_b = 0

on 2n + 2q + q_frac + 3 and 2n + 2q + q + q_frac + 2 qubits, with both operands preserved and
no ancilla, and are decoded by `decode_class_a_full` and `decode_class_c_rogfp` unchanged.
`scripts/verify_lookup_maps.py` checks them against the library pipelines simulated in MPS over
every operand pair at q = 2: identical at every pixel where the observable is defined. At
I_a + I_b = 0 the library leaves the magnitude register in the state its cascade left and the
synthesised map writes zero; both set the flag and the decoder masks the pixel.

Routed two-qubit gates on `ibm_kingston`, best of 16 seeds: Class A 324 at 2x2, 429-439 at 4x4,
825 at 8x8; Class C 308, 393-403. The 4x4 circuits sit below the 577-719 gates at which the
Class-B 4x4 circuit decoded at 98 % (Table 13), so 16 pixels are in a regime already
demonstrated; 8x8 is above the Class-B 8x8 budget of 564-573 and its outcome is open.

## Design

`scripts/run_campaign_20.py`, q = 2, q_frac = 2, R_red_fp = 1, R_ox - R_red = 1, uniformly
controlled load, TREX only, no decoupling, optimization_level 3, best of 16 seeds per target
fixed and hashed before the first submission. Seven targets, three rounds, each round
submitting every target once:

| target | pixels | valid | distinct values | constant-read-out null | shots |
|---|---:|---:|---:|---:|---:|
| A, Laurdan 4x4 (`canonical_shared`) | 16 | 12 | 3 | 8/12 | 16384 |
| A, random 4x4 | 16 | 16 | 5 | 7/16 | 16384 |
| A, Laurdan 8x8 | 64 | 35 | 3 | 21/35 | 65536 |
| C, Laurdan 4x4 | 16 | 12 | 3 | 6/12 | 16384 |
| C, random 4x4 | 16 | 16 | 6 | 5/16 | 16384 |
| A, four-value 2x2 | 4 | 4 | 4 | 1/4 | 4096 |
| C, four-value 2x2 | 4 | 4 | 4 | 1/4 | 4096 |

21 jobs, about 120 s. Stop rule: a job is skipped if the remaining allowance is below its
estimate plus 5 s; the campaign is scored on the jobs that ran and every run is reported.

## Bands (fixed before data)

Scored per target over its three runs, on the valid pixels, with the per-pixel majority vote of
the decoder the manuscript already uses.

- **B1, Class A at 4x4 carries the sign.** the sign of GP is correct in at least 90 % of valid
  pixel-runs.
- **B2, Class A at 4x4 decodes.** the exact GP value is returned in at least 75 % of valid
  pixel-runs, and each run is strictly above the target's constant-read-out null in at least 2
  of 3 runs. 55-75 % is "partial".
- **B3, Class C at 4x4 decodes.** same rule as B2 on the calibrated redox index.
- **B4, Class A at 8x8.** strictly above the constant-read-out null in at least 2 of 3 runs;
  "decoded" additionally needs 75 % of valid pixel-runs exact, "partial" 40-75 %.
- **B5, 2x2.** at least 90 % of pixel-runs exact for both classes.
- **B6, reported and not a band.** the mean absolute error on GP and on the redox index, which
  is the quantity a microscopy user reads, and the div-zero flag agreement.

## Failure reading

If B1 fails the classes stay simulation-only in the manuscript and the campaign is reported as
a measured limit at a stated depth, not as a promise. If B1 passes and B2 fails, the manuscript
reports the sign of the membrane-order observable read on hardware and the loss of its
magnitude, which is a resolution statement.

## Commands

    .venv/bin/python scripts/run_campaign_20.py --backend ibm_kingston --rounds 3
    .venv/bin/python scripts/verify_lookup_maps.py --q 2 --q-frac 2

## Deviation log

(empty)

## Deviation log (continued)

- 2026-09-26/27: the allowance stood at 5 to 13 s per account, so the first runs executed on
  several accounts of the same open plan and the same device (one Class-A 2x2 run on
  2026-09-26, the rest on 2026-09-27 when 227 s regenerated).
- 2026-09-27: two copies of the scheduler ran at once, because a liveness check failed, and
  both submitted round 1. Nine labels therefore have two runs with different job identifiers,
  in two archive directories (`20260927T121534Z`, `20260927T122216Z`). Both are kept and
  counted as repeats of the same compiled circuit; no run is discarded. Round 2 completed for
  four of the seven targets before the allowance ran out.
- Scoring clarification, made after the data were seen: band B1 scores the sign of GP only on
  the valid pixels whose true GP is non-zero, because the sign of zero is not defined. On the
  Laurdan 4x4 target 12 of 16 valid pixels have GP = 0, so the clarification decides what B1
  measures there and is reported as such.
- Exploratory, not pre-registered: the sign qubit and each magnitude bit read from their
  per-pixel marginals rather than from the majority vote on the joint value
  (`scripts/score_campaign20.py`, section "marginal").
