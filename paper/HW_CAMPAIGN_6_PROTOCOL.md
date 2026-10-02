# Hardware campaign 6: pre-registered protocol

Written 10 Sep 2026 after campaign 5 was scored and before any campaign-6 job
was submitted. Fixed; deviations logged at the bottom.

## Purpose

Two objections to the v2 hardware section, and the experiment for each.

1. "It is a 2 x 2 image." Scale in pixels (4 x 4, n = 2) is closed at this
   depth: campaign 5 left 34 % accuracy at 1650 routed CX, and ancilla-assisted
   load synthesis recovers only 3 % (1516 vs 1557 CX, measured). Scale in
   **intensity width** is open: q = 3 gives 3-bit quotients, eight possible
   values, at 1272 routed CX on `ibm_kingston` (36 qubits), between the 650 CX
   where the decoder is exact and the 1650 CX where signal is marginal.
   Target `fourvalue_q3`: I_a = [[2,7],[5,7]], I_b = [[3,2],[1,1]],
   R = [[0,3],[5,7]], four distinct 3-bit quotients, null 1/4, chance 1/8.
2. "The decoder works only on `ibm_kingston`." On `ibm_marrakesh` (70 % of
   pixels) the residual is position-dependent, and the per-position
   calibration of campaign 4 failed because an offset estimated on one known
   target carries that target's own true-bin deficit. The fix is a
   **cyclic calibration**: four targets in which every pixel takes each
   quotient value exactly once (`fourvalue`, `_p2`, `_p3`, `_p4`, R patterns
   [[0,1],[2,3]], [[1,2],[3,0]], [[2,3],[0,1]], [[3,0],[1,2]]). Under the
   model P = (1-f) onehot(v) + f u + o_p, the mean over the four targets of
   (P - onehot(v)) equals o_p exactly, content-independent. The offsets are
   estimated on `ibm_marrakesh` and validated on a DIFFERENT target on the same
   device (the 20 balanced runs of campaigns 2–3, and the 5 archived Fura-2
   runs), never on the calibration targets.

## Configurations (all TREX only, non-restoring, optimization_level=3)

| Id | Dataset | n, q | Shots | Repeats | Backend | Expected QPU |
|---|---|---|---:|---:|---|---:|
| C6a | `fourvalue_q3` | 1, 3 | 16 384 | 4 | ibm_kingston | ~40 s |
| C6b | `fourvalue_p2`, `_p3`, `_p4` | 1, 2 | 16 384 | 3 each | ibm_marrakesh | ~65 s |

`fourvalue` on `ibm_marrakesh` at 16 384 shots (campaign 3, C2, five runs) is
the fourth calibration target and is not re-run.

## Decoders

- argmax and A flat-field, unchanged.
- **C', cyclic per-position offset**: o_p = mean over the four calibration
  targets (all their runs) of (P_p - onehot(true_p)), then zero-meaned over
  bins; decode = argmax(P - o). Applied only to held-out targets.
- For C6a (8 bins) argmax and flat-field only.

## Scoring

As Table 13. For C6a: 8 x 8 confusion, pixel accuracy against chance 1/8 and
match against the 1/4 null, residual on the true bin. For C': match and pixel
accuracy on the 20 balanced `ibm_marrakesh` runs and the 5 Fura-2 runs, beside
argmax and flat-field on the same runs; the estimated o_p is reported.

## Predictions

- C6a: flat-field 4/4 in >= 3/4 runs: 3-bit quotients decoded, goes into the
  hardware claim ("over the whole 2-bit codomain and four distinct 3-bit
  values"). Accuracy 40–75 %: partial, reported. <= 3/8: no signal at 1272 CX
  on this device.
- C': balanced `ibm_marrakesh` mean match >= 3.5/4 (from 2.4 flat-field):
  the residual was a stable per-position offset and is now calibrated out;
  the device-dependence objection is answered. 2.5–3.5: partial. <= 2.5: the
  offset drifts between jobs faster than a calibration survives, and the
  paper says per-position calibration must be in-run.

## Commands

    bash scripts/run_campaign_6.sh
    .venv/bin/python scripts/decode_bias_corrected.py --campaign6

## Deviation log

- 10 Sep 2026: after C6a (4/4 runs archived) and the first two calibration targets (3/3 runs each), a DNS outage on the client machine killed the launcher while it was polling the first `fourvalue_p4` job (`dahc78b9k43c73aejv00`, submitted, result not persisted). That job is discarded; `fourvalue_p4` was relaunched for three fresh runs with identical settings. No change to targets, decoders or scoring.

## Addendum, 10 Sep 2026, written after C' was scored and BEFORE decoder D was scored

C' (additive per-position offset) scored 0.6/4 on the balanced runs, 2.2/4 on
Fura-2, and 0/4 in leave-one-target-out among the calibration targets: the
decoded value is wrong in a structured way (pixels with true 0 read 1 or 2),
so the read-out error is value-dependent, not an additive offset. The cyclic
targets are exactly the data to estimate the full per-pixel, per-value
response.

**Decoder D, per-pixel template (read-out calibration on the quotient
register).** For pixel p and value v, the template T_p[v,:] is the mean
quotient histogram over all calibration runs of the target whose true value
at p is v (each v occurs in exactly one of the four targets). A held-out pixel
with observed histogram h is decoded as
    v* = argmax_v  sum_b h_b log T_p[v,b]
(multinomial log-likelihood against the templates). Validation: the same
held-out targets as C' (20 balanced runs, 5 Fura-2 runs on `ibm_marrakesh`),
never the calibration targets; and leave-one-target-out among the calibration
targets as a secondary check (biased, since the left-out value's template at
each pixel is then missing: reported only as such).

Prediction, fixed now: balanced >= 3.5/4 mean answers the device objection
with a standard read-out calibration; 2.5–3.5 partial; <= 2.5 the per-pixel
response drifts between jobs and in-run calibration is required.
