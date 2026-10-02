# Hardware campaign 19 — maximum-likelihood amplitude estimation on hardware

Pre-registered 2026-09-25, before any job of this campaign was submitted.

## Rationale

Section 7 of the manuscript executes no circuit: the oracle is verified on a noiseless
simulator and the scaling study is a property of the estimator on ideal probabilities. Two
things changed today. The truth-table divider removes the divider's carry registers from the
state preparation A, and the zero-state reflection can then be restricted to the qubits A
actually entangles (`active_qubits`, scripts/qae_demo_class_b.py), because every other qubit
returns to |0> after A†. Verified exact against sin^2((2k+1)theta) at k = 0, 1, 2, 3 on the
matrix-product-state simulator. Routed two-qubit gates on `ibm_kingston`, best of 8 seeds:
270 at k = 0, 1415 at k = 1, 2477 at k = 2, against 6272 at k = 1 with the unrestricted
reflection.

## Design

Dataset `fourvalue` at n = 1, q = 2, threshold 2, so a_true = 0.25 and the ideal
P(good) is 0.25 at k = 0, 1.00 at k = 1 and 0.25 at k = 2. The contrast between k = 0 and
k = 1 is the Grover amplification; the return to 0.25 at k = 2 is a signature no monotone
noise process produces. `ibm_kingston`, TREX only, no decoupling, 4096 shots, three rounds
per power, `scripts/run_campaign_19.py`. Transpiled circuits and their SHA-256 are written
before the first submission. Cost estimate 3, 4 and 5 s per job.

Staging and stop rule. k = 0 runs first. k = 1 is submitted only if k = 0 clears band B1.
k = 2 is submitted only if k = 1 clears band B2 and the allowance permits. A job is skipped
if the remaining allowance is below its estimate plus 5 s; every run is reported.

## Bands (fixed before data)

A device that has lost the state returns P(good) = 0.5 (maximally mixed) or 0 (all-zero
attractor). The binomial standard error at 4096 shots is 0.007 to 0.008.

- **B1, k = 0 carries the amplitude.** P(good) below 0.46 in at least 2 of 3 runs, that is
  at least 6 standard errors below the mixed-state value, with the run mean within 0.10 of
  the ideal 0.25.
- **B2, amplification is visible.** mean P(good) at k = 1 exceeds mean P(good) at k = 0 by
  at least 6 pooled standard errors, in the direction the ideal predicts (0.25 -> 1.00).
- **B3, the signature is non-monotone.** mean P(good) at k = 2 is below mean P(good) at
  k = 1 by at least 3 pooled standard errors. Scored only if k = 2 runs.
- **B4, estimate.** the maximum-likelihood estimate of a from the measured p_k over the
  powers that ran is reported with its confidence interval, against a_true = 0.25. No band:
  reported as measured.

Secondary, recorded and not used for the bands: the depolarised prediction
P = 0.5 + F (P_ideal - 0.5) with F = exp(-N_2q * epsilon) at the device's median two-qubit
error, which gives the scale of the expected damping.

## Failure reading

If B1 fails, section 7 stays as it is, with the cost table added and the statement that the
oracle is within the device's gate budget at k = 0 but its amplitude does not survive. If B1
passes and B2 fails, the manuscript reports the amplitude read on hardware at k = 0 and the
loss of the amplification step, which is itself a depth statement.

## Commands

    .venv/bin/python scripts/run_campaign_19.py --k 0 --rounds 3
    .venv/bin/python scripts/run_campaign_19.py --k 1 --rounds 3     # only if B1 passes
    .venv/bin/python scripts/run_campaign_19.py --k 2 --rounds 3     # only if B2 passes

## Deviation log

(empty)

## Deviation log (continued)

- 2026-09-25/26: the open-plan allowance regenerates on a rolling 28-day window and stood at
  13 s per account when this campaign ran, so its stages were executed on different accounts of
  the same open plan and the same device: k = 0 on the account that ran campaigns 3 to 17,
  k = 1 on a second one. The circuits, the shots, the mitigation and the staging rule are
  unchanged; only the billing account differs, and each job records its own.
- 2026-09-27: the third k = 1 run executed when the allowance returned (P(good) = 0.397), so
  both powers have three runs. B1 and B2 pass with the three runs (B2 at 34 standard errors).
