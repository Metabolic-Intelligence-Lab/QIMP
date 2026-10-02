# Hardware campaign 18 — 256 pixels

Pre-registered 2026-09-25, before any job of this campaign was submitted.

## Rationale

With the truth-table divider the Class-B circuit routes to 564-573 two-qubit gates at 64
pixels, and the first campaign-17 run decoded the random 8x8 target at 84 per cent of its 64
pixels (flat-field, against a constant-read-out null of 35/64). At 256 pixels the load is what
decides: the uniformly controlled load is 2q*2^(2n) CNOTs before compilation, but its angle
spectrum is sparse when the image is block-constant, and the compiler exploits that. Measured
routed two-qubit gates on `ibm_kingston` at 16x16 with the truth-table divider: 552 for a
block-constant target, 114 for a tiled one, 1740 for the real Laurdan frame and 2106 for a
random image. Only the first two are within the budget that decodes.

The library also carries an explicit QPIXL-style compression of the load
(`dual_neqr_load(..., compress=True)`, exact against the uncompressed form on statevector).
It reduces the logical CNOT count (64 to 14 on a tiled 4x4 plane) and changes the routed count
by nothing at optimisation level 3, because the compiler already performs those cancellations.
This is recorded as a measurement, not used as a lever.

## Design

`scripts/run_campaign_18.py`, divider `lookup`, uniformly controlled load, q = 2, TREX only,
no decoupling, best of 16 transpiler seeds per target fixed and hashed before submission.
`ibm_kingston`, three rounds, each round submitting every target once, 65 536 shots.

| target | pixels | distinct operand pairs | quotient values | constant-read-out null | routed 2q |
|---|---:|---:|---:|---:|---:|
| `blocks` 16x16 (claim) | 256 | 11 | 0,1,2,3 balanced | 64/256 | 552 |
| `tiled` 16x16 (control) | 256 | 4 | 0,1,2,3 balanced | 64/256 | 114 |
| `blocks` 8x8 (bridge) | 64 | 7 | 0,1,2,3 balanced | 16/64 | 223 |

9 jobs, about 200 s. Stop rule: a job is skipped if the remaining allowance is below its
estimate plus 5 s; every run that ran is reported. The campaign is not launched at all if
campaign 17 fails both of its bands.

The tiled target is the same four division problems replicated 64 times: it bounds what image
size alone buys and is never the claim. The block-constant target carries 11 distinct operand
pairs over 64 independent blocks and is the claim-bearing one. The real 16x16 Laurdan frame
is out of budget at 1740 gates and is not attempted.

## Bands (fixed before data)

- **B1, 256 pixels decode.** on `blocks` 16x16, pooled flat-field accuracy over three runs at
  least 75 per cent of the 256 pixels, and each run strictly above the constant-read-out null
  of 64/256 in at least 2 of 3 runs. 40-75 per cent is "partial".
- **B2, decoder-free.** on the same target, the number of pixels whose most frequent
  (q, r, d) triple is the true one exceeds the run's constant-read-out null, and the identity
  excess over the cross-pixel null on the pixels with q >= 1 is positive, in at least 2 of 3
  runs.
- **B3, the bridge.** `blocks` 8x8 at least 85 per cent flat-field pooled, which the 8x8
  results of campaign 17 make the expected value at 223 gates.
- **B4, reported and not a band.** the tiled control, with its four distinct operand pairs
  stated wherever it appears.

Shots may be raised to 131 072 on the claim target in a second pass if its per-pixel statistics
are the binding limit (256 shots per pixel at 65 536); such a pass is a deviation and logged.

## Commands

    .venv/bin/python scripts/run_campaign_18.py --backend ibm_kingston --rounds 3 --phase main
    .venv/bin/python scripts/arithmetic_identity.py --prefix c18_ --json paper/data_autonomous/arithmetic_identity_c18.json

## Deviation log

(empty)

## Deviation log (continued)

- 2026-10-02: the allowance had not returned when the manuscript was submitted, so the campaign
  was executed on a noise model calibrated on the device instead
  (`scripts/calibrate_noise_model.py c18`). It uses the compilations fixed for this protocol in
  `data/output/ibm_hw/20260925T153234Z/transpiled_c18`, whose SHA-256 is checked against their
  manifest before each run, the protocol's 65 536 shots, three runs per target and the bands
  above, unchanged. The model is `NoiseModel.from_backend(ibm_kingston)` at the calibration of
  2 October 2026 (11:08 CEST), frozen to `data/output/noise_sim/kingston_noise_model_snapshot.pkl`,
  with the couplers and the qubit reported broken that day given the device's median values,
  plus a two-qubit depolarizing channel composed with every CZ whose strength was fitted on the
  four 8x8 targets of campaign 17, on the same frozen model, before this run. The simulated runs are archived
  under `data/output/noise_sim/c18/` with `simulated` set in their metadata. They are
  projections, not hardware results, and the device campaign remains to be run.
- The fingerprint in `paper/data_autonomous/campaign18_protocol.sha256` is that of this file up to
  the line "(empty)" above, as pre-registered on 2026-09-25.
