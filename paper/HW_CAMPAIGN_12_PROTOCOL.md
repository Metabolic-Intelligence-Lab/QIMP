# Hardware campaign 12: overcoming two declared limits

Written 11 Sep 2026 after campaign 11 and the second referee pass, before any
campaign-12 job. Fixed; deviations logged below.

## 12a. Pulse-count-matched spectator test (`ibm_marrakesh`)

Campaign 11 left open whether the loss inside the padded register comes from
each qubit's own pulses or from its padded neighbours' pulses: padding the six
spectators gave them only 24 pulses in total against 562 on the register.
Condition `dd_spect_matched` puts 12 XX pairs (24 pulses) on EACH spectator,
spread over the circuit duration with explicit delays (167 single-qubit X in
total, no pulses on the active register), same layout and circuit as campaign
11, four runs of 4096 shots.
Predictions: if the active read-out degrades (sigma over null falls below
~9, i.e. by half or more from 17.9, or the separation falls below +0.07),
crosstalk from neighbours' pulses into the register is demonstrated at matched
pulse count. If it stays at 15 sigma or above, the loss under dd_active is
per-qubit pulse error on the padded qubits themselves.

## 12b. Balanced 8 x 8 image (`ibm_kingston`)

The three 8 x 8 targets of campaigns 7–10 are value-imbalanced, and the
flat-field decoder penalises the majority value; all three sit at or below
their constant-read-out null. `balanced8` places each quotient value on
exactly 16 of 64 pixels (seeded permutation, no divide-by-zero), null 16/64,
uniformly controlled load, 987 routed CX on `ibm_kingston` (best of three
seeds), 64/64 on the MPS simulator. Three runs of 65 536 shots.
Predictions: flat-field >= 32/64 in >= 2/3 runs: a 64-pixel image decoded
above its null by a clear margin, and the imbalance confound of the other 8 x 8
targets is confirmed as the cause of their null-level scores. 20–32/64:
signal above null, partial. <= 20/64: 64 pixels at ~1000 CX are beyond this
device with this decoder.

## Decoders and scoring

argmax and A flat-field, unchanged; scoring as Table 13 with the null over
valid pixels; spectator analysis as campaign 11.

## Commands

    .venv/bin/python scripts/run_spectator_experiment.py --condition dd_spect_matched --repeat 4
    .venv/bin/python scripts/run_hardware_class_b_nonrestoring.py --q 2 --n 3 --divider nonrestoring --mitigation trex --load ucry --backend ibm_kingston --shots 65536 --dataset balanced8 --repeat 3 --label c12_b_balanced8_n3_ucry_kingston

## Deviation log

(empty)
