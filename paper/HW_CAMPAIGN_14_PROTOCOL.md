# Hardware campaign 14: population-inversion test replicated on ibm_kingston

Written 11 Sep 2026 after campaign 13, before any campaign-14 job. Fixed.

Same design as campaigns 11 and 13 on a second device: balanced 2 x 2 Class-B
circuit, non-restoring divider, uniformly controlled load, `ibm_kingston`,
seed 0 layout, six idle physical neighbours measured as spectators, read-out
twirling on, runtime decoupling off. Three conditions, four runs of 4096 shots:

| Condition | XX padding |
|---|---|
| nodd | none |
| dd_active | on the active qubits, spread (spacing 1/4, 1/2, 1/4) |
| dd_active_bunched | on the active qubits, both pulses at the start of each window (0, 0, 1) |

Predictions (fixed): dd_active reproduces a penalty on this device (sigma
over null and separation fall by a factor >= 2 from nodd); dd_active_bunched
recovers at least two thirds of the lost separation. `ibm_kingston` has a
longer median T1 (232 µs against 159 µs), so the spread-padding penalty is
predicted to be smaller than on `ibm_marrakesh` (loss below 78 % of the
separation).

Commands: for c in nodd dd_active dd_active_bunched; do .venv/bin/python scripts/run_spectator_experiment.py --backend ibm_kingston --condition $c --repeat 4; done

Deviation log: (empty)
