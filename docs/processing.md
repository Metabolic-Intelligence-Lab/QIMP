# Processing

Processing functions take a circuit and act on its registers in place. They
also return the circuit, so calls can be chained.

## Geometric (`qimp.processing.geometric`)

Geometric operations act only on the position register, so they work on every
encoding. `pos_offset` gives the first position qubit: 0 for FRQI and QPIE,
`q` for NEQR.

| Function | Effect |
|---|---|
| `axis_flip(qc, n, axis, pos_offset=0)` | flip along the X or Y axis |
| `coord_swap(qc, n, pos_offset=0)` | transpose (swap rows and columns) |
| `ort_rotation(qc, n, angle, pos_offset=0)` | rotation by 90, 180 or 270 degrees |
| `pos_shift(qc, n, axis, direction="+", magnitude=1, pos_offset=0)` | cyclic shift |
| `restr_flip(qc, n, axis, region_bits, pos_offset=0)` | flip of a sub-region |
| `restr_coord_swap(qc, n, region_bits, pos_offset=0)` | transpose of a sub-region |

```python
import numpy as np
from qimp.encoding.neqr import NeqrEncoder
from qimp.processing.geometric import axis_flip
from qimp.testing import exact_counts

image = np.arange(16).reshape(4, 4)
encoder = NeqrEncoder(q=4)
qc = encoder.encode(image)
axis_flip(qc, 2, axis="x", pos_offset=4)
assert (encoder.decode(exact_counts(qc)) == image[::-1, :]).all()
```

The tests check each operation by encoding with NEQR, applying it, decoding
and comparing with the NumPy equivalent (`np.flipud`, `np.fliplr`,
`np.transpose`, `np.rot90`, `np.roll`).

## Chromatic (`qimp.processing.chromatic`)

| Function | Encoding | Effect |
|---|---|---|
| `frqi_color_complement(qc)` | FRQI | angle theta becomes pi/2 - theta |
| `frqi_color_change(qc, theta)` | FRQI | rotation of all intensities by theta |
| `neqr_color_complement(qc, q)` | NEQR | intensity I becomes 2^q - 1 - I |
| `neqr_half_intensity(qc, q)` | NEQR | right shift of the intensity bits |
| `neqr_classify_complement(qc, q, threshold_bit)` | NEQR | flip of the intensity bits below `threshold_bit` |

## Edge detection (`qimp.processing.filters`)

Quantum Hadamard edge detection (QHED) on QPIE states. `qhed_filter` adds one
auxiliary qubit, Hadamard gates and a cyclic decrement, and `qhed_decode`
reads the intensity differences of neighbouring pixels from the counts.
`qhed_full_edges` combines a pass on the image and a pass on its transpose.

```python
from qimp.processing.filters import qhed_filter, qhed_full_edges
from qimp.testing import ideal_simulation

image = np.zeros((8, 8))
image[2:6, 2:6] = 1.0
qc_h, n, rms = qhed_filter(image)
qc_v, _, _ = qhed_filter(image.T)
edges = qhed_full_edges(
    image,
    ideal_simulation(qc_h, shots=40_000, seed=0),
    ideal_simulation(qc_v, shots=40_000, seed=0),
)
```

!!! note "Wrap-around"
    The cyclic decrement runs over the flattened image, so the last pixel of
    each row is compared with the first pixel of the next row. The last
    column of the horizontal pass, and the last row of the vertical pass once
    transposed back, contain these wrap-around values and are usually
    discarded.

## Arithmetic (`qimp.processing.arithmetic`)

Reversible arithmetic on NEQR intensity registers. Registers are lists of
qubit indices, least significant bit first. Every block has an exact inverse,
so it can be uncomputed or used inside a larger algorithm.

| Group | Functions |
|---|---|
| Addition and subtraction | `qc_add_1`, `q_add`, `q_sub`, their inverses (`_inv`), controlled versions (`_ctrl`, `_ctrl_inv`) and `q_add_sub_ctrl` |
| Comparison and sorting | `neqr_comparator`, `neqr_compare_swap`, `neqr_sort` and their inverses |
| Multiplication | `q_mul_const`, `q_mul_const_inv` (by a classical constant) |
| Division | `q_div_restoring`, `q_div_nonrestoring`, `q_div_general`, `q_div_lookup` and their inverses; `q_div` is the default divider |
| Synthesis | `synth_pixel_map`, which builds any classical map between registers from its truth table |

The dividers set a flag qubit on division by zero. The non-restoring divider
gives the same result as the restoring one with fewer two-qubit gates.

## Ratiometric operators (`qimp.processing.ratiometric_circuit`)

Circuits that compute per-pixel ratios of two co-registered NEQR images
inside the circuit, built on the arithmetic above. `dual_neqr_load` loads the
two images on a shared position register.

| Builder | Decoder | Result |
|---|---|---|
| `class_b_ratio` | `decode_class_b_ratio` | integer ratio `I_a // I_b` with a divide-by-zero flag |
| `class_a_gp_prefix`, `class_a_gp_full` | `decode_class_a_prefix`, `decode_class_a_full` | generalised polarisation `(I_a - G I_b) / (I_a + G I_b)` |
| `class_c_rogfp_full` | `decode_class_c_rogfp` | calibrated redox ratio `(R - R_red) / (R_ox - R_red)` |

The Class A builders take the G-factor as `G` (default 1.0), represented in
fixed point as `K / 2^f`; `gp_g_fixed_point` returns `(K, f)`, and a value
that is not dyadic needs an explicit `g_frac_bits`. `class_a_gp_lookup` and
`class_c_rogfp_lookup` build the same maps by truth-table synthesis. `mark_good_oracle` marks the pixels whose quotient
exceeds a threshold, for use in amplitude estimation. `class_b_ratio_inv`,
`dual_neqr_load_inv` and `mark_good_oracle_inv` uncompute the corresponding
blocks.

## GP ratio by parameter synthesis (`qimp.processing.gp_ratio`)

A variational route to the Laurdan GP image on FRQI states:
`classical_gp_image` is the NumPy reference, `analytical_gp_params` gives
closed-form parameters, `apply_gp_function` adds the parametrised
sub-circuit, and `optimize_gp` fits the parameters with a weighted objective
of MSE, PSNR and total variation (`combined_objective`).

## Classification (`qimp.qml.classifier`)

`FrqiClassifier` is a binary variational classifier: a `RealAmplitudes`
ansatz on the FRQI state, read out as the expectation of Z on the colour
qubit, trained with COBYLA on labels -1 and +1.

```python
from qimp.qml.classifier import FrqiClassifier

clf = FrqiClassifier(reps=1, seed=0)
dark, bright = np.zeros((2, 2)), np.ones((2, 2))
clf.fit([dark, bright], [-1, 1], max_iter=20)
clf.predict([dark, bright])
```
