# Processing

Once an image is encoded, processing operations act on its quantum register.
QIMP groups them into four sub-modules.

## Geometric (`qimp.processing.geometric`)

Pure position-qubit manipulations — **encoding-agnostic**: the same `axis_flip`
works on FRQI, NEQR, and QPIE circuits, you just pass a `pos_offset` argument
telling it where the position register starts (0 for FRQI / QPIE, `q` for
NEQR).

| Function | Effect |
|---|---|
| `axis_flip(qc, n, axis, pos_offset=0)` | Flip along X or Y axis |
| `coord_swap(qc, n, pos_offset=0)` | Transpose (swap row ↔ col) |
| `ort_rotation(qc, n, angle, pos_offset=0)` | 90° / 180° / 270° rotation |
| `pos_shift(qc, n, axis, direction, magnitude, pos_offset=0)` | Cyclic shift |
| `restr_flip(qc, n, axis, region_bits, pos_offset=0)` | Flip a sub-region |
| `restr_coord_swap(qc, n, region_bits, pos_offset=0)` | Transpose a sub-region |

```python
from qimp.encoding.neqr import neqr_circuit, neqr_decode
from qimp.processing.geometric import axis_flip

qc = neqr_circuit(image, q=8)
axis_flip(qc, n=2, axis="y", pos_offset=8)   # flip horizontally (X axis)
# ... measure, decode ...
```

End-to-end semantics are verified by encoding with NEQR (exact retrieval),
applying the op, decoding, and comparing against the numpy reference
(`np.fliplr`, `np.transpose`, `np.rot90`, `np.roll`).

## Chromatic (`qimp.processing.chromatic`)

Encoding-specific intensity manipulations.

| Function | Encoding | Effect |
|---|---|---|
| `frqi_color_complement(qc)` | FRQI | θ → π − θ on color qubit |
| `frqi_color_change(qc, theta)` | FRQI | rotate intensities by θ |
| `neqr_color_complement(qc, q)` | NEQR | bit-flip all intensity qubits → ``(2^q − 1) − I`` |
| `neqr_half_intensity(qc, q)` | NEQR | right-shift intensity bits |
| `neqr_classify_complement(qc, q, threshold_bit)` | NEQR | flip bits below the threshold |

## Filters (`qimp.processing.filters`)

Currently exposes the **QHED** (Quantum Hadamard Edge Detection) algorithm,
the showcase application of the thesis: detect edges with only `2n + 1` qubits
using H gates and a cyclic decrement on an auxiliary register.

```python
from qimp.processing.filters import qhed_filter, qhed_decode, qhed_full_edges
from qimp.testing import ideal_simulation

# Horizontal gradients
qc, n, rms = qhed_filter(image)
counts = ideal_simulation(qc, shots=40_000)
edges_h = qhed_decode(counts, n=n, rms=rms)

# Full edge map (horizontal + vertical via transposed image)
qc_v, _, _ = qhed_filter(image.T)
counts_v = ideal_simulation(qc_v, shots=40_000)
edges = qhed_full_edges(image, counts, counts_v)
```

!!! warning "QHED is *not* a pure horizontal-only filter"
    The QHED implementation runs a single cyclic decrement on the combined
    aux+position register. The resulting adjacency is **row-major Z-curve**:
    pixel ``(r, 2^n − 1)`` is considered adjacent to ``(r + 1, 0)``. This
    creates spurious "edges" at the last column of every row. Inspect the
    output carefully — at minimum, mask out the last column before
    visualising; for true row-only gradients, post-process classically.

## Arithmetic (`qimp.processing.arithmetic`) — v0.2

Scaffolded but not yet implemented:

- `qc_add_1`, `q_ADD_SUB` — quantum addition / subtraction on NEQR intensities
- `neqr_comparator`, `neqr_sort` — pixel-wise comparison and sort

Tracking this work in milestone v0.2.

## GP-ratio (`qimp.processing.gp_ratio`)

Application module for the Metabolic-Intelligence Lab's Green-Purple
microscopy pipeline. Not part of the core thesis library, but ships as a
first-class example of how to build a variational quantum image processor on
top of QIMP.

- `classical_gp_image(green, red)` — reference numpy implementation
- `apply_gp_function(qc, n, m, params)` — parameterised quantum sub-circuit
- `combined_objective(mse, psnr, tv, α, β, γ)` — weighted optimisation target
