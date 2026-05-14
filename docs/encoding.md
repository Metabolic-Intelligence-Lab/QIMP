# Encodings

Three canonical encodings, three different trade-offs.

| Encoding | Type | Qubits | Retrieval | Complexity |
|---|---|---|---|---|
| **FRQI** | Phase | `2n + 1` | Approximated (shot-noise) | `O(2^{4n})` |
| **NEQR** | Basis | `2n + q` | **Exact** | `O(q·n·2^{2n})` |
| **QPIE** | Amplitude | `2n` | Approximated (shot-noise) | `O(2^n)` |

## FRQI — Flexible Representation of Quantum Images

Pixel intensities map to RY rotation angles applied to a *color* qubit
controlled by the position register. Compact (one extra qubit on top of
position), but retrieval is approximate and intensity precision is
shot-limited.

```python
from qimp.encoding.frqi import FrqiEncoder

encoder = FrqiEncoder()
qc = encoder.encode(image)         # image: (2^n, 2^n)
# ... run, get counts ...
recovered = encoder.decode(counts)[0]
```

Multi-image FRQI: pass a stack of images (`(num_images, H, W)`) and `m =
⌈log₂(num_images)⌉` selection qubits are added automatically. Non-power-of-two
batches are zero-padded.

```python
qc = encoder.encode(np.stack([image_a, image_b]))    # 2n + 1 + 1 qubits
```

## NEQR — Novel Enhanced Quantum Representation

Intensity is stored as a basis state in a `q`-qubit register, conditional on
the position register. This gives **exact** retrieval — useful for arithmetic
operations that need precise pixel values.

```python
from qimp.encoding.neqr import NeqrEncoder

encoder = NeqrEncoder(q=8)         # 8-bit intensity, max value 255
qc = encoder.encode(image)         # image: (2^n, 2^n), integer dtype
recovered = encoder.decode(counts)
assert (recovered == image).all()  # byte-for-byte recovery
```

The cost is the extra `q` qubits and a higher per-pixel gate count.

## QPIE — Quantum Probability Image Encoding

Pixel intensities are stored directly as amplitude probabilities:
`|Ψ⟩ = Σ (I_i / ‖I‖) |i⟩`. The most compact encoding (only `2n` qubits, no
color register), at the cost of an arbitrary state preparation that transpiles
to deep circuits.

```python
from qimp.encoding.qpie import QpieEncoder

encoder = QpieEncoder()
qc = encoder.encode(image.astype(float))
recovered = encoder.decode(counts)
```

Note that QPIE preserves intensity *proportions*, not absolute scale. The
decoder uses the RMS computed at encoding time to rescale back; you only get
proportional reconstruction otherwise.

## Choosing between them

- Need **exact pixel values** for arithmetic / sorting? → NEQR.
- Need **compact circuits** for shallow-depth experiments on near-term
  hardware? → QPIE.
- Need the **classical interface point** for variational pipelines (e.g.
  QHED-style algorithms)? → QPIE (QHED is QPIE-only).
- Don't have multi-controlled X gates with auxiliary qubits but accept
  amplitude encoding of intensity? → FRQI.

## RGB extensions (v0.2)

`MCRQI` (FRQI for RGB) and `NCQI` (NEQR for RGB) are scaffolded but not yet
implemented. Track [issue #N](https://github.com/Metabolic-Intelligence-Lab/QIMP/issues)
for progress.
