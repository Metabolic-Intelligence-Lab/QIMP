# Encodings

All encoders share one interface: `encode(image)` returns a `QuantumCircuit`
and `decode(counts)` returns the image reconstructed from measurement counts.
Image side `2^n` and, where relevant, bit depth `q` are taken from the input
or passed explicitly.

| Encoding | Class | Image | Qubits | Retrieval |
|---|---|---|---|---|
| FRQI | `FrqiEncoder` | grey-scale | `2n + 1` | statistical (shot noise) |
| NEQR | `NeqrEncoder` | grey-scale | `2n + q` | exact |
| QPIE | `QpieEncoder` | grey-scale | `2n` | statistical (shot noise) |
| MCRQI | `McrqiEncoder` | RGB | `2n + 3` | statistical (shot noise) |
| NCQI | `NcqiEncoder` | RGB | `2n + 3q` | exact |

## FRQI

Flexible representation of quantum images. Each pixel intensity sets the angle
of a rotation of one colour qubit, controlled by the position register. The
circuit is compact in qubits, and the intensity is recovered from measurement
statistics.

```python
import numpy as np
from qimp.encoding.frqi import FrqiEncoder
from qimp.testing import ideal_simulation

image = np.random.default_rng(0).integers(0, 256, (4, 4)).astype(np.uint8)
encoder = FrqiEncoder()
qc = encoder.encode(image)
recovered = encoder.decode(ideal_simulation(qc, shots=40_000, seed=0))
```

A stack of images of shape `(num_images, H, W)` adds `m = ceil(log2(num_images))`
selection qubits, and `decode` returns a list of images. Stacks whose length is
not a power of two are zero-padded.

## NEQR

Novel enhanced quantum representation. The intensity of each pixel is written
as a basis state of a `q`-qubit register, conditioned on the position register.
Retrieval is exact, which is what the arithmetic operations need, at the cost
of `q` extra qubits and more gates per pixel.

```python
from qimp.encoding.neqr import NeqrEncoder
from qimp.testing import exact_counts

encoder = NeqrEncoder(q=8)
qc = encoder.encode(image)
assert (encoder.decode(exact_counts(qc)) == image).all()
```

## QPIE

Quantum probability image encoding. The normalised intensities are the
amplitudes of the position register, `|psi> = sum_i (I_i / ||I||) |i>`. It uses
the fewest qubits (`2n`), but the state preparation transpiles to deep
circuits. The decoder rescales with the norm computed at encoding time.

```python
from qimp.encoding.qpie import QpieEncoder

encoder = QpieEncoder()
qc = encoder.encode(image.astype(float))
recovered = encoder.decode(ideal_simulation(qc, shots=80_000, seed=0))
```

QHED edge detection works on QPIE states.

## MCRQI and NCQI

RGB extensions of FRQI and NEQR. MCRQI uses one colour qubit per channel
(`2n + 3` qubits); NCQI uses one `q`-qubit intensity register per channel
(`2n + 3q` qubits) and retrieves the image exactly. Both take an array of
shape `(H, W, 3)`.

```python
from qimp.encoding.ncqi import NcqiEncoder

rgb = np.random.default_rng(1).integers(0, 4, (2, 2, 3))
encoder = NcqiEncoder(q=2)
qc = encoder.encode(rgb)
assert (encoder.decode(exact_counts(qc)) == rgb).all()
```

## Compression

`qimp.encoding.compression` reduces the multi-controlled gates of FRQI and
NEQR. The pixels that share an intensity (FRQI) or an intensity bit (NEQR) form
a Boolean function of the position bits; minimising that function gives fewer,
partially controlled gates. `FrqiCompressor` and `NeqrCompressor` build the
compressed circuits and produce the same state as the uncompressed encoders.
The cover of the function must be disjoint, because the gates act by angle sum
(FRQI) or XOR (NEQR). Two minimisers are available through the `method`
argument: exact Quine-McCluskey (`"qm"`) and the ESPRESSO heuristic
(`"espresso"`), which scales to larger images. The default, `"auto"`, uses
Quine-McCluskey up to `n = 4` and ESPRESSO above.

```python
from qimp.encoding.compression import NeqrCompressor

compressor = NeqrCompressor(image, q=8, method="auto")
qc = compressor.neqr_image()        # same state as NeqrEncoder(q=8).encode(image)
```

## Qubit layout

All encodings use the same position register:

| Qubits | Role |
|---|---|
| `0 .. n-1` | column index, least significant bit first |
| `n .. 2n-1` | row index, least significant bit first |

FRQI puts its colour qubit at `2n`. NEQR puts the intensity register on qubits
`0 .. q-1` and the position register above it, so geometric operations on NEQR
circuits take `pos_offset=q`. QPIE has no extra qubits.

In a Qiskit count string, which is written most significant bit first, the
leftmost position bit is the high bit of the row index and the rightmost is the
low bit of the column index.

## Choosing an encoding

- Exact pixel values for arithmetic, comparison or sorting: NEQR (or NCQI for
  RGB).
- Fewest qubits: QPIE, which is also the input of QHED.
- One extra qubit and a rotation per pixel: FRQI (or MCRQI for RGB).
