"""FRQI encoding: Flexible Representation of Quantum Images.

A 2^n × 2^n grayscale image is encoded into 2n+1 qubits: 2n position qubits
plus 1 color qubit. The pixel intensity is mapped to an RY rotation angle and
applied via a fully-controlled RY gate. Multiple images can be stacked by
adding m = ceil(log2(num_images)) selection qubits, for a total of 2n+m+1 qubits.

Scales freely with n (image side = 2^n) and m (number of stacked images = 2^m).

Reference: docs/tesi.pdf §3.1.1, §2.2.1
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate

__all__ = [
    "FrqiEncoder",
    "angles_to_intensities",
    "frqi_circuit",
    "frqi_decode",
    "intensities_to_angles",
]


def intensities_to_angles(intensities: np.ndarray, normalization: float) -> np.ndarray:
    """Map pixel intensities to RY angles in [0, π].

    Convention: intensity 0 → θ = 0, intensity = normalization → θ = π.
    After RY(θ), prob(color qubit = 1) = sin²(θ/2) = intensity / normalization.
    """
    if normalization <= 0:
        raise ValueError(f"normalization must be > 0, got {normalization}")
    x = np.clip(1.0 - 2.0 * intensities.astype(np.float64) / normalization, -1.0, 1.0)
    return np.arccos(x)


def angles_to_intensities(angles: np.ndarray, normalization: float) -> np.ndarray:
    """Inverse of `intensities_to_angles`."""
    return np.sin(angles / 2.0) ** 2 * normalization


def _validate_image_shape(image: np.ndarray) -> int:
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError(f"image must be square 2D, got shape {image.shape}")
    side = image.shape[0]
    if side < 2 or (side & (side - 1)) != 0:
        raise ValueError(f"image side must be a power of two ≥ 2, got {side}")
    return int(math.log2(side))


def _stack_images(images: np.ndarray | Sequence[np.ndarray]) -> tuple[np.ndarray, int, int]:
    """Normalize the input to a (num_images, side, side) array.

    Returns (stack, n, m) where m = ceil(log2(num_images)) and m = 0 if num_images == 1.
    """
    if isinstance(images, np.ndarray) and images.ndim == 2:
        arr = images[np.newaxis, ...]
    elif isinstance(images, np.ndarray) and images.ndim == 3:
        arr = images
    else:
        seq = list(images)
        if not seq:
            raise ValueError("images list must be non-empty")
        arr = np.stack([np.asarray(img) for img in seq], axis=0)
    n = _validate_image_shape(arr[0])
    for i in range(1, arr.shape[0]):
        if arr[i].shape != arr[0].shape:
            raise ValueError(f"image {i} has shape {arr[i].shape}, expected {arr[0].shape}")
    num_images = arr.shape[0]
    m = 0 if num_images == 1 else math.ceil(math.log2(num_images))
    if num_images != (1 << m) and num_images != 1:
        # Pad with zero images to reach 2^m
        pad = (1 << m) - num_images
        arr = np.concatenate([arr, np.zeros((pad, *arr.shape[1:]), dtype=arr.dtype)], axis=0)
    return arr, n, m


def frqi_circuit(
    images: np.ndarray | Sequence[np.ndarray],
    *,
    normalization: float | None = None,
) -> QuantumCircuit:
    """Build an FRQI quantum circuit for one or more square images.

    Parameters
    ----------
    images
        A single 2D image (shape `(2^n, 2^n)`), a 3D stack
        (shape `(num_images, 2^n, 2^n)`), or a sequence of 2D images. When more
        than one image is supplied, `num_images` is padded with zero images to
        the next power of two and `m = log2(num_images_padded)` selection qubits
        are added.
    normalization
        Intensity that maps to angle π (full color amplitude on |1⟩). Defaults to
        the max value of the dtype (255 for uint8, 65535 for uint16, 1.0 for float).

    Returns
    -------
    QuantumCircuit
        Circuit with `2n + m + 1` qubits and an equal number of classical bits.
        No measurement gates are appended; call `ideal_simulation` / `device_test`
        on the returned circuit (they add measurements automatically).
    """
    stack, n, m = _stack_images(images)
    if normalization is None:
        normalization = (
            float(np.iinfo(stack.dtype).max) if np.issubdtype(stack.dtype, np.integer) else 1.0
        )
    total_qubits = 2 * n + m + 1
    # No classical bits: the caller (or qimp.testing._ensure_measured) appends
    # measurement when needed, avoiding a duplicate classical register.
    qc = QuantumCircuit(total_qubits)

    # Superposition over position (and selection) qubits.
    qc.h(range(total_qubits - 1))
    qc.barrier()

    num_controls = 2 * n + m
    control_target = list(range(total_qubits))

    for s in range(stack.shape[0]):
        sel_bits = format(s, f"0{m}b") if m > 0 else ""
        angles = intensities_to_angles(stack[s].flatten(), normalization)

        for pixel_idx, theta in enumerate(angles):
            # Position bits LSB-first → qubits 0..2n-1.
            pos_bits = format(pixel_idx, f"0{2 * n}b")[::-1]

            # Flip controls that should be |0⟩ so the multi-controlled gate fires
            # only on this pixel of this selection.
            x_targets: list[int] = []
            for q, bit in enumerate(pos_bits):
                if bit == "0":
                    x_targets.append(q)
            for q, bit in enumerate(sel_bits):
                # sel_bits is MSB-first; qubit index is 2n + (m-1-q).
                if bit == "0":
                    x_targets.append(2 * n + (m - 1 - q))

            for q in x_targets:
                qc.x(q)
            controlled_ry = RYGate(float(theta)).control(num_controls, annotated=True)
            qc.append(controlled_ry, control_target)
            for q in x_targets:
                qc.x(q)
            qc.barrier()

    return qc


def _decode_state_string(state: str, n: int, m: int) -> tuple[int, int, int]:
    """Return (image_index, pixel_index_in_image, color_bit) from a Qiskit count key.

    Qiskit count strings are space-padded for separate classical registers and
    high-qubit-first within a register. The encoder uses a single n-bit classical
    register, so we strip whitespace and read left-to-right as q[N-1]…q[0].
    """
    flat = state.replace(" ", "")
    color_bit = int(flat[0])
    # Position bits high-to-low for qubits 2n+m-1 … 2n correspond to selection,
    # then 2n-1 … 0 correspond to position.
    sel_str = flat[1 : 1 + m] if m > 0 else ""
    pos_str = flat[1 + m :]
    image_index = int(sel_str, 2) if sel_str else 0
    # pos_str is high-to-low: qubit (2n-1) … qubit 0. During encoding, qubit q
    # carries the bit-q of `pixel_idx`. Reading high-to-low therefore matches
    # the canonical big-endian integer representation of `pixel_idx` directly.
    pixel_idx = int(pos_str, 2) if pos_str else 0
    return image_index, pixel_idx, color_bit


def frqi_decode(
    counts: dict[str, int],
    n: int,
    m: int = 0,
    normalization: float = 1.0,
) -> list[np.ndarray]:
    """Decode measurement counts from an FRQI circuit into images.

    Parameters
    ----------
    counts
        Qiskit-style histogram from `ideal_simulation(qc, shots)`.
    n
        Number of spatial qubits per axis (image side = 2^n).
    m
        Number of selection qubits (number of images = 2^m). Defaults to 0
        (single image).
    normalization
        Intensity that maps to θ = π — must match what was used at encoding.
        The reconstructed intensity for pixel j of image s is
        `prob(color=1 | pos=j, sel=s) * normalization`.

    Returns
    -------
    list[np.ndarray]
        One `(2^n, 2^n)` float array per image. Empty buckets are filled with 0.
    """
    if n < 1 or m < 0:
        raise ValueError("n must be >= 1 and m >= 0")
    num_images = 1 << m
    side = 1 << n
    total_per_image = 1 << (2 * n)
    sum_p1 = np.zeros((num_images, total_per_image), dtype=np.float64)
    sum_p_total = np.zeros((num_images, total_per_image), dtype=np.float64)

    for state, count in counts.items():
        image_idx, pixel_idx, color_bit = _decode_state_string(state, n, m)
        sum_p_total[image_idx, pixel_idx] += count
        if color_bit == 1:
            sum_p1[image_idx, pixel_idx] += count

    with np.errstate(invalid="ignore", divide="ignore"):
        intensities = np.where(
            sum_p_total > 0,
            sum_p1 / sum_p_total * normalization,
            0.0,
        )

    images = []
    for s in range(num_images):
        # Pixel index → (row, col) using the same MSB-row / LSB-col mapping as encode.
        img = np.zeros((side, side), dtype=np.float64)
        for pixel_idx in range(total_per_image):
            bits = format(pixel_idx, f"0{2 * n}b")  # MSB-first
            row = int(bits[:n], 2)
            col = int(bits[n:], 2)
            img[row, col] = intensities[s, pixel_idx]
        images.append(img)
    return images


class FrqiEncoder:
    """Stateful FRQI encoder.

    Bundles the normalization factor and inferred `n`/`m` so the same instance
    can be reused for encode→measure→decode round-trips.
    """

    def __init__(self, *, normalization: float | None = None) -> None:
        self._normalization = normalization
        self.n: int | None = None
        self.m: int | None = None

    @property
    def normalization(self) -> float:
        if self._normalization is None:
            raise RuntimeError("normalization is set only after encode() is called")
        return self._normalization

    def encode(self, images: np.ndarray | Sequence[np.ndarray]) -> QuantumCircuit:
        stack, n, m = _stack_images(images)
        if self._normalization is None:
            self._normalization = (
                float(np.iinfo(stack.dtype).max) if np.issubdtype(stack.dtype, np.integer) else 1.0
            )
        self.n = n
        self.m = m
        return frqi_circuit(images, normalization=self._normalization)

    def decode(self, counts: dict[str, int]) -> list[np.ndarray]:
        if self.n is None or self._normalization is None:
            raise RuntimeError("call encode() before decode()")
        m = self.m if self.m is not None else 0
        return frqi_decode(counts, n=self.n, m=m, normalization=self._normalization)
