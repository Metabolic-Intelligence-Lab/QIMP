"""FRQI / NEQR compression via Boolean-function minimisation.

Pixels that share the same intensity (FRQI) or the same intensity-bit (NEQR)
form a Boolean function of the position bits. Minimising that function with
Quine–McCluskey shrinks the number of control patterns needed, which reduces
the per-pixel multi-controlled gates issued by the encoder.

This implementation ships a small self-contained Quine–McCluskey simplifier
so the optional `pyeda` dependency stays optional. For very large fan-ins
(``n ≥ 5``, i.e. ``2n ≥ 10``), prefer the heuristic ESPRESSO algorithm via
``pyeda.boolalg.minimization.espresso_exprs``.

Reference: docs/tesi.pdf §2.5.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate

from qimp.encoding.frqi import intensities_to_angles

__all__ = [
    "FrqiCompressor",
    "NeqrCompressor",
    "compress_minterms",
    "position_strings_to_implicants",
]


# ---------------------------------------------------------- Quine–McCluskey ----


def _count_ones(s: str) -> int:
    return s.count("1")


def _can_combine(a: str, b: str) -> tuple[bool, str]:
    """Return (can_combine, merged) — two implicants combine iff they share the
    same ``'-'`` positions and differ in exactly one explicit bit.
    """
    if len(a) != len(b):
        return False, ""
    diffs = 0
    merged_chars: list[str] = []
    for ca, cb in zip(a, b, strict=True):
        if ca == "-" or cb == "-":
            # Both must be don't-care at this position to combine.
            if ca != cb:
                return False, ""
            merged_chars.append("-")
        elif ca != cb:
            diffs += 1
            if diffs > 1:
                return False, ""
            merged_chars.append("-")
        else:
            merged_chars.append(ca)
    return (diffs == 1, "".join(merged_chars))


def compress_minterms(minterms: Iterable[str]) -> list[str]:
    """Return prime implicants for a Boolean function given by `minterms`.

    Each input is a fixed-width binary string. Output strings use ``"-"`` for
    don't-cares. The set returned covers every minterm in `minterms` (no
    further essential-prime-implicant selection — the result is a *cover*, not
    necessarily the smallest one).

    For typical FRQI/NEQR uses (n ≤ 5) this is fast and correct. For larger
    fan-ins use ESPRESSO.
    """
    current = set(minterms)
    if not current:
        return []
    width = len(next(iter(current)))
    primes: set[str] = set()

    while True:
        by_ones: dict[int, list[str]] = defaultdict(list)
        for term in current:
            by_ones[_count_ones(term.replace("-", "0"))].append(term)

        next_round: set[str] = set()
        used: set[str] = set()
        for k in sorted(by_ones.keys()):
            if k + 1 not in by_ones:
                continue
            for term_a in by_ones[k]:
                for term_b in by_ones[k + 1]:
                    ok, merged = _can_combine(term_a, term_b)
                    if ok:
                        next_round.add(merged)
                        used.add(term_a)
                        used.add(term_b)
        # Anything that wasn't combined this round is a prime implicant.
        primes |= current - used
        if not next_round:
            break
        current = next_round

    # Sanity: every cover string has the right width.
    assert all(len(p) == width for p in primes)
    return sorted(primes)


def _implicant_matches(minterm: str, implicant: str) -> bool:
    """True iff every explicit bit of `implicant` matches the corresponding bit
    of `minterm`. ``-`` is a wildcard.
    """
    return all(ic == "-" or ic == mc for mc, ic in zip(minterm, implicant, strict=True))


def _disjoint_cover(minterms: set[str], primes: list[str]) -> list[str]:
    """Pick a subset of `primes` such that each minterm is covered by exactly one.

    Greedy: at each step pick the prime that covers the most still-uncovered
    minterms AND doesn't overlap any already-covered minterm. If no such prime
    fits, fall back to including the bare minterm as its own implicant — this
    only happens for pathological residuals.

    Disjointness matters because:
    - NEQR's circuit XORs intensity bits — overlap → double-flip → wrong bit.
    - FRQI sums controlled-RY angles — overlap → double-rotation → wrong intensity.
    """
    remaining = set(minterms)
    selected: list[str] = []
    primes_left = list(primes)
    while remaining:
        best: str | None = None
        best_new: set[str] = set()
        for p in primes_left:
            covered = {m for m in minterms if _implicant_matches(m, p)}
            if covered - remaining:
                continue  # would overlap previously-covered minterms
            if len(covered) > len(best_new):
                best = p
                best_new = covered
        if best is None:
            # Pathological: take a single minterm as its own implicant.
            m = next(iter(remaining))
            selected.append(m)
            remaining.discard(m)
            continue
        selected.append(best)
        remaining -= best_new
        primes_left.remove(best)
    return selected


def position_strings_to_implicants(positions: Iterable[int], width: int) -> list[str]:
    """Pixel indices → disjoint cover of width-bit implicants.

    First runs Quine–McCluskey to find prime implicants, then greedy
    set-partitioning to ensure the returned implicants do not overlap on any
    minterm. This is the cover that can be turned directly into a sequence of
    multi-controlled gates.
    """
    minterm_strings = {format(p, f"0{width}b") for p in positions}
    if not minterm_strings:
        return []
    primes = compress_minterms(minterm_strings)
    return _disjoint_cover(minterm_strings, primes)


# ----------------------------------------------------------------- FRQI ----


class FrqiCompressor:
    """Compressed FRQI encoder.

    Groups pixels by intensity, minimises the resulting Boolean cover, and emits
    one multi-controlled RY per implicant instead of per pixel.

    Parameters
    ----------
    image
        Square ``(2^n, 2^n)`` non-negative integer image (use uint8/uint16).
    normalization
        Intensity that maps to angle π. Auto-inferred from dtype if None.
    """

    def __init__(self, image: np.ndarray, *, normalization: float | None = None) -> None:
        if image.ndim != 2 or image.shape[0] != image.shape[1]:
            raise ValueError(f"image must be square 2D, got {image.shape}")
        side = image.shape[0]
        if side < 2 or (side & (side - 1)) != 0:
            raise ValueError(f"image side must be a power of two ≥ 2, got {side}")
        self.image = image
        self.n = int(np.log2(side))
        if normalization is None:
            if np.issubdtype(image.dtype, np.integer):
                normalization = float(np.iinfo(image.dtype).max)
            else:
                observed = float(image.max())
                normalization = observed if observed > 1.0 else 1.0
        self.normalization = normalization
        self._groups: dict[int, list[int]] | None = None
        self._implicants: dict[int, list[str]] | None = None

    def turn_to_funct(self) -> dict[int, list[int]]:
        """Group pixel linear-indices by intensity. Returns ``{intensity: [pixel_idx, …]}``."""
        flat = self.image.flatten()
        groups: dict[int, list[int]] = defaultdict(list)
        for idx, value in enumerate(flat):
            v = int(value)
            if v == 0:
                continue  # zero-intensity pixels need no RY rotation
            groups[v].append(idx)
        self._groups = dict(groups)
        return self._groups

    def compress(self) -> dict[int, list[str]]:
        """Run Quine–McCluskey on each intensity group; cache and return implicants."""
        if self._groups is None:
            self.turn_to_funct()
        assert self._groups is not None
        width = 2 * self.n
        self._implicants = {
            intensity: position_strings_to_implicants(positions, width)
            for intensity, positions in self._groups.items()
        }
        return self._implicants

    def frqi_image(self) -> QuantumCircuit:
        """Build the compressed FRQI circuit. ``2n + 1`` qubits, fewer gates than
        the per-pixel `frqi_circuit`.
        """
        if self._implicants is None:
            self.compress()
        assert self._implicants is not None

        n = self.n
        total_qubits = 2 * n + 1
        qc = QuantumCircuit(total_qubits)
        qc.h(range(2 * n))
        qc.barrier()

        for intensity, implicants in self._implicants.items():
            theta = float(
                intensities_to_angles(np.array([float(intensity)]), self.normalization)[0]
            )
            for implicant in implicants:
                # Implicant string is MSB-first; encoder convention is LSB-first.
                bits_lsb_first = implicant[::-1]
                # Active controls: positions where the implicant has '0' or '1'
                # (a '-' is a don't-care → no control). For '0' bits we pre-flip
                # so the multi-controlled gate fires when the underlying qubit is 0.
                active_qubits = [q for q, c in enumerate(bits_lsb_first) if c != "-"]
                flips = [q for q, c in enumerate(bits_lsb_first) if c == "0"]
                for q in flips:
                    qc.x(q)
                if active_qubits:
                    ry_gate = RYGate(theta).control(len(active_qubits), annotated=True)
                    qc.append(ry_gate, [*active_qubits, total_qubits - 1])
                else:
                    # All-don't-care implicant: unconditional rotation.
                    qc.ry(theta, total_qubits - 1)
                for q in flips:
                    qc.x(q)
                qc.barrier()

        return qc


# ----------------------------------------------------------------- NEQR ----


class NeqrCompressor:
    """Compressed NEQR encoder.

    For each intensity bit ``i`` of the q-bit register, group the pixels whose
    bit-i is 1, minimise the cover, and emit one multi-CNOT per implicant.
    """

    def __init__(self, image: np.ndarray, q: int = 8) -> None:
        if image.ndim != 2 or image.shape[0] != image.shape[1]:
            raise ValueError(f"image must be square 2D, got {image.shape}")
        side = image.shape[0]
        if side < 2 or (side & (side - 1)) != 0:
            raise ValueError(f"image side must be a power of two ≥ 2, got {side}")
        if q < 1:
            raise ValueError("q must be >= 1")
        max_int = (1 << q) - 1
        if image.min() < 0 or image.max() > max_int:
            raise ValueError(
                f"intensity range {image.min()}–{image.max()} doesn't fit in q={q} bits"
            )
        self.image = image
        self.n = int(np.log2(side))
        self.q = q
        self._implicants: dict[int, list[str]] | None = None

    def compress(self) -> dict[int, list[str]]:
        """Per-bit implicants: ``{bit_i: [implicant, …]}``. Pixels where bit-i = 1
        constitute that bit's minterms.
        """
        flat = self.image.flatten().astype(np.int64)
        width = 2 * self.n
        groups: dict[int, list[str]] = {}
        for bit_i in range(self.q):
            mask = (flat >> bit_i) & 1
            positions = [idx for idx, m in enumerate(mask) if m]
            if positions:
                groups[bit_i] = position_strings_to_implicants(positions, width)
        self._implicants = groups
        return groups

    def neqr_image(self) -> QuantumCircuit:
        """Build the compressed NEQR circuit. ``2n + q`` qubits."""
        if self._implicants is None:
            self.compress()
        assert self._implicants is not None

        n, q = self.n, self.q
        total = 2 * n + q
        qc = QuantumCircuit(total)
        pos_qubits = list(range(q, q + 2 * n))
        qc.h(pos_qubits)
        qc.barrier()

        for bit_i, implicants in self._implicants.items():
            for implicant in implicants:
                bits_lsb_first = implicant[::-1]
                active = [pos_qubits[i] for i, c in enumerate(bits_lsb_first) if c != "-"]
                flips = [pos_qubits[i] for i, c in enumerate(bits_lsb_first) if c == "0"]
                for qbit in flips:
                    qc.x(qbit)
                if active:
                    qc.mcx(active, bit_i)
                else:
                    qc.x(bit_i)
                for qbit in flips:
                    qc.x(qbit)
                qc.barrier()
        return qc
