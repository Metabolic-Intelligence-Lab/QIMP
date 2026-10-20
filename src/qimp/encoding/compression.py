"""FRQI / NEQR compression via Boolean-function minimisation.

Pixels that share the same intensity (FRQI) or the same intensity-bit (NEQR)
form a Boolean function of the position bits. Minimising that function
shrinks the number of control patterns needed, which reduces
the per-pixel multi-controlled gates issued by the encoder.

Two self-contained minimisers are provided, with no extra dependency:

- Quine–McCluskey (`compress_minterms`) followed by a greedy disjoint cover.
  Used by default for small position registers.
- An ESPRESSO-style heuristic (`espresso_minimize`: EXPAND / IRREDUNDANT /
  REDUCE on cube covers) followed by `make_disjoint_cover`. Used by default
  for ``2n >= ESPRESSO_AUTO_MIN_WIDTH`` (``n >= 5``), where enumerating all
  prime implicants becomes expensive.

Both paths return a disjoint cover that reproduces the on-set exactly, which
the encoders require. Select the path with ``method="auto" | "qm" |
"espresso"`` on `FrqiCompressor`, `NeqrCompressor` or
`position_strings_to_implicants`.

References
----------
- W. V. Quine, "The problem of simplifying truth functions", Amer. Math.
  Monthly 59 (1952); E. J. McCluskey, "Minimization of Boolean functions",
  Bell Syst. Tech. J. 35 (1956).
- R. K. Brayton, G. D. Hachtel, C. T. McMullen, A. L. Sangiovanni-Vincentelli,
  "Logic Minimization Algorithms for VLSI Synthesis", Kluwer (1984).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate

from qimp.encoding.frqi import intensities_to_angles

__all__ = [
    "ESPRESSO_AUTO_MIN_WIDTH",
    "CompressionMethod",
    "FrqiCompressor",
    "NeqrCompressor",
    "compress_minterms",
    "espresso_minimize",
    "make_disjoint_cover",
    "position_strings_to_implicants",
]

#: Minimiser selection for the compressors.
CompressionMethod = Literal["auto", "qm", "espresso"]
_METHODS: tuple[str, ...] = ("auto", "qm", "espresso")


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

    For typical FRQI/NEQR uses (n ≤ 4) this is fast and correct. For larger
    fan-ins use `espresso_minimize`.
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


# ----------------------------------------------------------------- ESPRESSO ----
#
# Cubes are held internally as ``(mask, value)`` integer pairs over a fixed
# width ``w``. Bit ``i`` of ``mask`` is set when variable ``i`` appears as a
# literal; the matching bit of ``value`` gives its polarity (``value`` is zero
# outside ``mask``). Bit ``i`` corresponds to character ``w - 1 - i`` of the
# MSB-first implicant string, so ``"1-0"`` is ``(0b101, 0b100)``.

_Cube = tuple[int, int]

#: Widths at or above this value use ESPRESSO when ``method="auto"``.
#: ``2n >= 10`` corresponds to images of side ``2^n >= 32``.
ESPRESSO_AUTO_MIN_WIDTH = 10

#: Upper bound on REDUCE / EXPAND / IRREDUNDANT rounds after the first pass.
_ESPRESSO_MAX_ROUNDS = 32


def _parse_cube(s: str, width: int) -> _Cube:
    if len(s) != width:
        raise ValueError(f"implicant {s!r} has width {len(s)}, expected {width}")
    mask = 0
    value = 0
    for j, ch in enumerate(s):
        bit = 1 << (width - 1 - j)
        if ch == "1":
            mask |= bit
            value |= bit
        elif ch == "0":
            mask |= bit
        elif ch != "-":
            raise ValueError(f"implicant {s!r} contains {ch!r}; allowed are '0', '1', '-'")
    return mask, value


def _cube_to_str(cube: _Cube, width: int) -> str:
    mask, value = cube
    chars: list[str] = []
    for j in range(width):
        bit = 1 << (width - 1 - j)
        if not mask & bit:
            chars.append("-")
        else:
            chars.append("1" if value & bit else "0")
    return "".join(chars)


def _intersects(a: _Cube, b: _Cube) -> bool:
    return ((a[1] ^ b[1]) & a[0] & b[0]) == 0


def _contains(a: _Cube, b: _Cube) -> bool:
    """True iff cube `a` contains cube `b`."""
    return (a[0] & ~b[0]) == 0 and ((a[1] ^ b[1]) & a[0]) == 0


def _cofactor(cubes: Iterable[_Cube], c: _Cube) -> list[_Cube]:
    """Cofactor of a cover with respect to cube `c` (variables of `c` are freed)."""
    cm, cv = c
    return [(m & ~cm, v & ~cm) for m, v in cubes if ((v ^ cv) & m & cm) == 0]


def _split_variable(cubes: list[_Cube]) -> tuple[int, bool]:
    """Pick the splitting bit for the unate recursive paradigm.

    Returns ``(bit, binate)``. A binate variable (both polarities present) is
    preferred; ties are broken by literal count, then by the lowest bit.
    """
    ones: dict[int, int] = defaultdict(int)
    zeros: dict[int, int] = defaultdict(int)
    for m, v in cubes:
        rest = m
        while rest:
            bit = rest & -rest
            rest ^= bit
            if v & bit:
                ones[bit] += 1
            else:
                zeros[bit] += 1
    best_bit = 0
    best_key: tuple[int, int, int] = (-1, -1, 0)
    for bit in set(ones) | set(zeros):
        binate = 1 if ones[bit] and zeros[bit] else 0
        key = (binate, ones[bit] + zeros[bit], -bit)
        if key > best_key:
            best_key = key
            best_bit = bit
    return best_bit, best_key[0] == 1


def _tautology(cubes: list[_Cube], width: int) -> bool:
    """True iff the cover is the constant-1 function on `width` variables."""
    if not cubes:
        return False
    if any(m == 0 for m, _ in cubes):
        return True
    # Minterm-count bound: overlapping cubes are counted more than once, so a
    # total below 2^width proves the cover is not a tautology.
    if sum(1 << (width - m.bit_count()) for m, _ in cubes) < (1 << width):
        return False
    bit, binate = _split_variable(cubes)
    if not binate:
        # A unate cover is a tautology only if it holds the universal cube.
        return False
    return _tautology(_cofactor(cubes, (bit, 0)), width) and _tautology(
        _cofactor(cubes, (bit, bit)), width
    )


def _cube_covered(c: _Cube, cover: list[_Cube], width: int) -> bool:
    """True iff cube `c` is contained in the union of `cover`."""
    return _tautology(_cofactor(cover, c), width)


def _complement(cubes: list[_Cube]) -> list[_Cube]:
    """Complement of a cover, by Shannon expansion on the most binate variable."""
    if not cubes:
        return [(0, 0)]
    if any(m == 0 for m, _ in cubes):
        return []
    if len(cubes) == 1:
        # De Morgan, written as a disjoint sum of cubes.
        m, v = cubes[0]
        out: list[_Cube] = []
        fixed_m = 0
        rest = m
        while rest:
            bit = rest & -rest
            rest ^= bit
            out.append((fixed_m | bit, (v & fixed_m) | (~v & bit)))
            fixed_m |= bit
        return out
    bit, _ = _split_variable(cubes)
    comp0 = _complement(_cofactor(cubes, (bit, 0)))
    comp1 = _complement(_cofactor(cubes, (bit, bit)))
    set1 = set(comp1)
    common = [c for c in comp0 if c in set1]
    common_set = set(common)
    out = list(common)
    out.extend((m | bit, v) for m, v in comp0 if (m, v) not in common_set)
    out.extend((m | bit, v | bit) for m, v in comp1 if (m, v) not in common_set)
    return out


def _supercube(cubes: list[_Cube]) -> _Cube:
    """Smallest cube containing every cube in a non-empty list."""
    mask = ~0
    v0 = cubes[0][1]
    disagree = 0
    for m, v in cubes:
        mask &= m
        disagree |= v ^ v0
    mask &= ~disagree
    return mask, v0 & mask


def _cover_cost(cover: list[_Cube]) -> tuple[int, int]:
    return len(cover), sum(m.bit_count() for m, _ in cover)


def _expand(cover: list[_Cube], off_set: list[_Cube]) -> list[_Cube]:
    """EXPAND: raise literals of each cube while it stays disjoint from the off-set.

    Cubes are visited largest first. Among the literals that can be raised,
    the one that makes the cube contain the most not-yet-covered cubes is
    chosen; ties prefer the literal that moves the most cubes closer to
    containment.
    """
    order = sorted(cover, key=lambda c: (c[0].bit_count(), c))
    expanded: list[_Cube] = []
    done = [False] * len(order)
    for idx, cube in enumerate(order):
        if done[idx]:
            continue
        done[idx] = True
        m, v = cube
        # Bits on which each off-set cube still conflicts with the current cube.
        blockers = [(v ^ rv) & m & rm for rm, rv in off_set]
        while True:
            blocked = 0
            for d in blockers:
                if d & (d - 1) == 0:
                    blocked |= d
            candidates = m & ~blocked
            if not candidates:
                break
            covers_now: dict[int, int] = defaultdict(int)
            progress: dict[int, int] = defaultdict(int)
            for j in range(idx + 1, len(order)):
                if done[j]:
                    continue
                gm, gv = order[j]
                diff = ((v ^ gv) & m) | (m & ~gm)
                if diff & ~candidates:
                    continue
                if diff.bit_count() == 1:
                    covers_now[diff] += 1
                rest = diff
                while rest:
                    bit = rest & -rest
                    rest ^= bit
                    progress[bit] += 1
            best_bit = 0
            best_key: tuple[int, int, int] = (-1, -1, 0)
            rest = candidates
            while rest:
                bit = rest & -rest
                rest ^= bit
                key = (covers_now[bit], progress[bit], bit)
                if key > best_key:
                    best_key = key
                    best_bit = bit
            m &= ~best_bit
            v &= ~best_bit
            blockers = [d & ~best_bit for d in blockers]
        new = (m, v)
        for j in range(idx + 1, len(order)):
            if not done[j] and _contains(new, order[j]):
                done[j] = True
        expanded.append(new)
    # Single-cube containment: drop cubes swallowed by a later expansion.
    result: list[_Cube] = []
    for i, c in enumerate(expanded):
        if any(j != i and _contains(o, c) and (o != c or j < i) for j, o in enumerate(expanded)):
            continue
        result.append(c)
    return result


def _irredundant(cover: list[_Cube], dc_set: list[_Cube], width: int) -> list[_Cube]:
    """IRREDUNDANT: greedily drop cubes covered by the rest of the cover plus the DC-set.

    Smaller cubes (more literals) are tried first, so large cubes are kept.
    """
    current = list(cover)
    for c in sorted(cover, key=lambda c: (-c[0].bit_count(), c)):
        others = [o for o in current if o != c]
        if _cube_covered(c, others + dc_set, width):
            current = others
    return current


def _reduce(cover: list[_Cube], dc_set: list[_Cube]) -> list[_Cube]:
    """REDUCE: shrink each cube to the smallest cube still needed by the cover.

    Each cube ``c`` is replaced by ``c`` intersected with the supercube of the
    part of ``c`` not covered by the other cubes and the DC-set. Cubes that
    become empty are dropped. Cubes are processed largest first and the cover
    is updated in place, so later cubes see the earlier reductions.
    """
    current = sorted(cover, key=lambda c: (c[0].bit_count(), c))
    i = 0
    while i < len(current):
        c = current[i]
        others = current[:i] + current[i + 1 :] + dc_set
        uncovered = _complement(_cofactor(others, c))
        if not uncovered:
            current.pop(i)
            continue
        sm, sv = _supercube(uncovered)
        current[i] = (sm | c[0], sv | c[1])
        i += 1
    return current


def _espresso_cubes(on_set: list[_Cube], dc_set: list[_Cube], width: int) -> list[_Cube]:
    off_set = _complement(on_set + dc_set)
    cover = _expand(on_set, off_set)
    cover = _irredundant(cover, dc_set, width)
    cost = _cover_cost(cover)
    for _ in range(_ESPRESSO_MAX_ROUNDS):
        candidate = _reduce(cover, dc_set)
        candidate = _expand(candidate, off_set)
        candidate = _irredundant(candidate, dc_set, width)
        new_cost = _cover_cost(candidate)
        if new_cost >= cost:
            break
        cover, cost = candidate, new_cost
    return cover


def _check_width(terms: Iterable[str], width: int | None) -> tuple[list[str], int | None]:
    items = list(terms)
    for s in items:
        if width is None:
            width = len(s)
        elif len(s) != width:
            raise ValueError(f"implicant {s!r} has width {len(s)}, expected {width}")
    return items, width


def espresso_minimize(minterms: Iterable[str], dont_cares: Iterable[str] = ()) -> list[str]:
    """Heuristic two-level minimisation of a single-output Boolean function.

    Pure-Python ESPRESSO-style loop: the on-set is expanded into prime cubes
    against the off-set (EXPAND), redundant cubes are removed (IRREDUNDANT),
    then the cover is repeatedly reduced (REDUCE), re-expanded and made
    irredundant until the cost (number of cubes, then number of literals)
    stops decreasing.

    Parameters
    ----------
    minterms
        On-set, as fixed-width strings over ``"0"``, ``"1"`` (and optionally
        ``"-"``, so cubes are accepted too). MSB first.
    dont_cares
        Optional DC-set in the same format. The result may cover any part of
        it.

    Returns
    -------
    list[str]
        Sorted implicant strings. Their union contains the on-set and is
        contained in on-set plus DC-set. The cubes are prime and irredundant
        but may overlap; see `make_disjoint_cover` for an exact partition.
        The cover is not guaranteed to be minimum.
    """
    on_items, width = _check_width(minterms, None)
    dc_items, width = _check_width(dont_cares, width)
    if not on_items:
        return []
    assert width is not None
    on_set = sorted({_parse_cube(s, width) for s in on_items})
    dc_set = sorted({_parse_cube(s, width) for s in dc_items})
    cover = _espresso_cubes(on_set, dc_set, width)
    return sorted(_cube_to_str(c, width) for c in cover)


def _disjoint_sharp(a: _Cube, b: _Cube) -> list[_Cube]:
    """``a # b`` as a list of pairwise disjoint cubes (empty if ``b`` contains ``a``)."""
    if not _intersects(a, b):
        return [a]
    am, av = a
    bm, bv = b
    out: list[_Cube] = []
    rest = bm & ~am
    while rest:
        bit = rest & -rest
        rest ^= bit
        out.append((am | bit, av | (~bv & bit)))
        am |= bit
        av |= bv & bit
    return out


def _merge_adjacent(cubes: list[_Cube]) -> list[_Cube]:
    """Merge pairs of cubes with the same mask that differ in one literal.

    The union of two such cubes is the cube with that literal raised, so a
    disjoint cover stays disjoint and covers the same minterms.
    """
    current = set(cubes)
    changed = True
    while changed:
        changed = False
        for m, v in sorted(current, key=lambda c: (-c[0].bit_count(), c)):
            if (m, v) not in current:
                continue
            rest = m
            while rest:
                bit = rest & -rest
                rest ^= bit
                partner = (m, v ^ bit)
                if partner in current:
                    current.discard((m, v))
                    current.discard(partner)
                    current.add((m & ~bit, v & ~bit))
                    changed = True
                    break
    return sorted(current, key=lambda c: (c[0].bit_count(), c))


def make_disjoint_cover(cover: Iterable[str]) -> list[str]:
    """Rewrite a cover as pairwise disjoint implicants with the same union.

    Cubes are taken largest first; each new cube is cut by disjoint sharp
    against every cube already placed, then adjacent pieces are merged. Every
    output cube contains at least one minterm, so the result never has more
    cubes than the function has minterms.
    """
    items, width = _check_width(cover, None)
    if not items:
        return []
    assert width is not None
    cubes = sorted({_parse_cube(s, width) for s in items}, key=lambda c: (c[0].bit_count(), c))
    placed: list[_Cube] = []
    for c in cubes:
        pieces = [c]
        for p in placed:
            pieces = [piece for q in pieces for piece in _disjoint_sharp(q, p)]
            if not pieces:
                break
        placed.extend(pieces)
    return [_cube_to_str(c, width) for c in _merge_adjacent(placed)]


def _check_method(method: str) -> None:
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")


def position_strings_to_implicants(
    positions: Iterable[int], width: int, *, method: CompressionMethod = "auto"
) -> list[str]:
    """Pixel indices → disjoint cover of width-bit implicants.

    With ``method="qm"``, runs Quine–McCluskey to find prime implicants, then
    greedy set-partitioning to ensure the returned implicants do not overlap
    on any minterm. With ``method="espresso"``, runs `espresso_minimize` and
    then `make_disjoint_cover`. ``method="auto"`` uses Quine–McCluskey for
    ``width < ESPRESSO_AUTO_MIN_WIDTH`` and ESPRESSO otherwise.

    Either way the result covers each given position exactly once and no
    other position, so it can be turned directly into a sequence of
    multi-controlled gates.
    """
    _check_method(method)
    minterm_strings = {format(p, f"0{width}b") for p in positions}
    if not minterm_strings:
        return []
    use_espresso = method == "espresso" or (method == "auto" and width >= ESPRESSO_AUTO_MIN_WIDTH)
    if use_espresso:
        return make_disjoint_cover(espresso_minimize(minterm_strings))
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
    method
        Minimiser: ``"qm"`` (Quine–McCluskey), ``"espresso"`` or ``"auto"``
        (ESPRESSO when ``2n >= ESPRESSO_AUTO_MIN_WIDTH``). See
        `position_strings_to_implicants`.
    """

    def __init__(
        self,
        image: np.ndarray,
        *,
        normalization: float | None = None,
        method: CompressionMethod = "auto",
    ) -> None:
        _check_method(method)
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
        self.method: CompressionMethod = method
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
        """Minimise each intensity group (see ``method``); cache and return implicants."""
        if self._groups is None:
            self.turn_to_funct()
        assert self._groups is not None
        width = 2 * self.n
        self._implicants = {
            intensity: position_strings_to_implicants(positions, width, method=self.method)
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

    Parameters
    ----------
    image
        Square ``(2^n, 2^n)`` integer image with values in ``[0, 2^q)``.
    q
        Number of intensity qubits.
    method
        Minimiser: ``"qm"``, ``"espresso"`` or ``"auto"``, as in
        `FrqiCompressor`.
    """

    def __init__(
        self, image: np.ndarray, q: int = 8, *, method: CompressionMethod = "auto"
    ) -> None:
        _check_method(method)
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
        self.method: CompressionMethod = method
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
                groups[bit_i] = position_strings_to_implicants(positions, width, method=self.method)
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
