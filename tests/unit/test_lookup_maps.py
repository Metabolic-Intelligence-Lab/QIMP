"""Tests for the generic per-pixel synthesis and the Class-A / Class-C maps built on it.

Parametrised over q (intensity width), q_frac (fractional width) and n (spatial qubits),
per the scalability constraint of the library.
"""
from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qimp.processing.arithmetic import synth_pixel_map
from qimp.processing.ratiometric_circuit import (
    class_a_gp_lookup,
    class_c_rogfp_lookup,
    decode_class_a_full,
    decode_class_c_rogfp,
)


def _out(qc: QuantumCircuit, x: int) -> int:
    sv = Statevector.from_int(x, 2**qc.num_qubits).evolve(qc)
    k = int(np.argmax(np.abs(sv.data)))
    assert abs(abs(sv.data[k]) - 1.0) < 1e-9
    return k


@pytest.mark.parametrize("n_in,n_out", [(2, 1), (3, 2), (4, 3)])
def test_synth_pixel_map_random_tables(n_in: int, n_out: int) -> None:
    rng = np.random.default_rng(n_in * 10 + n_out)
    table = [int(v) for v in rng.integers(0, 1 << n_out, size=1 << n_in)]
    qc = QuantumCircuit(n_in + n_out)
    synth_pixel_map(qc, list(range(n_in)), list(range(n_in, n_in + n_out)), table)
    for x in range(1 << n_in):
        assert _out(qc, x) == x | (table[x] << n_in), (x, table[x])


@pytest.mark.parametrize("n_in,n_out", [(2, 2), (4, 2)])
def test_synth_pixel_map_self_inverse(n_in: int, n_out: int) -> None:
    rng = np.random.default_rng(7 + n_in)
    table = [int(v) for v in rng.integers(0, 1 << n_out, size=1 << n_in)]
    qc = QuantumCircuit(n_in + n_out)
    ins, outs = list(range(n_in)), list(range(n_in, n_in + n_out))
    synth_pixel_map(qc, ins, outs, table)
    synth_pixel_map(qc, ins, outs, table)
    for x in range(1 << n_in):
        assert _out(qc, x) == x


def test_synth_pixel_map_rejects_wrong_table_size() -> None:
    qc = QuantumCircuit(3)
    with pytest.raises(ValueError):
        synth_pixel_map(qc, [0, 1], [2], [0, 1, 0])


@pytest.mark.parametrize("q", [1, 2, 3])
@pytest.mark.parametrize("q_frac", [1, 2])
def test_class_a_gp_lookup_matches_its_definition(q: int, q_frac: int) -> None:
    """Every pixel of a 2x2 image decodes to the documented GP convention."""
    rng = np.random.default_rng(100 * q + q_frac)
    Ia = rng.integers(0, 1 << q, size=(2, 2)); Ib = rng.integers(0, 1 << q, size=(2, 2))
    Ia[0, 0], Ib[0, 0] = 0, 0  # force a div-zero pixel
    qc, lay = class_a_gp_lookup(Ia, Ib, q=q, q_frac=q_frac)
    probs = {s: p for s, p in Statevector(qc).probabilities_dict().items() if p > 1e-9}
    counts = {s: int(round(p * 4096)) for s, p in probs.items()}
    gp, dz = decode_class_a_full(counts, 1, q, q_frac, lay, qc.num_qubits)
    for r in range(2):
        for c in range(2):
            a, b = int(Ia[r, c]), int(Ib[r, c])
            assert dz[r, c] == (a + b == 0)
            if a + b:
                mag = (abs(a - b) << q_frac) // (a + b)
                assert gp[r, c] == pytest.approx((-1.0 if b > a else 1.0) * mag / (1 << q_frac))


@pytest.mark.parametrize("q", [1, 2, 3])
@pytest.mark.parametrize("q_frac", [1, 2])
def test_class_c_rogfp_lookup_matches_its_definition(q: int, q_frac: int) -> None:
    rng = np.random.default_rng(200 * q + q_frac)
    Ia = rng.integers(0, 1 << q, size=(2, 2)); Ib = rng.integers(1, 1 << q, size=(2, 2))
    Ib[1, 1] = 0  # force a div-zero pixel
    R_red_fp = 1
    qc, lay = class_c_rogfp_lookup(Ia, Ib, q=q, q_frac=q_frac, R_red_fp=R_red_fp)
    probs = {s: p for s, p in Statevector(qc).probabilities_dict().items() if p > 1e-9}
    counts = {s: int(round(p * 4096)) for s, p in probs.items()}
    rc, dz = decode_class_c_rogfp(counts, 1, q, q_frac, 0.0, 1.0, lay, qc.num_qubits)
    denom = float(1 << q_frac)
    for r in range(2):
        for c in range(2):
            a, b = int(Ia[r, c]), int(Ib[r, c])
            assert dz[r, c] == (b == 0)
            if b:
                assert rc[r, c] == pytest.approx(((a << q_frac) // b - R_red_fp) / denom)


@pytest.mark.parametrize("n", [1, 2])
def test_class_a_gp_lookup_qubit_budget(n: int) -> None:
    side = 1 << n
    Ia = np.ones((side, side), dtype=int); Ib = np.ones((side, side), dtype=int)
    qc, _ = class_a_gp_lookup(Ia, Ib, q=2, q_frac=2)
    assert qc.num_qubits == 2 * n + 2 * 2 + (2 + 1) + 2


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("q", [1, 2])
def test_compressed_ucry_load_is_exact(n: int, q: int) -> None:
    """The QPIXL-style compression of the uniformly controlled load leaves the state
    it prepares unchanged, plane by plane and image by image."""
    from qimp.processing.ratiometric_circuit import dual_neqr_load

    rng = np.random.default_rng(400 + 10 * n + q)
    side = 1 << n
    Ia = rng.integers(0, 1 << q, size=(side, side))
    Ib = rng.integers(0, 1 << q, size=(side, side))

    def build(compress: bool) -> QuantumCircuit:
        qc = QuantumCircuit(2 * n + 2 * q)
        dual_neqr_load(
            qc, Ia, Ib, q,
            position_qubits=list(range(2 * n)),
            intensity_a_qubits=list(range(2 * n, 2 * n + q)),
            intensity_b_qubits=list(range(2 * n + q, 2 * n + 2 * q)),
            mode="ucry", compress=compress,
        )
        return qc

    plain, compressed = Statevector(build(False)), Statevector(build(True))
    assert np.allclose(plain.data, compressed.data, atol=1e-9)


@pytest.mark.parametrize("n", [1, 2])
def test_compressed_ucry_load_round_trip(n: int) -> None:
    """Load then unload with compression returns every qubit to |0>."""
    from qimp.processing.ratiometric_circuit import dual_neqr_load, dual_neqr_load_inv

    rng = np.random.default_rng(500 + n)
    side, q = 1 << n, 2
    Ia = rng.integers(0, 1 << q, size=(side, side))
    Ib = rng.integers(0, 1 << q, size=(side, side))
    qc = QuantumCircuit(2 * n + 2 * q)
    kw = dict(
        position_qubits=list(range(2 * n)),
        intensity_a_qubits=list(range(2 * n, 2 * n + q)),
        intensity_b_qubits=list(range(2 * n + q, 2 * n + 2 * q)),
        mode="ucry", compress=True,
    )
    dual_neqr_load(qc, Ia, Ib, q, **kw)
    dual_neqr_load_inv(qc, Ia, Ib, q, **kw)
    assert abs(abs(Statevector(qc).data[0]) - 1.0) < 1e-9
