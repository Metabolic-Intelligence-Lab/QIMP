"""Numbers and the scaling figure of the Computer Physics Communications paper.

Every number of the "Verification and measured resource costs" section that is
not a static property of the code is produced here, with fixed seeds and no
hardware access:

    python scripts/cpc_numbers.py   # writes paper/cpc/cpc_numbers.json and
                                    # paper/cpc/figures/fig3_scaling.png

Sections: state-preparation resources of the five encoders for n = 1..4,
arithmetic blocks versus register width, dividers, sorting network versus the
number of registers, compression of the NEQR preparation, zero-noise
extrapolation on a depolarising model, FRQI PSNR versus shots, and statevector
timing. Transpilation is to {rz, sx, x, cx} at optimisation level 1 without a
coupling map.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

import qimp.processing.arithmetic as arith
from qimp.encoding import compression
from qimp.encoding.frqi import FrqiEncoder
from qimp.encoding.mcrqi import McrqiEncoder
from qimp.encoding.ncqi import NcqiEncoder
from qimp.encoding.neqr import NeqrEncoder
from qimp.encoding.qpie import QpieEncoder
from qimp.metrics import psnr
from qimp.runtime.zne import fold_gates, zne_extrapolate
from qimp.testing import exact_counts, ideal_simulation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "cpc"
BASIS = ["rz", "sx", "x", "cx"]


def resources(qc: QuantumCircuit) -> dict[str, int]:
    t = transpile(qc, basis_gates=BASIS, optimization_level=1, seed_transpiler=0)
    return {"qubits": qc.num_qubits, "depth": t.depth(), "cx": int(t.count_ops().get("cx", 0))}


def log(row: dict[str, Any]) -> None:
    print(row, file=sys.stderr)


class Layout:
    """Allocates consecutive qubit registers."""

    def __init__(self) -> None:
        self.n = 0

    def reg(self, width: int) -> list[int]:
        r = list(range(self.n, self.n + width))
        self.n += width
        return r


def encoders() -> list[dict[str, Any]]:
    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    for n in (1, 2, 3, 4):
        side = 1 << n
        grey = rng.integers(0, 256, (side, side)).astype(np.uint8)
        rgb = rng.integers(0, 256, (side, side, 3)).astype(np.uint8)
        for name, qc in (
            ("FRQI", FrqiEncoder().encode(grey)),
            ("NEQR", NeqrEncoder().encode(grey)),
            ("QPIE", QpieEncoder().encode(grey)),
            ("MCRQI", McrqiEncoder().encode(rgb)),
            ("NCQI", NcqiEncoder().encode(rgb)),
        ):
            rows.append({"n": n, "encoder": name, **resources(qc)})
            log(rows[-1])
    return rows


def arithmetic_blocks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for q in (2, 4, 8):
        lay = Layout()
        a, b, c = lay.reg(q), lay.reg(q), lay.reg(q + 1)
        qc = QuantumCircuit(lay.n)
        arith.q_add(qc, a, b, c)
        rows.append({"q": q, "block": "q_add", **resources(qc)})
        qc = QuantumCircuit(lay.n)
        arith.q_sub(qc, a, b, c)
        rows.append({"q": q, "block": "q_sub", **resources(qc)})
        lay = Layout()
        a, b, c, flag = lay.reg(q), lay.reg(q), lay.reg(q + 1), lay.reg(1)[0]
        qc = QuantumCircuit(lay.n)
        arith.neqr_comparator(qc, a, b, c, flag)
        rows.append({"q": q, "block": "neqr_comparator", **resources(qc)})
        qc = QuantumCircuit(lay.n)
        arith.neqr_compare_swap(qc, a, b, c, flag)
        rows.append({"q": q, "block": "neqr_compare_swap", **resources(qc)})
        lay = Layout()
        k_bits = [1, 0, 1]  # k = 5
        b_, acc, c_, guard = lay.reg(q), lay.reg(q + 3), lay.reg(2 * (q + 2)), lay.reg(1)[0]
        qc = QuantumCircuit(lay.n)
        arith.q_mul_const(qc, b_, k_bits, acc, c_, guard)
        rows.append({"q": q, "block": "q_mul_const(k=5)", **resources(qc)})
        for r in rows[-5:]:
            log(r)
    return rows


def dividers() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for q in (2, 3, 4):
        for name in ("q_div_restoring", "q_div_nonrestoring"):
            lay = Layout()
            a, d, quo, work = lay.reg(q), lay.reg(q), lay.reg(q), lay.reg(q)
            pad, c, flag = lay.reg(1)[0], lay.reg((q + 1) * (q + 2)), lay.reg(1)[0]
            qc = QuantumCircuit(lay.n)
            getattr(arith, name)(qc, a, d, quo, work, pad, c, flag)
            rows.append({"q": q, "block": name, **resources(qc)})
            log(rows[-1])
        if q <= 3:
            lay = Layout()
            a, d, quo, flag = lay.reg(q), lay.reg(q), lay.reg(q), lay.reg(1)[0]
            qc = QuantumCircuit(lay.n)
            arith.q_div_lookup(qc, a, d, quo, flag)
            rows.append({"q": q, "block": "q_div_lookup", **resources(qc)})
            log(rows[-1])
        n_, m_ = q + 2, q
        lay = Layout()
        a, d, quo, work = lay.reg(n_), lay.reg(m_), lay.reg(n_), lay.reg(m_)
        pad, c, flag = lay.reg(1)[0], lay.reg((n_ + 1) * (m_ + 2)), lay.reg(1)[0]
        qc = QuantumCircuit(lay.n)
        arith.q_div_general(qc, a, d, quo, work, pad, c, flag)
        rows.append({"q": q, "block": f"q_div_general(n={n_},m={m_})", **resources(qc)})
        log(rows[-1])
    return rows


def sorting_network(q: int = 2) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in (2, 3, 4, 6, 8):
        lay = Layout()
        regs = [lay.reg(q) for _ in range(k)]
        carry = lay.reg(q + 1)
        flags = lay.reg(k * (k - 1) // 2)
        qc = QuantumCircuit(lay.n)
        arith.neqr_sort(qc, regs, carry, flags)
        rows.append({"k": k, "q": q, "comparators": len(flags), **resources(qc)})
        log(rows[-1])
    return rows


def disc_image(side: int) -> np.ndarray:
    yy, xx = np.mgrid[:side, :side]
    r = side / 3.0
    img = np.zeros((side, side), dtype=np.uint8)
    img[(xx - side / 2 + 0.5) ** 2 + (yy - side / 2 + 0.5) ** 2 <= r * r] = 200
    img[0, :] = 50
    return img


def compression_rows() -> list[dict[str, Any]]:
    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    for n in (2, 3, 4, 5):
        side = 1 << n
        images = (
            ("disc", disc_image(side)),
            ("random", rng.integers(0, 256, (side, side)).astype(np.uint8)),
        )
        for shape, img in images:
            row: dict[str, Any] = {
                "n": n,
                "shape": shape,
                "q": 8,
                "pixels": side * side,
                "set_bits": int(sum(bin(int(v)).count("1") for v in img.flatten())),
            }
            for method in ("qm", "espresso"):
                if method == "qm" and n >= 5:
                    continue
                t0 = time.time()
                comp = compression.NeqrCompressor(img, q=8, method=method)
                cover = comp.compress()
                row[f"cubes_{method}"] = sum(len(v) for v in cover.values())
                row[f"seconds_{method}"] = round(time.time() - t0, 3)
                if n <= 3:
                    row[f"cx_{method}"] = resources(comp.neqr_image())["cx"]
            if n <= 3:
                row["cx_plain"] = resources(NeqrEncoder(q=8).encode(img))["cx"]
            rows.append(row)
            log(row)
    return rows


def tvd(p: dict[str, int], q: dict[str, int]) -> float:
    sp, sq = sum(p.values()), sum(q.values())
    return 0.5 * sum(abs(p.get(k, 0) / sp - q.get(k, 0) / sq) for k in set(p) | set(q))


def zne_rows(shots: int = 20000) -> list[dict[str, Any]]:
    """Counts-level ZNE on the depolarising model of the unit tests."""
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(0.03, 2), ["cx"])
    noise.add_all_qubit_quantum_error(depolarizing_error(0.003, 1), ["sx", "x"])
    sim = AerSimulator(noise_model=noise, basis_gates=BASIS)
    rng = np.random.default_rng(1)
    circuits = (
        (
            "FRQI 2x2 (5 qubits)",
            FrqiEncoder().encode(rng.integers(0, 256, (2, 2)).astype(np.uint8)),
        ),
        ("NEQR 2x2, q=2 (6 qubits)", NeqrEncoder(q=2).encode(rng.integers(0, 4, (2, 2)))),
    )
    rows: list[dict[str, Any]] = []
    for label, qc in circuits:
        ideal = exact_counts(qc)
        measured = qc.copy()
        measured.measure_all()
        t = transpile(measured, basis_gates=BASIS, optimization_level=1, seed_transpiler=0)
        scales = (1.0, 3.0, 5.0)
        folded = [fold_gates(t, s) for s in scales]
        result = sim.run(folded, shots=shots, seed_simulator=7).result()
        raw = [dict(result.get_counts(i)) for i in range(len(folded))]
        achieved = [float(f.metadata["zne_fold_scale"]) for f in folded]
        row: dict[str, Any] = {
            "circuit": label,
            "cx": int(t.count_ops().get("cx", 0)),
            "depth": t.depth(),
            "shots": shots,
        }
        for s, counts in zip(scales, raw, strict=True):
            row[f"tvd_scale{int(s)}"] = round(tvd(counts, ideal), 4)
        for method in ("richardson", "linear", "exponential"):
            est = zne_extrapolate(raw, achieved, method=method, shots=shots)
            row[f"tvd_{method}"] = round(tvd(est, ideal), 4)
        rows.append(row)
        log(row)
    return rows


def psnr_vs_shots() -> list[dict[str, Any]]:
    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    for n in (2, 3):
        img = rng.integers(0, 256, (1 << n, 1 << n)).astype(np.uint8)
        enc = FrqiEncoder()
        qc = enc.encode(img)
        for shots in (1024, 4096, 16384, 65536, 262144):
            vals = [
                psnr(img, enc.decode(ideal_simulation(qc, shots=shots, seed=s)), max_intensity=255)
                for s in range(3)
            ]
            rows.append(
                {
                    "n": n,
                    "shots": shots,
                    "psnr_mean_db": round(float(np.mean(vals)), 2),
                    "psnr_sd_db": round(float(np.std(vals)), 2),
                }
            )
            log(rows[-1])
    return rows


def timing() -> list[dict[str, Any]]:
    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    for n in (1, 2, 3):
        img = rng.integers(0, 256, (1 << n, 1 << n)).astype(np.uint8)
        for name, enc in (("NEQR", NeqrEncoder()), ("FRQI", FrqiEncoder())):
            qc = enc.encode(img)
            t0 = time.time()
            exact_counts(qc)
            rows.append(
                {
                    "n": n,
                    "encoder": name,
                    "qubits": qc.num_qubits,
                    "seconds": round(time.time() - t0, 2),
                }
            )
            log(rows[-1])
    return rows


def scaling_figure(enc: list[dict[str, Any]], ps: list[dict[str, Any]], path: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, marker in (
        ("FRQI", "o"),
        ("NEQR", "s"),
        ("QPIE", "^"),
        ("MCRQI", "v"),
        ("NCQI", "D"),
    ):
        rows = [r for r in enc if r["encoder"] == name]
        axs[0].semilogy(
            [r["n"] for r in rows], [max(int(r["cx"]), 1) for r in rows], marker=marker, label=name
        )
    axs[0].set_xlabel("position qubits per axis n (image side 2^n)")
    axs[0].set_ylabel("CX gates after transpilation")
    axs[0].set_xticks([1, 2, 3, 4])
    axs[0].legend(fontsize=8)
    axs[0].set_title("State preparation, 8-bit random images", fontsize=10)
    for n, marker in ((2, "o"), (3, "s")):
        rows = [r for r in ps if r["n"] == n]
        x = np.array([r["shots"] for r in rows], dtype=float)
        y = np.array([r["psnr_mean_db"] for r in rows], dtype=float)
        e = np.array([r["psnr_sd_db"] for r in rows], dtype=float)
        axs[1].errorbar(x, y, yerr=e, marker=marker, capsize=2, label=f"FRQI {1 << n}x{1 << n}")
    x = np.array([1024.0, 262144.0])
    y0 = float(next(r for r in ps if r["n"] == 3)["psnr_mean_db"])
    axs[1].plot(x, y0 + 10 * np.log10(x / 1024), "k--", lw=0.8, label="10 dB per decade")
    axs[1].set_xscale("log")
    axs[1].set_xlabel("shots")
    axs[1].set_ylabel("PSNR (dB)")
    axs[1].legend(fontsize=8)
    axs[1].set_title("Shot-noise limit of amplitude read-out", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    numbers = {
        "encoders": encoders(),
        "arithmetic": arithmetic_blocks(),
        "dividers": dividers(),
        "sort": sorting_network(),
        "compression": compression_rows(),
        "zne": zne_rows(),
        "psnr_vs_shots": psnr_vs_shots(),
        "timing": timing(),
    }
    (OUT / "cpc_numbers.json").write_text(json.dumps(numbers, indent=2))
    scaling_figure(
        numbers["encoders"], numbers["psnr_vs_shots"], OUT / "figures" / "fig3_scaling.png"
    )
    print(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()
