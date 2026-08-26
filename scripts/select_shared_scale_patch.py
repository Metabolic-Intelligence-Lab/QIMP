"""Pick the Laurdan hardware patch under *shared-scale* quantisation.

The canonical patch of `run_autonomous_n3_laurdan.prepare_canonical` quantises
the two emission channels **independently**, each onto its own [min, max] ->
[0, 2^q-1] scale. On a 2x2 block-mean that renormalises the inter-channel
contrast away and lands on I_a == I_b, so the Class-B integer ratio degenerates
to R = 1 everywhere (§6.5.2). §5.3 avoids exactly this by putting both channels
on a single shared scale, the two emission bands being physically on one
photometric scale.

This script scans the frame for the best 2x2 patch under shared-scale
quantisation and reports it with the null baselines a reviewer will ask for.

Selection criteria, in order:
  1. non-degenerate: I_a != I_b and R takes more than one value
  2. low-order: R subset of {0, 1} — the regime that survives ~700 CX (§7.2)
  3. balanced: minimise the score an all-one-value readout would obtain,
     so a floored device cannot match by accident
  4. few divide-by-zero pixels: those match on the flag bit alone, and the
     paper already discounts a match rate dominated by them (Table 5, note †)

Usage:
    python scripts/select_shared_scale_patch.py [--q 2] [--n 1] [--top 5]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SRC = (
    REPO
    / "data"
    / "immagini"
    / "trainQML"
    / "membraneStack_Sample011_L_UV_DC_001rbc3DM2.tif"
)
OUT = REPO / "paper" / "data_autonomous"


def shared_quantise(
    a: np.ndarray, b: np.ndarray, q: int
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Quantise both channels onto one shared [min, max] -> [0, 2^q-1] scale."""
    lo = min(float(a.min()), float(b.min()))
    hi = max(float(a.max()), float(b.max()))
    if hi == lo:
        return None, None
    span = float((1 << q) - 1)

    def f(x: np.ndarray) -> np.ndarray:
        return np.clip(np.round((x - lo) / (hi - lo) * span), 0, span).astype(np.int64)

    return f(a), f(b)


def block_mean(arr: np.ndarray, side: int) -> np.ndarray:
    b = arr.shape[0] // side
    return arr[: b * side, : b * side].reshape(side, b, side, b).mean(axis=(1, 3))


def classical_ratio(I_a: np.ndarray, I_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    divzero = I_b == 0
    R = np.where(~divzero, I_a // np.maximum(I_b, 1), 0)
    return R, divzero


def null_baselines(ratio: np.ndarray, divzero: np.ndarray, q: int) -> dict[str, float]:
    """What a device that has lost the signal would still score.

    `constant` is the best match rate obtainable by emitting one fixed
    quotient value at every pixel (the failure mode of a floored readout);
    `uniform` is the expected rate of a uniform-random readout over the
    2^q codomain. A result is only informative above both.
    """
    n = ratio.size
    best_const = 0
    for v in range(1 << q):
        # a constant-v readout also emits a constant flag, so credit the
        # divzero pixels only if that constant flag is the divzero one
        best_const = max(
            best_const,
            int(((ratio == v) & ~divzero).sum()),
            int(divzero.sum()),
        )
    return {
        "constant_readout": best_const / n,
        "uniform_random": (1.0 / (1 << q)) * float((~divzero).sum()) / n
        + float(divzero.sum()) / n * 0.5,
    }


def score(I_a: np.ndarray, I_b: np.ndarray, q: int) -> tuple | None:
    if np.array_equal(I_a, I_b):
        return None
    R, divzero = classical_ratio(I_a, I_b)
    live = R[~divzero]
    if live.size == 0 or len(set(live.tolist())) < 2:
        return None
    if live.max() > 1:  # keep the low-order regime that survives on hardware
        return None
    base = null_baselines(R, divzero, q)
    # higher is better: distinct values, then low constant-readout baseline,
    # then few divzero pixels
    return (
        len(set(live.tolist())),
        -base["constant_readout"],
        -int(divzero.sum()),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--q", type=int, default=2)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--block", type=int, default=8, help="block-mean window per cell")
    ap.add_argument("--save", action="store_true", help="write the winner to an npz")
    args = ap.parse_args(argv)

    side = 1 << args.n
    img = np.asarray(Image.open(SRC))
    R_ch = img[..., 0].astype(np.float64)
    G_ch = img[..., 1].astype(np.float64)
    h, w = R_ch.shape
    win = side * args.block

    cands = []
    for y, x in itertools.product(range(0, h - win + 1, 2), range(0, w - win + 1, 2)):
        Rd = block_mean(R_ch[y : y + win, x : x + win], side)
        Gd = block_mean(G_ch[y : y + win, x : x + win], side)
        I_a, I_b = shared_quantise(Rd, Gd, args.q)
        if I_a is None:
            continue
        s = score(I_a, I_b, args.q)
        if s is not None:
            cands.append((s, y, x, I_a, I_b))

    if not cands:
        print("no usable patch found")
        return 1
    cands.sort(key=lambda c: c[0], reverse=True)
    print(f"{len(cands)} usable shared-scale patches at n={args.n}, q={args.q}\n")
    for s, y, x, I_a, I_b in cands[: args.top]:
        Rq, dz = classical_ratio(I_a, I_b)
        base = null_baselines(Rq, dz, args.q)
        shown = np.where(dz, -1, Rq).tolist()
        print(
            f"  ({y:3d},{x:3d})  I_a={I_a.tolist()} I_b={I_b.tolist()}\n"
            f"             R={shown} (-1 = divzero)  distinct={s[0]}  divzero={-s[2]}\n"
            f"             null: best constant readout {base['constant_readout']:.0%},"
            f" uniform random {base['uniform_random']:.0%}"
        )

    s, y, x, I_a, I_b = cands[0]
    Rq, dz = classical_ratio(I_a, I_b)
    print(f"\nwinner: offset ({y},{x})")
    if args.save:
        OUT.mkdir(parents=True, exist_ok=True)
        dest = OUT / f"canonical_shared_n{args.n}_q{args.q}.npz"
        np.savez(
            dest,
            I_a=I_a,
            I_b=I_b,
            R_classical=Rq,
            divzero=dz,
            offset=np.array([y, x]),
            n=args.n,
            q=args.q,
            block=args.block,
        )
        meta = OUT / f"canonical_shared_n{args.n}_q{args.q}.json"
        meta.write_text(
            json.dumps(
                {
                    "offset": [int(y), int(x)],
                    "n": args.n,
                    "q": args.q,
                    "block": args.block,
                    "I_a": I_a.tolist(),
                    "I_b": I_b.tolist(),
                    "R": Rq.tolist(),
                    "divzero": dz.tolist(),
                    "null_baselines": null_baselines(Rq, dz, args.q),
                },
                indent=2,
            )
        )
        print(f"wrote {dest.name} and {meta.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
