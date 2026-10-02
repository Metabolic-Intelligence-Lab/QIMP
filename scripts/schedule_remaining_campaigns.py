"""Run campaigns 19 and 20 as soon as the open-plan allowance regenerates.

The allowance is a rolling 28-day window, so it returns in steps. This waits for each step,
runs the next campaign, and honours the staging rule of HW_CAMPAIGN_19_PROTOCOL.md: k = 1 is
submitted only if k = 0 clears band B1 (P(good) below 0.46 in at least 2 of 3 runs, run mean
within 0.10 of the ideal 0.25).

Usage: .venv/bin/python scripts/schedule_remaining_campaigns.py [--poll 600]
"""
from __future__ import annotations
import argparse, glob, json, subprocess, sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from qimp.runtime import ibm

def remaining() -> int:
    try:
        return int(ibm.get_service().usage().get("usage_remaining_seconds", 0))
    except Exception as e:
        print(f"  usage query failed ({str(e)[:60]}), assuming 0", flush=True); return 0

def wait_for(seconds: int, poll: int) -> None:
    while True:
        r = remaining()
        if r >= seconds:
            print(f"allowance {r} s >= {seconds} s, going", flush=True); return
        print(f"allowance {r} s, waiting for {seconds} s", flush=True); time.sleep(poll)

def run(cmd: list[str], attempts: int = 3) -> bool:
    for a in range(1, attempts + 1):
        print(f"$ {' '.join(cmd)} (attempt {a})", flush=True)
        if subprocess.run(cmd, cwd=REPO).returncode == 0:
            return True
        print("  failed, retrying in 60 s", flush=True); time.sleep(60)
    return False

def k0_band_passes() -> bool:
    ps = []
    for d in sorted(glob.glob(str(REPO / "data/output/ibm_hw/*/runs/c19_k0_*_hw"))):
        ps.append(json.load(open(Path(d) / "metadata.json"))["p_measured"])
    if not ps:
        print("no k=0 runs found", flush=True); return False
    below = sum(1 for p in ps if p < 0.46)
    ok = below >= 2 and abs(float(np.mean(ps)) - 0.25) <= 0.10
    print(f"k=0 runs {['%.3f' % p for p in ps]}, mean {np.mean(ps):.3f}, below 0.46 in {below}/{len(ps)} -> B1 {'passes' if ok else 'fails'}", flush=True)
    return ok

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--poll", type=int, default=600)
    a = ap.parse_args()
    py = str(REPO / ".venv/bin/python")
    wait_for(40, a.poll)
    run([py, "scripts/run_campaign_19.py", "--k", "0", "--rounds", "3"])
    if k0_band_passes() and remaining() >= 20:
        run([py, "scripts/run_campaign_19.py", "--k", "1", "--rounds", "3"])
    for r in (1, 2, 3):
        wait_for(60, a.poll)
        run([py, "scripts/run_campaign_20.py", "--backend", "ibm_kingston", "--rounds", "1", "--start-round", str(r)]
            if r > 1 else [py, "scripts/run_campaign_20.py", "--backend", "ibm_kingston", "--rounds", "1"])
    print("scheduled campaigns finished", flush=True)

if __name__ == "__main__":
    main()
