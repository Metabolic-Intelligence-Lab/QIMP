#!/bin/bash
# Relaunch of campaign 3 after the C2 argparse failure (see protocol deviation log). C2 and C3 only.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
RUN=".venv/bin/python scripts/run_hardware_class_b_nonrestoring.py --n 1 --q 2 --divider nonrestoring --mitigation trex"
$RUN --dataset fourvalue --shots 16384 --repeat 5 --backend ibm_marrakesh --label c3_c2_fourvalue_16k_trexonly
$RUN --dataset fourvalue --shots 4096  --repeat 3 --backend ibm_kingston  --label c3_c3_fourvalue_4k_kingston
echo "campaign 3 (C2, C3) submitted"
