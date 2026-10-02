#!/bin/bash
# Hardware campaign 3 (paper/HW_CAMPAIGN_3_PROTOCOL.md). Pre-flight: refuse to
# start unless the open-plan allowance has at least 200 s left.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
PY=.venv/bin/python
REMAIN=$($PY - <<'PYEOF' 2>/dev/null
import sys; sys.path.insert(0,'src')
from qimp.runtime import ibm
print(ibm.get_service().usage().get("usage_remaining_seconds", 0))
PYEOF
)
echo "open-plan QPU seconds remaining: $REMAIN"
if [ "${REMAIN:-0}" -lt 200 ]; then echo "Not enough allowance (need >= 200 s). Aborting."; exit 1; fi
RUN="$PY scripts/run_hardware_class_b_nonrestoring.py --n 1 --q 2 --divider nonrestoring --mitigation trex"
$RUN --dataset canonical_shared --shots 16384 --repeat 5 --backend ibm_marrakesh --label c3_c1_balanced_16k_trexonly
$RUN --dataset fourvalue        --shots 16384 --repeat 5 --backend ibm_marrakesh --label c3_c2_fourvalue_16k_trexonly
$RUN --dataset fourvalue        --shots 4096  --repeat 3 --backend ibm_kingston  --label c3_c3_fourvalue_4k_kingston
echo "campaign 3 submitted; outputs under data/output/ibm_hw/"
