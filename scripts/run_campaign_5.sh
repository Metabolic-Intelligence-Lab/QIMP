#!/bin/bash
# Hardware campaign 5 (paper/HW_CAMPAIGN_5_PROTOCOL.md): 4 more 4x4 runs on ibm_kingston.
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
if [ "${REMAIN:-0}" -lt 120 ]; then echo "Not enough allowance. Aborting."; exit 1; fi
$PY scripts/run_hardware_class_b_nonrestoring.py --q 2 --divider nonrestoring --mitigation trex --dataset fourvalue --n 2 --shots 16384 --repeat 4 --backend ibm_kingston --label c5_fourvalue_n2_16k_kingston
echo "campaign 5 submitted"
