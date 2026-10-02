#!/bin/bash
# Hardware campaign 4 (paper/HW_CAMPAIGN_4_PROTOCOL.md).
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
if [ "${REMAIN:-0}" -lt 150 ]; then echo "Not enough allowance. Aborting."; exit 1; fi
RUN="$PY scripts/run_hardware_class_b_nonrestoring.py --q 2 --divider nonrestoring --mitigation trex"
$RUN --dataset fourvalue --n 1 --shots 16384 --repeat 5 --backend ibm_kingston --label c4_a_fourvalue_16k_kingston
$RUN --dataset fourvalue --n 1 --shots 4096  --repeat 3 --backend ibm_fez      --label c4_b_fourvalue_4k_fez
$RUN --dataset fourvalue --n 2 --shots 16384 --repeat 2 --backend ibm_kingston --label c4_c_fourvalue_n2_16k_kingston
echo "campaign 4 submitted"
