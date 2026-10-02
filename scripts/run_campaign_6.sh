#!/bin/bash
# Hardware campaign 6 (paper/HW_CAMPAIGN_6_PROTOCOL.md).
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
if [ "${REMAIN:-0}" -lt 130 ]; then echo "Not enough allowance. Aborting."; exit 1; fi
RUN="$PY scripts/run_hardware_class_b_nonrestoring.py --n 1 --divider nonrestoring --mitigation trex --shots 16384"
$RUN --dataset fourvalue_q3 --q 3 --repeat 4 --backend ibm_kingston  --label c6_a_fourvalue_q3_16k_kingston
$RUN --dataset fourvalue_p2 --q 2 --repeat 3 --backend ibm_marrakesh --label c6_b_fourvalue_p2_16k_marrakesh
$RUN --dataset fourvalue_p3 --q 2 --repeat 3 --backend ibm_marrakesh --label c6_b_fourvalue_p3_16k_marrakesh
$RUN --dataset fourvalue_p4 --q 2 --repeat 3 --backend ibm_marrakesh --label c6_b_fourvalue_p4_16k_marrakesh
echo "campaign 6 submitted"
