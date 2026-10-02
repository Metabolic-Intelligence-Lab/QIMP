#!/bin/bash
# Hardware campaign 8 (paper/HW_CAMPAIGN_8_PROTOCOL.md): 8x8 images, UCRY load, ibm_kingston.
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
RUN="$PY scripts/run_hardware_class_b_nonrestoring.py --q 2 --n 3 --divider nonrestoring --mitigation trex --load ucry --backend ibm_kingston --shots 65536"
$RUN --dataset random4 --repeat 3 --label c8_a_random4_n3_ucry_kingston
$RUN --dataset fura2   --repeat 3 --label c8_b_fura2_n3_ucry_kingston
echo "campaign 8 submitted"
