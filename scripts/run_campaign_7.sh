#!/bin/bash
# Hardware campaign 7 (paper/HW_CAMPAIGN_7_PROTOCOL.md): UCRY load, 4x4 and 8x8 on ibm_kingston.
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
if [ "${REMAIN:-0}" -lt 160 ]; then echo "Not enough allowance. Aborting."; exit 1; fi
RUN="$PY scripts/run_hardware_class_b_nonrestoring.py --q 2 --divider nonrestoring --mitigation trex --load ucry --backend ibm_kingston"
$RUN --dataset fourvalue        --n 2 --shots 16384 --repeat 4 --label c7_a_fourvalue_n2_ucry_kingston
$RUN --dataset random4          --n 2 --shots 16384 --repeat 4 --label c7_b_random4_n2_ucry_kingston
$RUN --dataset canonical_shared --n 2 --shots 16384 --repeat 3 --label c7_c_laurdan_n2_ucry_kingston
$RUN --dataset random4          --n 3 --shots 65536 --repeat 2 --label c7_d_random4_n3_ucry_kingston
echo "campaign 7 submitted"
