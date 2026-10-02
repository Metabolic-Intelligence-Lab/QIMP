#!/bin/bash
# Hardware campaign 10 (paper/HW_CAMPAIGN_10_PROTOCOL.md). Uses the default saved account.
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
if [ "${REMAIN:-0}" -lt 170 ]; then echo "Not enough allowance. Aborting."; exit 1; fi
RUN="$PY scripts/run_hardware_class_b_nonrestoring.py --q 2 --divider nonrestoring --mitigation trex --load ucry"
$RUN --dataset fourvalue        --n 2 --shots 16384 --repeat 3 --backend ibm_fez       --label c10_a_fourvalue_n2_ucry_fez
$RUN --dataset fourvalue        --n 2 --shots 16384 --repeat 3 --backend ibm_marrakesh --label c10_b_fourvalue_n2_ucry_marrakesh
$RUN --dataset canonical_shared --n 2 --shots 16384 --repeat 3 --backend ibm_kingston  --label c10_c_laurdan_n2_ucry_kingston
$RUN --dataset canonical_shared --n 3 --shots 65536 --repeat 3 --backend ibm_kingston  --label c10_d_laurdan_n3_ucry_kingston
$RUN --dataset random4          --n 2 --shots 16384 --repeat 3 --backend ibm_marrakesh --label c10_e_random4_n2_ucry_marrakesh
echo "campaign 10 submitted"
