#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/deps:$PWD/src"
PYTHON=/home/gxhen/Wisig/jscc/deepjscc_backdoor/.venv-direct-v2/bin/python
set +e
"$PYTHON" -m pytest -c /dev/null --rootdir . tests/test_ota_adversarial.py -q > contract_tests.log 2>&1
code=$?
if [ "$code" -eq 0 ]; then
  "$PYTHON" scripts/run_ota_adversarial.py \
    --checkpoint /home/gxhen/Wisig/jscc/deepjscc_backdoor/runs/div2k_lmdb_cv16_20260923/p0_full/best.pt \
    --output-dir runs/pilot_eot --steps 400 --batch-size 4 --psr-db -16 --snr-db 10
  code=$?
fi
echo "$code" > exit_code.txt
exit "$code"
