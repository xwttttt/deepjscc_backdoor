#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/deps:$PWD/src"
PYTHON=/home/gxhen/Wisig/jscc/deepjscc_backdoor/.venv-direct-v2/bin/python
TARGET=/home/gxhen/Wisig/jscc/deepjscc_backdoor/runs/div2k_lmdb_cv16_20260923/p0_full/best.pt
set +e
"$PYTHON" scripts/run_magmaw_c16_surrogate.py --stage fit --output-dir runs/full --proxy-steps 10000
code=$?
if [ "$code" -eq 0 ]; then
  "$PYTHON" scripts/run_magmaw_c16_surrogate.py --stage attack --output-dir runs/full --attack-steps 12000 --adaptive-steps 80
  code=$?
fi
if [ "$code" -eq 0 ]; then
  "$PYTHON" scripts/run_magmaw_c16_surrogate.py --stage evaluate --output-dir runs/full --target-checkpoint "$TARGET" --evaluation-repeats 5
  code=$?
fi
if [ "$code" -eq 0 ]; then
  "$PYTHON" scripts/report_magmaw_c16_surrogate.py runs/full/metrics.json --output-dir runs/full/report > report.log 2>&1
  code=$?
fi
echo "$code" > exit_code.txt
exit "$code"
