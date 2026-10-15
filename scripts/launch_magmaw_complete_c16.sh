#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/deps:$PWD/src"
PYTHON=/home/gxhen/Wisig/jscc/deepjscc_backdoor/.venv-direct-v2/bin/python
TARGET=/home/gxhen/Wisig/jscc/deepjscc_backdoor/runs/div2k_lmdb_cv16_20260923/p0_full/best.pt
PROXIES=/home/gxhen/Wisig/jscc/deepjscc_backdoor/experiments/magmaw_c16_surrogate_20261014_01/runs/full/models
WARM=/home/gxhen/Wisig/jscc/deepjscc_backdoor/experiments/magmaw_c16_uap_diagnosis_20261014_01/runs/full/shared1408_no_permutation.pt
set +e
"$PYTHON" scripts/run_magmaw_complete_c16.py --checkpoint "$TARGET" --proxy-dir "$PROXIES" --warm-checkpoint "$WARM" --output-dir runs/smoke --steps 2 --profiles balanced --smoke
code=$?
if [ "$code" -eq 0 ]; then
  "$PYTHON" scripts/run_magmaw_complete_c16.py --checkpoint "$TARGET" --proxy-dir "$PROXIES" --warm-checkpoint "$WARM" --output-dir runs/full --steps 1600 --eot-samples 2 --profiles balanced regularizer_off --evaluation-repeats 3
  code=$?
fi
if [ "$code" -eq 0 ]; then
  "$PYTHON" scripts/report_magmaw_complete_c16.py runs/full/metrics.json --output-dir runs/full/report > report.log 2>&1
  code=$?
fi
echo "$code" > exit_code.txt
exit "$code"
