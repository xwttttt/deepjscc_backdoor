#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/deps:$PWD/src"
PYTHON=/home/gxhen/Wisig/jscc/deepjscc_backdoor/.venv-direct-v2/bin/python
PREVIOUS=/home/gxhen/Wisig/jscc/deepjscc_backdoor/experiments/magmaw_complete_c16_20261014_02
set +e
"$PYTHON" scripts/run_magmaw_priority_c16.py --previous-root "$PREVIOUS" --output-dir runs/smoke --pilot-steps 2 --steps 2 --batch-size 2 --eot-samples 2 --seeds 1234 --smoke
code=$?
if [ "$code" -eq 0 ]; then
  "$PYTHON" scripts/run_magmaw_priority_c16.py --previous-root "$PREVIOUS" --output-dir runs/full --pilot-steps 400 --steps 4000 --batch-size 8 --eot-samples 4 --evaluation-repeats 5 --seeds 1234 5678
  code=$?
fi
if [ "$code" -eq 0 ]; then
  "$PYTHON" scripts/report_magmaw_priority_c16.py runs/full/metrics.json --output-dir runs/full/report > report.log 2>&1
  code=$?
fi
echo "$code" > exit_code.txt
exit "$code"
