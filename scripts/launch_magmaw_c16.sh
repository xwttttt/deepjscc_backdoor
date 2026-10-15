#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/deps:$PWD/src"
PYTHON=/home/gxhen/Wisig/jscc/deepjscc_backdoor/.venv-direct-v2/bin/python
CHECKPOINT=/home/gxhen/Wisig/jscc/deepjscc_backdoor/runs/div2k_lmdb_cv16_20260923/p0_full/best.pt
set +e
"$PYTHON" scripts/run_magmaw_c16.py --checkpoint "$CHECKPOINT" --output-dir runs/primary --steps 12000 --batch-size 8 --profiles mse psnr --seed 1234 --evaluation-repeats 5 --pgd-reference
code=$?
if [ "$code" -eq 0 ]; then
  "$PYTHON" scripts/report_magmaw_c16.py runs/primary/metrics.json --output-dir runs/primary/report > primary_report.log 2>&1
  code=$?
fi
if [ "$code" -eq 0 ]; then
  selected=$("$PYTHON" -c 'import json; a=json.load(open("runs/primary/metrics.json")); print(max(a["profiles"],key=lambda n:a["profiles"][n]["validation_drop_db"]))')
  "$PYTHON" scripts/run_magmaw_c16.py --checkpoint "$CHECKPOINT" --output-dir runs/confirm --steps 12000 --batch-size 8 --profiles "$selected" --seed 5678 --evaluation-repeats 5
  code=$?
fi
if [ "$code" -eq 0 ]; then
  "$PYTHON" scripts/report_magmaw_c16.py runs/confirm/metrics.json --output-dir runs/confirm/report > confirm_report.log 2>&1
  code=$?
fi
echo "$code" > exit_code.txt
exit "$code"
