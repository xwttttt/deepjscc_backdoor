#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/deps:$PWD/src"
PYTHON=/home/gxhen/Wisig/jscc/deepjscc_backdoor/.venv-direct-v2/bin/python
CODECS=/home/gxhen/Wisig/jscc/deepjscc_backdoor/experiments/magmaw_image_full_20261013_01/runs/full
set +e
"$PYTHON" scripts/run_magmaw_effective.py --codec-dir "$CODECS" --output-dir runs/primary --steps 12000 --batch-size 8 --profiles distortion_only psnr_only --evaluation-repeats 5 --seed 1234
code=$?
if [ "$code" -eq 0 ]; then
  "$PYTHON" scripts/report_magmaw_effective.py runs/primary/metrics.json --output-dir runs/primary/report > primary_report.log 2>&1
  code=$?
fi
if [ "$code" -eq 0 ]; then
  "$PYTHON" scripts/run_magmaw_effective.py --codec-dir "$CODECS" --output-dir runs/confirm --steps 12000 --batch-size 8 --profiles psnr_only --evaluation-repeats 5 --seed 5678
  code=$?
fi
if [ "$code" -eq 0 ]; then
  "$PYTHON" scripts/report_magmaw_effective.py runs/confirm/metrics.json --output-dir runs/confirm/report > confirm_report.log 2>&1
  code=$?
fi
echo "$code" > exit_code.txt
exit "$code"
