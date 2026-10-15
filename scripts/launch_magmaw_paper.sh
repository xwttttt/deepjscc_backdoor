#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/deps:$PWD/src"
PYTHON=/home/gxhen/Wisig/jscc/deepjscc_backdoor/.venv-direct-v2/bin/python
CODECS=/home/gxhen/Wisig/jscc/deepjscc_backdoor/experiments/magmaw_image_full_20261013_01/runs/full
set +e
"$PYTHON" scripts/run_magmaw_paper.py --codec-dir "$CODECS" --output-dir runs/diagnosis --steps 400 --batch-size 4 --evaluation-repeats 2
code=$?
if [ "$code" -eq 0 ]; then
  "$PYTHON" scripts/run_magmaw_paper.py --codec-dir "$CODECS" --output-dir runs/literal_full --steps 12000 --batch-size 8 --profiles literal --evaluation-repeats 5
  code=$?
fi
echo "$code" > exit_code.txt
exit "$code"
