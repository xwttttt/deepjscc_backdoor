#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/deps:$PWD/src"
PYTHON=/home/gxhen/Wisig/jscc/deepjscc_backdoor/.venv-direct-v2/bin/python
TARGET=/home/gxhen/Wisig/jscc/deepjscc_backdoor/runs/div2k_lmdb_cv16_20260923/p0_full/best.pt
PGM=/home/gxhen/Wisig/jscc/deepjscc_backdoor/experiments/magmaw_c16_20261014_01/runs/primary/psnr_best.pt
set +e
"$PYTHON" scripts/diagnose_magmaw_c16_uap.py --checkpoint "$TARGET" --pgm-checkpoint "$PGM" --output-dir runs/full --steps 800 --evaluation-repeats 3
code=$?
if [ "$code" -eq 0 ]; then
  "$PYTHON" scripts/report_magmaw_c16_uap_diagnosis.py runs/full/metrics.json --output-dir runs/full/report > report.log 2>&1
  code=$?
fi
echo "$code" > exit_code.txt
exit "$code"
