#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/deps:$PWD/src"
set +e
/home/gxhen/Wisig/jscc/deepjscc_backdoor/.venv-direct-v2/bin/python scripts/run_ota_adversarial.py \
  --checkpoint /home/gxhen/Wisig/jscc/deepjscc_backdoor/runs/div2k_lmdb_cv16_20260923/p0_full/best.pt \
  --output-dir runs/pilot_eot_evaluated --steps 400 --batch-size 4 --psr-db -16 --snr-db 10 \
  --evaluate-generator runs/pilot_eot/generator.pt
code=$?
echo "$code" > evaluation_exit_code.txt
exit "$code"
