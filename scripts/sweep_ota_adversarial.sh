#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/deps:$PWD/src"
PYTHON=/home/gxhen/Wisig/jscc/deepjscc_backdoor/.venv-direct-v2/bin/python
set +e
code=0
for budget in -20 -10; do
  "$PYTHON" scripts/run_ota_adversarial.py \
    --checkpoint /home/gxhen/Wisig/jscc/deepjscc_backdoor/runs/div2k_lmdb_cv16_20260923/p0_full/best.pt \
    --output-dir "runs/pilot_eot_psr${budget}" --steps 400 --batch-size 4 --psr-db "$budget" --snr-db 10 \
    --evaluate-generator runs/pilot_eot/generator.pt
  code=$?
  [ "$code" -eq 0 ] || break
done
echo "$code" > sweep_exit_code.txt
exit "$code"
