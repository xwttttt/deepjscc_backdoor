#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/deps:$PWD/src"
PYTHON=/home/gxhen/Wisig/jscc/deepjscc_backdoor/.venv-direct-v2/bin/python
set +e
"$PYTHON" -m pytest -c /dev/null --rootdir . tests/test_magmaw_full.py -q > contract_tests.log 2>&1
code=$?
if [ "$code" -eq 0 ]; then
  "$PYTHON" scripts/run_magmaw_image_full.py --config configs/magmaw_image_full.toml --output-dir runs/smoke --smoke
  code=$?
fi
if [ "$code" -eq 0 ]; then
  "$PYTHON" scripts/run_magmaw_image_full.py --config configs/magmaw_image_full.toml --output-dir runs/full
  code=$?
fi
echo "$code" > exit_code.txt
exit "$code"
