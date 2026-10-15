#!/usr/bin/env bash
set -euo pipefail

repo=/home/gxhen/Wisig/jscc/deepjscc_backdoor
python_bin=/home/gxhen/anaconda3/bin/python
run_root=runs/div2k_lmdb_cv16_20260923
cd "$repo"

test -d data/DIV2K/lmdb/train_256_CV16_float32.lmdb
test -d data/DIV2K/lmdb/valid_256_CV16_float32.lmdb
test -f runs/clean_p0_c16_server_20260827/best.pt
test ! -e "$run_root/p0_full"
test ! -e "$run_root/backdoor_pure_full"

echo "[pipeline] P0 start $(date -Is)"
"$python_bin" -u scripts/run_div2k_clean_p0.py \
  --output-dir "$run_root/p0_full" \
  --steps 5000 --batch-size 4 --validation-interval 250 --mixed-precision
echo "[pipeline] P0 done $(date -Is)"

echo "[pipeline] backdoor start $(date -Is)"
"$python_bin" -u scripts/run_div2k_two_drop.py \
  --config configs/div2k_two_drop_struct_lmdb_pure_256.toml \
  --clean-checkpoint "$run_root/p0_full/best.pt" \
  --output-root "$run_root/backdoor_pure_full"
echo "[pipeline] backdoor done $(date -Is)"
