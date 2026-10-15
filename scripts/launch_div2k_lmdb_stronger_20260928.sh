#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_dir="runs/div2k_lmdb_cv16_20260928/stronger_struct_60ep_001"
if [[ -e "$run_dir" ]]; then
  echo "Refusing to reuse training output: $run_dir" >&2
  exit 1
fi

exec /home/gxhen/anaconda3/bin/python -u scripts/run_div2k_two_drop.py \
  --config configs/div2k_two_drop_struct_lmdb_stronger_20260928.toml \
  --clean-checkpoint runs/div2k_lmdb_cv16_20260923/p0_full/best.pt \
  --output-root "$run_dir" --device cuda:0
