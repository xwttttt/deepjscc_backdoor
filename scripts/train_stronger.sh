#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
output="${1:-runs/three_stage_new}"
"${PYTHON:-python3}" -u apps/train_clean/main.py \
  --output-dir "$output/clean" --device "${DEVICE:-cuda:0}"
exec "${PYTHON:-python3}" -u apps/train_triggered/main.py \
  --config configs/div2k_two_drop_struct_lmdb_stronger_20260928.toml \
  --clean-checkpoint "$output/clean/stage2_best.pt" \
  --output-root "$output/triggered" --device "${DEVICE:-cuda:0}"
