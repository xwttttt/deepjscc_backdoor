#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
exec "${PYTHON:-python3}" -u scripts/run_div2k_two_drop.py \
  --config configs/div2k_two_drop_struct_lmdb_stronger_20260928.toml \
  --clean-checkpoint models/p0_clean.pt \
  --output-root "${1:-runs/stronger_new}" --device "${DEVICE:-cuda:0}"
