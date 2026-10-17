#!/usr/bin/env bash
# Run the provided maker, not a substitute image preprocessing implementation.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ctjscc="${1:?Usage: prepare_lmdb.sh CTJSCC_ROOT [NEW_OUTPUT_DIR] [maker options]}"; shift
ctjscc="$(cd "$ctjscc" && pwd)"
out="${1:-$root/data/DIV2K/lmdb}"
if (($#)); then shift; fi
if [[ ! -d "$ctjscc/lib/cosmos_tokenizer" ]]; then echo "Missing CTJSCC lib/cosmos_tokenizer" >&2; exit 1; fi
if [[ -e "$out" ]]; then echo "Refusing existing LMDB output (maker can erase old records): $out" >&2; exit 1; fi
export PYTHONPATH="$ctjscc:$root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" "$root/src/src/make_train_lmdb_dataset.py" "$@" \
  --dataset_root "${DATASET_ROOT:-$root/data}" --output_dir "$out" \
  --tokenizer_type CV --spatial_compression 16 --image_size 256 --num_crops 4 --dtype float32
