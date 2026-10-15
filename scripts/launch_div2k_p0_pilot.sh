#!/usr/bin/env bash
set -euo pipefail

cd /home/gxhen/Wisig/jscc/deepjscc_backdoor
run_dir=runs/div2k_256_20260923
python_bin=/home/gxhen/anaconda3/bin/python
mkdir -p "$run_dir"

for attempt in $(seq 1 240); do
    train_count=0
    valid_count=0
    if [[ -d data/DIV2K/DIV2K_train_HR ]]; then
        train_count=$(find data/DIV2K/DIV2K_train_HR -maxdepth 1 -type f -name '*.png' | wc -l)
    fi
    if [[ -d data/DIV2K/DIV2K_valid_HR ]]; then
        valid_count=$(find data/DIV2K/DIV2K_valid_HR -maxdepth 1 -type f -name '*.png' | wc -l)
    fi
    if [[ "$train_count" -eq 800 && "$valid_count" -eq 100 ]]; then
        break
    fi
    sleep 15
done
if [[ "$train_count" -ne 800 || "$valid_count" -ne 100 ]]; then
    echo "DIV2K download did not finish: train=$train_count valid=$valid_count" >&2
    exit 1
fi

"$python_bin" scripts/prepare_div2k_static.py \
    --source data/DIV2K \
    --output data/DIV2K/prepared_256_seed_20260923 \
    --crop-size 256 --train-crops-per-image 8 --seed 20260923 \
    >"$run_dir/prepare.log" 2>&1

"$python_bin" scripts/run_div2k_clean_p0.py \
    --prepared-dir data/DIV2K/prepared_256_seed_20260923 \
    --init-checkpoint runs/clean_p0_c16_server_20260827/best.pt \
    --output-dir "$run_dir/p0_pilot" \
    --steps 100 --batch-size 2 --validation-interval 50 \
    --mixed-precision \
    >"$run_dir/p0_pilot.log" 2>&1

echo "P0 pilot complete: $run_dir/p0_pilot" 
