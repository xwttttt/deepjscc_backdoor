#!/bin/bash
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate sionna

python scripts/make_train_lmdb_dataset.py \
    --tokenizer_type CV \
    --spatial_compression 16 \
    --dataset_root ~/datasets \
    --output_dir ./datasets/DIV2K