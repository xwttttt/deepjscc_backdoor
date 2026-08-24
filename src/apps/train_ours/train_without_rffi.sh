#!/bin/bash
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate FedSem

# ==================================================
# AWGN
# ==================================================
python apps/train_ours/train.py \
  --snr_curriculum \
  --snr_curriculum_stages 20:20,10:20,0:10 \
  --snr_curriculum_epochs 200 300 500 \
  --curriculum_lr_boundary_decay 0.5 \
  --tokenizer_type CV \
  --epochs 1000 \
  --snr_min 0 \
  --snr_max 20 \
  --patience 400 \
  --lr 3e-4 \
  --beta1 0.9 \
  --grad_clip 1.0 \
  --channel_type "awgn" \
  --target_m 1 \
  --target_m_choices 1 2 3 4 \
  --exp_name awgn_codec_without_impairment


python apps/train_ours/train.py \
  --snr_curriculum \
  --snr_curriculum_stages 20:20,10:20,0:10 \
  --snr_curriculum_epochs 200 300 500 \
  --curriculum_lr_boundary_decay 0.5 \
  --tokenizer_type CV \
  --epochs 1000 \
  --snr_min 0 \
  --snr_max 20 \
  --patience 400 \
  --lr 3e-4 \
  --beta1 0.9 \
  --grad_clip 1.0 \
  --channel_type "awgn" \
  --target_m 1 \
  --target_m_choices 1 2 3 4 \
  --disable_encoder_conditioning \
  --exp_name awgn_decoder_without_impairment


# ==================================================
# Rayleigh
# ==================================================
python apps/train_ours/train.py \
  --snr_curriculum \
  --snr_curriculum_stages 20:20,10:20,0:10 \
  --snr_curriculum_epochs 200 300 500 \
  --curriculum_lr_boundary_decay 0.5 \
  --tokenizer_type CV \
  --epochs 1000 \
  --snr_min 0 \
  --snr_max 20 \
  --patience 400 \
  --lr 3e-4 \
  --beta1 0.9 \
  --grad_clip 1.0 \
  --channel_type "rayleigh" \
  --rayleigh_apply_amp \
  --rayleigh_apply_phi \
  --target_m 1 \
  --target_m_choices 1 2 3 4 \
  --exp_name rayleigh_codec_without_impairment


python apps/train_ours/train.py \
  --snr_curriculum \
  --snr_curriculum_stages 20:20,10:20,0:10 \
  --snr_curriculum_epochs 200 300 500 \
  --curriculum_lr_boundary_decay 0.5 \
  --tokenizer_type CV \
  --epochs 1000 \
  --snr_min 0 \
  --snr_max 20 \
  --patience 400 \
  --lr 3e-4 \
  --beta1 0.9 \
  --grad_clip 1.0 \
  --channel_type "rayleigh" \
  --rayleigh_apply_amp \
  --rayleigh_apply_phi \
  --target_m 1 \
  --target_m_choices 1 2 3 4 \
  --disable_encoder_conditioning \
  --exp_name rayleigh_decoder_without_impairment
