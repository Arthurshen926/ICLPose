#!/bin/bash
# exp24e: 4 heads + wider MLP (2048-1024-512) — GPU 5
set -e
cd /root/ICLPose-loc

FEATURE_DIR="output/feature_extract/features_radio_dual_128/OldHospital_pilot"
DATASET_DIR="dataset/OldHospital"
OUTPUT_BASE="output/feature_retrieval/pose_regression"

for seed in 314 123 456 42 7; do
  echo "=== exp24e_attn_h4_wider_seed${seed} ==="
  python -u feature_retrieval/patch_regressor_v7.py \
    --exp_name "exp24e_attn_h4_wider_seed${seed}" \
    --pool attn --feat both+sum --attn_heads 4 \
    --gpu 5 --feature_dir "$FEATURE_DIR" \
    --dataset_dir "$DATASET_DIR" --output_base "$OUTPUT_BASE" \
    --patch_dim 128 --hidden_dims 2048 1024 512 \
    --batch_size 128 \
    --dropout 0.15 --feature_dropout 0.15 \
    --lr 0.001 --weight_decay 0.0001 --epochs 10000 \
    --log_every 10 --seed $seed
  echo "--- seed${seed} done ---"
done
echo "=== GPU5 ALL DONE ==="
