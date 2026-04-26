#!/bin/bash
# exp24c: 16 heads (6656d input) — GPU 3
set -e
cd /root/ICLPose-loc

FEATURE_DIR="output/feature_extract/features_radio_dual_128/OldHospital_pilot"
DATASET_DIR="dataset/OldHospital"
OUTPUT_BASE="output/feature_retrieval/pose_regression"

for seed in 314 123 456; do
  echo "=== exp24c_attn_h16_seed${seed} ==="
  python -u feature_retrieval/patch_regressor_v7.py \
    --exp_name "exp24c_attn_h16_seed${seed}" \
    --pool attn --feat both+sum --attn_heads 16 \
    --gpu 3 --feature_dir "$FEATURE_DIR" \
    --dataset_dir "$DATASET_DIR" --output_base "$OUTPUT_BASE" \
    --patch_dim 128 --hidden_dims 1024 512 256 \
    --batch_size 128 \
    --dropout 0.15 --feature_dropout 0.15 \
    --lr 0.001 --weight_decay 0.0001 --epochs 10000 \
    --log_every 10 --seed $seed
  echo "--- seed${seed} done ---"
done
echo "=== GPU3 ALL DONE ==="
