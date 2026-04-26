#!/bin/bash
# Massive seed sweep for exp24e (attention, 4 heads, wider MLP 2048-1024-512)
# This is our best single-model architecture (R@10=75.3% with seed 123)
set -e
cd /root/ICLPose-loc

FEATURE_DIR="output/feature_extract/features_radio_dual_128/OldHospital_pilot"
DATASET_DIR="dataset/OldHospital"
OUTPUT_BASE="output/feature_retrieval/pose_regression"
BATCH_SIZE=128
GPU=$1
shift
SEEDS="$@"

for seed in $SEEDS; do
  echo "=== exp24e_attn_h4_wider_seed${seed} on GPU ${GPU} ==="
  python -u feature_retrieval/patch_regressor_v7.py \
    --exp_name "exp24e_attn_h4_wider_seed${seed}" \
    --pool attn --feat both+sum --attn_heads 4 \
    --gpu $GPU --feature_dir "$FEATURE_DIR" \
    --dataset_dir "$DATASET_DIR" --output_base "$OUTPUT_BASE" \
    --patch_dim 128 --hidden_dims 2048 1024 512 \
    --batch_size $BATCH_SIZE \
    --dropout 0.15 --feature_dropout 0.15 \
    --lr 0.001 --weight_decay 0.0001 --epochs 10000 \
    --log_every 10 --seed $seed
  echo "--- seed${seed} done ---"
done
echo "=== GPU${GPU} ALL DONE ==="
