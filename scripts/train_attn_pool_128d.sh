#!/bin/bash
# ==============================================================================
# Experiment 24: Attention Pooling End-to-End (128d features)
# ==============================================================================
# Fixed: use batch_size=128 to avoid OOM during backward pass
# Full-batch (895) OOMs at ~22.5 GB; mini-batch (128) should use ~15 GB
# ==============================================================================

set -e
cd /root/ICLPose-loc

FEATURE_DIR="output/feature_extract/features_radio_dual_128/OldHospital_pilot"
DATASET_DIR="dataset/OldHospital"
OUTPUT_BASE="output/feature_retrieval/pose_regression"
BATCH_SIZE=128  # Mini-batch to avoid OOM with attention pooling

# GPU 0: exp24a — 4 heads (3584d), standard MLP 1024-512-256
for seed in 314 123 456 42 7; do
  echo "=== exp24a_attn_h4_seed${seed} ==="
  python -u feature_retrieval/patch_regressor_v7.py \
    --exp_name "exp24a_attn_h4_seed${seed}" \
    --pool attn --feat both+sum --attn_heads 4 \
    --gpu 0 --feature_dir "$FEATURE_DIR" \
    --dataset_dir "$DATASET_DIR" --output_base "$OUTPUT_BASE" \
    --patch_dim 128 --hidden_dims 1024 512 256 \
    --batch_size $BATCH_SIZE \
    --dropout 0.15 --feature_dropout 0.15 \
    --lr 0.001 --weight_decay 0.0001 --epochs 10000 \
    --log_every 10 --seed $seed
  echo "--- seed${seed} done ---"
done
echo "=== GPU0 ALL DONE ==="
