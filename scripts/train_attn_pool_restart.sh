#!/bin/bash
# Restart attention pooling training for best configs with multiple seeds
# GPU 0: exp24a (4 heads) - best R@10/2m among attn models
# GPU 1: exp24c (16 heads) - best R@5/1m
# GPU 3: exp24e (4h wider) - best raw metrics
set -e
cd /root/ICLPose-loc

FEATURE_DIR="output/feature_extract/features_radio_dual_128/OldHospital_pilot"
DATASET_DIR="dataset/OldHospital"
OUTPUT_BASE="output/feature_retrieval/pose_regression"
BATCH_SIZE=128

GPU=$1
CONFIG=$2  # a, c, or e

case $CONFIG in
  a)
    HEADS=4
    HIDDEN="1024 512 256"
    PREFIX="exp24a_attn_h4"
    ;;
  c)
    HEADS=16
    HIDDEN="1024 512 256"
    PREFIX="exp24c_attn_h16"
    ;;
  e)
    HEADS=4
    HIDDEN="2048 1024 512"
    PREFIX="exp24e_attn_h4_wider"
    ;;
esac

for seed in 123 456 42 7 999; do
  echo "=== ${PREFIX}_seed${seed} on GPU ${GPU} ==="
  python -u feature_retrieval/patch_regressor_v7.py \
    --exp_name "${PREFIX}_seed${seed}" \
    --pool attn --feat both+sum --attn_heads $HEADS \
    --gpu $GPU --feature_dir "$FEATURE_DIR" \
    --dataset_dir "$DATASET_DIR" --output_base "$OUTPUT_BASE" \
    --patch_dim 128 --hidden_dims $HIDDEN \
    --batch_size $BATCH_SIZE \
    --dropout 0.15 --feature_dropout 0.15 \
    --lr 0.001 --weight_decay 0.0001 --epochs 10000 \
    --log_every 10 --seed $seed
  echo "--- seed${seed} done ---"
done
echo "=== GPU${GPU} ${CONFIG} ALL DONE ==="
