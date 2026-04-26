#!/bin/bash
set -e
cd /root/ICLPose-loc

FEATURE_DIR="output/feature_extract/features_radio_dual_128/OldHospital_pilot"
DATASET_DIR="dataset/OldHospital"
OUTPUT_BASE="output/feature_retrieval/pose_regression"
BATCH_SIZE=128
GPU=$1

case $2 in
  a)
    HEADS=4
    HIDDEN="1024 512 256"
    PREFIX="exp24a_attn_h4_seed314_full"
    ;;
  c)
    HEADS=16
    HIDDEN="1024 512 256"
    PREFIX="exp24c_attn_h16_seed314_full"
    ;;
  e)
    HEADS=4
    HIDDEN="2048 1024 512"
    PREFIX="exp24e_attn_h4_wider_seed314_full"
    ;;
esac

echo "=== ${PREFIX} on GPU ${GPU} ==="
python -u feature_retrieval/patch_regressor_v7.py \
  --exp_name "${PREFIX}" \
  --pool attn --feat both+sum --attn_heads $HEADS \
  --gpu $GPU --feature_dir "$FEATURE_DIR" \
  --dataset_dir "$DATASET_DIR" --output_base "$OUTPUT_BASE" \
  --patch_dim 128 --hidden_dims $HIDDEN \
  --batch_size $BATCH_SIZE \
  --dropout 0.15 --feature_dropout 0.15 \
  --lr 0.001 --weight_decay 0.0001 --epochs 10000 \
  --log_every 10 --seed 314
echo "=== ${PREFIX} DONE ==="
