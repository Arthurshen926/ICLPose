#!/bin/bash
# Verify dropout fix: run same seeds as exp14 with precompute_pool=True (fixed)
# Expected: results should match exp14 (~70% mean, not 67%)
set -e

GPU=0
FEAT_DIR="output/feature_extract/features_radio_dual_128/OldHospital_pilot"
DATASET_DIR="dataset/OldHospital"
OUT_BASE="output/feature_retrieval/pose_regression"
SCRIPT="feature_retrieval/patch_regressor_v7.py"

COMMON="--gpu $GPU --feature_dir $FEAT_DIR --dataset_dir $DATASET_DIR \
  --output_base $OUT_BASE --pool spp --feat both+sum \
  --hidden_dims 2048 1024 512 --patch_dim 128 \
  --dropout 0.15 --feature_dropout 0.15 \
  --lr 0.001 --weight_decay 0.0001 --epochs 5000 \
  --precompute_pool --log_every 10"

run_exp() {
  local name=$1
  shift
  echo ""
  echo "=== $name ==="
  python -u $SCRIPT --exp_name "$name" $COMMON "$@"
  echo "--- $name done ---"
}

echo "============================================================"
echo "DROPOUT FIX VERIFICATION (GPU 0)"
echo "Expected: ~70% mean (was 67% with buggy dropout)"
echo "============================================================"

# Same 5 seeds as the exp14 experiments
for seed in 314 123 456 9999 11111; do
  run_exp "exp22_128d_fixdrop_seed${seed}" --seed $seed
done

# Also run some mass sweep seeds for direct comparison
for seed in 1 2 3 4 5; do
  run_exp "exp22_128d_fixdrop_seed${seed}" --seed $seed
done

echo ""
echo "============================================"
echo "Dropout fix verification complete!"
echo "============================================"
