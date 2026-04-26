#!/bin/bash
# Hard Example Mining experiments on GPU 0
# Tests: OHEM (keep top K% hardest) and Focal-style loss weighting
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

# =============================================
# OHEM: keep only top K% hardest samples
# =============================================

# OHEM 50% (keep top half) - 3 seeds
for seed in 314 123 456; do
  run_exp "exp21a_128d_ohem50_seed${seed}" --ohem_ratio 0.5 --seed $seed
done

# OHEM 30% (keep top 30% hardest) - more aggressive
for seed in 314 123; do
  run_exp "exp21b_128d_ohem30_seed${seed}" --ohem_ratio 0.3 --seed $seed
done

# OHEM 70% (keep top 70% - mild)
run_exp "exp21c_128d_ohem70_seed314" --ohem_ratio 0.7 --seed 314

# =============================================
# Focal-style weighting: upweight hard samples
# =============================================

# Focal gamma=1 (moderate focus on hard samples)
for seed in 314 123 456; do
  run_exp "exp21d_128d_focal1_seed${seed}" --focal_gamma 1.0 --seed $seed
done

# Focal gamma=2 (strong focus on hard samples)
run_exp "exp21e_128d_focal2_seed314" --focal_gamma 2.0 --seed 314

# Focal gamma=0.5 (mild focus)
run_exp "exp21f_128d_focal05_seed314" --focal_gamma 0.5 --seed 314

echo ""
echo "============================================"
echo "All Hard Example Mining experiments complete!"
echo "============================================"
