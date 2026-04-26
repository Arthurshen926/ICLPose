#!/bin/bash
# Spatial Feature Interpolation Augmentation Experiments
# =====================================================
# Unlike random mixup (57.1% - FAILED), spatial interpolation only mixes
# features between NEARBY training images (within 5m), creating plausible
# virtual viewpoints that fill coverage gaps in the training set.
#
# Key insight from error analysis:
# - 43% of test images have no training image within 2m
# - Failures cluster in sparsely covered areas
# - Spatial interpolation creates synthetic samples in these gaps
set -e

GPU=5  # Will run after snapshot ensemble finishes
FEAT_DIR="output/feature_extract/features_radio_dual_128/OldHospital_pilot"
DATASET_DIR="dataset/OldHospital"
BASE="python -m feature_retrieval.patch_regressor_v7 --gpu ${GPU} --feature_dir ${FEAT_DIR} --dataset_dir ${DATASET_DIR} --patch_dim 128 --feat both+sum --pool spp --hidden_dims 2048 1024 512 --epochs 5000 --lr 0.001 --weight_decay 0.0001 --precompute_pool --dropout 0.15 --feature_dropout 0.15"

echo "============================================================"
echo "Spatial Feature Interpolation Experiments"
echo "============================================================"
echo ""

# --- Baseline config for comparison ---
# k=5, alpha_max=0.3, dist=5m, prob=0.5
for SEED in 314 123 456; do
    echo "=== exp18a_128d_spatiaug_k5_a03_seed${SEED} ==="
    ${BASE} --spatial_aug_k 5 --spatial_aug_alpha 0.3 --spatial_aug_dist 5.0 --spatial_aug_prob 0.5 \
        --seed ${SEED} --exp_name exp18a_128d_spatiaug_k5_a03_seed${SEED}
    echo "--- exp18a_128d_spatiaug_k5_a03_seed${SEED} done ---"
    echo ""
done

# --- Larger radius (10m) ---
for SEED in 314 123; do
    echo "=== exp18b_128d_spatiaug_k5_a03_d10_seed${SEED} ==="
    ${BASE} --spatial_aug_k 5 --spatial_aug_alpha 0.3 --spatial_aug_dist 10.0 --spatial_aug_prob 0.5 \
        --seed ${SEED} --exp_name exp18b_128d_spatiaug_k5_a03_d10_seed${SEED}
    echo "--- exp18b_128d_spatiaug_k5_a03_d10_seed${SEED} done ---"
    echo ""
done

# --- Smaller alpha (more conservative interpolation) ---
for SEED in 314 123; do
    echo "=== exp18c_128d_spatiaug_k5_a01_seed${SEED} ==="
    ${BASE} --spatial_aug_k 5 --spatial_aug_alpha 0.1 --spatial_aug_dist 5.0 --spatial_aug_prob 0.5 \
        --seed ${SEED} --exp_name exp18c_128d_spatiaug_k5_a01_seed${SEED}
    echo "--- exp18c_128d_spatiaug_k5_a01_seed${SEED} done ---"
    echo ""
done

# --- Higher augmentation probability ---
for SEED in 314 123; do
    echo "=== exp18d_128d_spatiaug_k5_a03_p08_seed${SEED} ==="
    ${BASE} --spatial_aug_k 5 --spatial_aug_alpha 0.3 --spatial_aug_dist 5.0 --spatial_aug_prob 0.8 \
        --seed ${SEED} --exp_name exp18d_128d_spatiaug_k5_a03_p08_seed${SEED}
    echo "--- exp18d_128d_spatiaug_k5_a03_p08_seed${SEED} done ---"
    echo ""
done

# --- More neighbors (k=10) ---
for SEED in 314 123; do
    echo "=== exp18e_128d_spatiaug_k10_a03_seed${SEED} ==="
    ${BASE} --spatial_aug_k 10 --spatial_aug_alpha 0.3 --spatial_aug_dist 5.0 --spatial_aug_prob 0.5 \
        --seed ${SEED} --exp_name exp18e_128d_spatiaug_k10_a03_seed${SEED}
    echo "--- exp18e_128d_spatiaug_k10_a03_seed${SEED} done ---"
    echo ""
done

echo ""
echo "=== ALL SPATIAL AUGMENTATION EXPERIMENTS DONE ==="
