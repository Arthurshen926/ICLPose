#!/bin/bash
# Cross-Architecture 128d Training for Ensemble Diversity
# =========================================================
# Train models with different pooling types (GAP, GeM, Attn) at 128d
# These have NEVER been tested with 128d features — only 64d.
# Different pooling encodes spatial info differently → true structural diversity.
#
# Feature dims:
#   GAP:  128+128+2560 = 2816d  (much smaller → less overfitting)
#   GeM:  128+128+2560 = 2816d  (learnable p parameter)
#   Attn: 128*4+128*4+2560 = 3584d (learned spatial attention)
#   SPP:  128*21+128*21+2560 = 7936d (current best, for reference)

set -e

GPU=3
FEAT_DIR="output/feature_extract/features_radio_dual_128/OldHospital_pilot"
DATASET_DIR="dataset/OldHospital"
# For GAP/GeM/SPP (no/minimal learnable pooling params) → precompute OK
BASE_PRECOMP="python -m feature_retrieval.patch_regressor_v7 --gpu ${GPU} --feature_dir ${FEAT_DIR} --dataset_dir ${DATASET_DIR} --patch_dim 128 --feat both+sum --epochs 5000 --lr 0.001 --weight_decay 0.0001 --precompute_pool"
# For Attn/Conv (learnable pooling params) → must NOT precompute
BASE_LEARNABLE="python -m feature_retrieval.patch_regressor_v7 --gpu ${GPU} --feature_dir ${FEAT_DIR} --dataset_dir ${DATASET_DIR} --patch_dim 128 --feat both+sum --epochs 5000 --lr 0.001 --weight_decay 0.0001"

echo "============================================================"
echo "Cross-Architecture 128d Experiments"
echo "============================================================"
echo ""

# --- GAP Pooling (2816d input) ---
# Smaller input → less overfitting with 895 samples
for SEED in 314 123 456 789; do
    echo "=== exp16a_128d_gap_seed${SEED} ==="
    ${BASE_PRECOMP} --pool gap --hidden_dims 1024 512 256 --dropout 0.15 --feature_dropout 0.15 \
        --seed ${SEED} --exp_name exp16a_128d_gap_seed${SEED}
    echo "--- exp16a_128d_gap_seed${SEED} done ---"
    echo ""
done

# --- GAP Wider (2816d input, wider MLP) ---
for SEED in 314 123; do
    echo "=== exp16a2_128d_gap_wider_seed${SEED} ==="
    ${BASE_PRECOMP} --pool gap --hidden_dims 2048 1024 512 --dropout 0.15 --feature_dropout 0.15 \
        --seed ${SEED} --exp_name exp16a2_128d_gap_wider_seed${SEED}
    echo "--- exp16a2_128d_gap_wider_seed${SEED} done ---"
    echo ""
done

# --- GeM Pooling (2816d input, learnable power parameter) ---
# NOTE: GeM's learnable p is just 1 scalar, precompute is OK (p=3.0 init is fine)
for SEED in 314 123 456; do
    echo "=== exp16b_128d_gem_seed${SEED} ==="
    ${BASE_PRECOMP} --pool gem --hidden_dims 1024 512 256 --dropout 0.15 --feature_dropout 0.15 \
        --seed ${SEED} --exp_name exp16b_128d_gem_seed${SEED}
    echo "--- exp16b_128d_gem_seed${SEED} done ---"
    echo ""
done

# --- GeM Wider ---
for SEED in 314 123; do
    echo "=== exp16b2_128d_gem_wider_seed${SEED} ==="
    ${BASE_PRECOMP} --pool gem --hidden_dims 2048 1024 512 --dropout 0.15 --feature_dropout 0.15 \
        --seed ${SEED} --exp_name exp16b2_128d_gem_wider_seed${SEED}
    echo "--- exp16b2_128d_gem_wider_seed${SEED} done ---"
    echo ""
done

# --- Attention Pooling (3584d input, learned spatial attention) ---
# NOTE: Attn has learnable conv layers → CANNOT precompute
for SEED in 314 123 456; do
    echo "=== exp16c_128d_attn_seed${SEED} ==="
    ${BASE_LEARNABLE} --pool attn --hidden_dims 1024 512 256 --dropout 0.15 --feature_dropout 0.15 \
        --seed ${SEED} --exp_name exp16c_128d_attn_seed${SEED}
    echo "--- exp16c_128d_attn_seed${SEED} done ---"
    echo ""
done

# --- Attention Wider ---
for SEED in 314 123; do
    echo "=== exp16c2_128d_attn_wider_seed${SEED} ==="
    ${BASE_LEARNABLE} --pool attn --hidden_dims 2048 1024 512 --dropout 0.15 --feature_dropout 0.15 \
        --seed ${SEED} --exp_name exp16c2_128d_attn_wider_seed${SEED}
    echo "--- exp16c2_128d_attn_wider_seed${SEED} done ---"
    echo ""
done

# --- Wing Loss variants (different loss → different error distribution for ensemble) ---
# SPP + wing loss
for SEED in 314 123 456; do
    echo "=== exp16d_128d_spp_wing_seed${SEED} ==="
    ${BASE_PRECOMP} --pool spp --hidden_dims 2048 1024 512 --dropout 0.15 --feature_dropout 0.15 \
        --trans_loss wing --seed ${SEED} --exp_name exp16d_128d_spp_wing_seed${SEED}
    echo "--- exp16d_128d_spp_wing_seed${SEED} done ---"
    echo ""
done

# --- Conv Encoder (learned spatial reduction) ---
# NOTE: Conv has learnable layers → CANNOT precompute
for SEED in 314 123; do
    echo "=== exp16e_128d_conv_seed${SEED} ==="
    ${BASE_LEARNABLE} --pool conv --hidden_dims 1024 512 256 --dropout 0.15 --feature_dropout 0.15 \
        --seed ${SEED} --exp_name exp16e_128d_conv_seed${SEED}
    echo "--- exp16e_128d_conv_seed${SEED} done ---"
    echo ""
done

echo ""
echo "=== ALL CROSS-ARCHITECTURE EXPERIMENTS DONE ==="
