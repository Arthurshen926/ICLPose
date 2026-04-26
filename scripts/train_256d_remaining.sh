#!/bin/bash
# Train remaining 256d models (seed123 already done)
set -e
cd /root/ICLPose-loc

FEATURE_DIR="output/feature_extract/features_radio_dual_256/OldHospital_pilot"

# Wider: remaining seeds
for SEED in 456 789 42 2024; do
    EXP_NAME="exp14q_256d_wider_seed${SEED}"
    echo "=== $EXP_NAME (GPU 5) ==="
    
    CUDA_VISIBLE_DEVICES=5 python -m feature_retrieval.patch_regressor_v7 \
        --pool spp \
        --feat both+sum \
        --exp_name "$EXP_NAME" \
        --gpu 0 \
        --feature_dir "$FEATURE_DIR" \
        --epochs 5000 \
        --lr 0.001 \
        --weight_decay 0.0001 \
        --dropout 0.15 \
        --feature_dropout 0.15 \
        --hidden_dims 2048 1024 512 \
        --patch_dim 256 \
        --seed "$SEED" \
        --precompute_pool \
        2>&1 | tail -40
    
    echo "--- $EXP_NAME done ---"
    echo ""
done

# Extra-wide: [4096,2048,1024]
for SEED in 123 456; do
    EXP_NAME="exp14q2_256d_extrawide_seed${SEED}"
    echo "=== $EXP_NAME (GPU 5) ==="
    
    CUDA_VISIBLE_DEVICES=5 python -m feature_retrieval.patch_regressor_v7 \
        --pool spp \
        --feat both+sum \
        --exp_name "$EXP_NAME" \
        --gpu 0 \
        --feature_dir "$FEATURE_DIR" \
        --epochs 5000 \
        --lr 0.001 \
        --weight_decay 0.0001 \
        --dropout 0.15 \
        --feature_dropout 0.15 \
        --hidden_dims 4096 2048 1024 \
        --patch_dim 256 \
        --seed "$SEED" \
        --precompute_pool \
        2>&1 | tail -40
    
    echo "--- $EXP_NAME done ---"
    echo ""
done

# Coarse-only
for SEED in 123 456; do
    EXP_NAME="exp14q3_256d_coarse_seed${SEED}"
    echo "=== $EXP_NAME (GPU 5) ==="
    
    CUDA_VISIBLE_DEVICES=5 python -m feature_retrieval.patch_regressor_v7 \
        --pool spp \
        --feat coarse+sum \
        --exp_name "$EXP_NAME" \
        --gpu 0 \
        --feature_dir "$FEATURE_DIR" \
        --epochs 5000 \
        --lr 0.001 \
        --weight_decay 0.0001 \
        --dropout 0.15 \
        --feature_dropout 0.15 \
        --hidden_dims 2048 1024 512 \
        --patch_dim 256 \
        --seed "$SEED" \
        --precompute_pool \
        2>&1 | tail -40
    
    echo "--- $EXP_NAME done ---"
    echo ""
done

echo "=== DONE ==="
