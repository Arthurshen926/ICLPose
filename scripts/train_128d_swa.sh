#!/bin/bash
# Train 128d wider models with SWA on GPU 5
set -e
cd /root/ICLPose-loc

FEATURE_DIR="output/feature_extract/features_radio_dual_128/OldHospital_pilot"

echo "=== SWA Training on 128d features ==="

# SWA with wider model (best config)
for SEED in 42 123 456 789; do
    EXP_NAME="exp14p_128d_swa_seed${SEED}"
    echo "=== $EXP_NAME ==="
    
    python -m feature_retrieval.patch_regressor_v7 \
        --pool spp \
        --feat both+sum \
        --exp_name "$EXP_NAME" \
        --gpu 5 \
        --feature_dir "$FEATURE_DIR" \
        --epochs 5000 \
        --lr 0.001 \
        --weight_decay 0.0001 \
        --dropout 0.15 \
        --feature_dropout 0.15 \
        --hidden_dims 2048 1024 512 \
        --patch_dim 128 \
        --seed "$SEED" \
        --swa_start 1000 \
        --swa_freq 50 \
        2>&1 | tail -20
    
    echo "--- $EXP_NAME done ---"
    echo ""
done

# Also try with default architecture + SWA
for SEED in 42 123; do
    EXP_NAME="exp14p2_128d_swa_default_seed${SEED}"
    echo "=== $EXP_NAME ==="
    
    python -m feature_retrieval.patch_regressor_v7 \
        --pool spp \
        --feat both+sum \
        --exp_name "$EXP_NAME" \
        --gpu 5 \
        --feature_dir "$FEATURE_DIR" \
        --epochs 5000 \
        --lr 0.001 \
        --weight_decay 0.0001 \
        --dropout 0.1 \
        --feature_dropout 0.1 \
        --hidden_dims 1024 512 256 \
        --patch_dim 128 \
        --seed "$SEED" \
        --swa_start 1000 \
        --swa_freq 50 \
        2>&1 | tail -20
    
    echo "--- $EXP_NAME done ---"
    echo ""
done

echo "=== All SWA training complete! ==="
