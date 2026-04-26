#!/bin/bash
# Train 128d attention pooling models on GPU 3
set -e
cd /root/ICLPose-loc

FEATURE_DIR="output/feature_extract/features_radio_dual_128/OldHospital_pilot"

echo "=== 128d Attention Pooling Training ==="

# Attention pooling with 8 heads (like exp14f but on 128d)
for SEED in 123 456 789; do
    EXP_NAME="exp14r_128d_attn_seed${SEED}"
    echo "=== $EXP_NAME (GPU 3) ==="
    
    CUDA_VISIBLE_DEVICES=3 python -m feature_retrieval.patch_regressor_v7 \
        --pool attn \
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
        --patch_dim 128 \
        --seed "$SEED" \
        --attn_heads 8 \
        2>&1 | tail -30
    
    echo "--- $EXP_NAME done ---"
    echo ""
done

# Also try 256d attention pooling  
FEATURE_DIR_256="output/feature_extract/features_radio_dual_256/OldHospital_pilot"

for SEED in 123 456; do
    EXP_NAME="exp14r2_256d_attn_seed${SEED}"
    echo "=== $EXP_NAME (GPU 3) ==="
    
    CUDA_VISIBLE_DEVICES=3 python -m feature_retrieval.patch_regressor_v7 \
        --pool attn \
        --feat both+sum \
        --exp_name "$EXP_NAME" \
        --gpu 0 \
        --feature_dir "$FEATURE_DIR_256" \
        --epochs 5000 \
        --lr 0.001 \
        --weight_decay 0.0001 \
        --dropout 0.15 \
        --feature_dropout 0.15 \
        --hidden_dims 2048 1024 512 \
        --patch_dim 256 \
        --seed "$SEED" \
        --attn_heads 8 \
        2>&1 | tail -30
    
    echo "--- $EXP_NAME done ---"
    echo ""
done

echo "=== All attention pooling training complete! ==="
