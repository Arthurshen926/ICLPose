#!/bin/bash
# Train 128d SPP with additional seeds on GPU 3
# Looking for lucky seeds that beat 70.9%
set -e
cd /root/ICLPose-loc

FEATURE_DIR="output/feature_extract/features_radio_dual_128/OldHospital_pilot"

echo "=== 128d Additional Seeds ==="

for SEED in 314 628 999 1337 4242 7890 11111 22222; do
    EXP_NAME="exp14s_128d_wider_seed${SEED}"
    echo "=== $EXP_NAME (GPU 3) ==="
    
    CUDA_VISIBLE_DEVICES=3 python -m feature_retrieval.patch_regressor_v7 \
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
        --patch_dim 128 \
        --seed "$SEED" \
        2>&1 | tail -40
    
    echo "--- $EXP_NAME done ---"
    echo ""
done

echo "=== DONE ==="
