#!/bin/bash
# Train more 128d wider models (exp14m2 config was best individual at 69.8%)
# Also try SWA (start averaging after epoch 1500)
set -e
cd /root/ICLPose-loc

FEATURE_DIR="output/feature_extract/features_radio_dual_128/OldHospital_pilot"

echo "=== Training 128d wider with multiple seeds ==="

# Seeds for diversity
SEEDS=(123 456 789 2024 7777 9999)

for SEED in "${SEEDS[@]}"; do
    EXP_NAME="exp14n_128d_wider_seed${SEED}"
    echo "=== $EXP_NAME ==="
    
    python -m feature_retrieval.patch_regressor_v7 \
        --pool spp \
        --feat both+sum \
        --exp_name "$EXP_NAME" \
        --gpu 3 \
        --feature_dir "$FEATURE_DIR" \
        --epochs 3000 \
        --lr 0.001 \
        --weight_decay 0.0001 \
        --dropout 0.15 \
        --feature_dropout 0.15 \
        --hidden_dims 2048 1024 512 \
        --patch_dim 128 \
        --seed "$SEED" \
        2>&1 | tail -20
    
    echo "--- $EXP_NAME done ---"
    echo ""
done

echo "=== All 128d wider training complete! ==="
