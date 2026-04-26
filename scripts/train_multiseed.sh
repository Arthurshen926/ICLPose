#!/bin/bash
# Multi-seed training for ensemble: exp14e config with different random seeds
# Run on GPU 3
set -e

cd /root/ICLPose-loc

SEEDS=(42 123 456 789 2024 7777)

echo "=== Multi-seed ensemble training ==="
echo "Config: SPP(1,2,4), both+sum, hidden=[1024,512,256], 3000 epochs"
echo "Seeds: ${SEEDS[@]}"
echo ""

for SEED in "${SEEDS[@]}"; do
    EXP_NAME="exp14k_seed${SEED}"
    echo "=== Training seed $SEED -> $EXP_NAME ==="
    
    python -m feature_retrieval.patch_regressor_v7 \
        --pool spp \
        --feat both+sum \
        --exp_name "$EXP_NAME" \
        --gpu 3 \
        --epochs 3000 \
        --lr 0.001 \
        --weight_decay 0.0001 \
        --dropout 0.1 \
        --feature_dropout 0.1 \
        --hidden_dims 1024 512 256 \
        --seed "$SEED" \
        2>&1 | tail -20
    
    echo ""
    echo "--- Seed $SEED done ---"
    echo ""
done

# Also train a smaller MLP variant with 2 seeds
echo "=== Smaller MLP (512,256,128) training ==="
SMALL_SEEDS=(42 123)

for SEED in "${SMALL_SEEDS[@]}"; do
    EXP_NAME="exp14l_small_seed${SEED}"
    echo "=== Training small MLP seed $SEED -> $EXP_NAME ==="
    
    python -m feature_retrieval.patch_regressor_v7 \
        --pool spp \
        --feat both+sum \
        --exp_name "$EXP_NAME" \
        --gpu 3 \
        --epochs 5000 \
        --lr 0.001 \
        --weight_decay 0.0001 \
        --dropout 0.05 \
        --feature_dropout 0.05 \
        --hidden_dims 512 256 128 \
        --seed "$SEED" \
        2>&1 | tail -20
    
    echo ""
    echo "--- Small MLP seed $SEED done ---"
    echo ""
done

echo "=== All training complete! ==="
echo "Models saved to output/feature_retrieval/pose_regression/exp14k_seed* and exp14l_small_seed*"
