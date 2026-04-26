#!/bin/bash
# Training on 128d PCA features
# Run on GPU 3 (multi-seed) and GPU 5 (additional configs)
set -e

cd /root/ICLPose-loc

FEATURE_DIR="output/feature_extract/features_radio_dual_128/OldHospital_pilot"

echo "=== Training on 128d PCA features ==="
echo "Feature dir: $FEATURE_DIR"
echo ""

# exp14m: Same architecture as exp14e but with 128d features
# Input: 128 * 21 * 2 + 2560 = 7936d
echo "=== exp14m: SPP(1,2,4) both+sum 128d, hidden=[1024,512,256] ==="
python -m feature_retrieval.patch_regressor_v7 \
    --pool spp \
    --feat both+sum \
    --exp_name exp14m_128d_spp \
    --gpu 3 \
    --feature_dir "$FEATURE_DIR" \
    --epochs 3000 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --dropout 0.1 \
    --feature_dropout 0.1 \
    --hidden_dims 1024 512 256 \
    --patch_dim 128 \
    --seed 42 \
    2>&1 | tail -25
echo "--- exp14m done ---"
echo ""

# exp14m2: Wider MLP for more capacity with higher-dim input
echo "=== exp14m2: SPP(1,2,4) both+sum 128d, hidden=[2048,1024,512] ==="
python -m feature_retrieval.patch_regressor_v7 \
    --pool spp \
    --feat both+sum \
    --exp_name exp14m2_128d_wider \
    --gpu 3 \
    --feature_dir "$FEATURE_DIR" \
    --epochs 3000 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --dropout 0.15 \
    --feature_dropout 0.15 \
    --hidden_dims 2048 1024 512 \
    --patch_dim 128 \
    --seed 42 \
    2>&1 | tail -25
echo "--- exp14m2 done ---"
echo ""

# exp14m3: Same as m but with different seeds for ensemble
for SEED in 123 456 789; do
    echo "=== exp14m3_seed${SEED}: SPP(1,2,4) both+sum 128d ==="
    python -m feature_retrieval.patch_regressor_v7 \
        --pool spp \
        --feat both+sum \
        --exp_name "exp14m3_128d_seed${SEED}" \
        --gpu 3 \
        --feature_dir "$FEATURE_DIR" \
        --epochs 3000 \
        --lr 0.001 \
        --weight_decay 0.0001 \
        --dropout 0.1 \
        --feature_dropout 0.1 \
        --hidden_dims 1024 512 256 \
        --patch_dim 128 \
        --seed "$SEED" \
        2>&1 | tail -25
    echo "--- exp14m3_seed${SEED} done ---"
    echo ""
done

# exp14m4: Smaller MLP to reduce overfitting with high-dim input
echo "=== exp14m4: SPP(1,2,4) both+sum 128d, hidden=[512,256,128] (small) ==="
python -m feature_retrieval.patch_regressor_v7 \
    --pool spp \
    --feat both+sum \
    --exp_name exp14m4_128d_small \
    --gpu 3 \
    --feature_dir "$FEATURE_DIR" \
    --epochs 5000 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --dropout 0.05 \
    --feature_dropout 0.05 \
    --hidden_dims 512 256 128 \
    --patch_dim 128 \
    --seed 42 \
    2>&1 | tail -25
echo "--- exp14m4 done ---"
echo ""

# exp14m5: Only coarse_sem 128d (it explained more variance)
echo "=== exp14m5: SPP(1,2,4) coarse+sum 128d ==="
python -m feature_retrieval.patch_regressor_v7 \
    --pool spp \
    --feat coarse+sum \
    --exp_name exp14m5_128d_coarse \
    --gpu 3 \
    --feature_dir "$FEATURE_DIR" \
    --epochs 3000 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --dropout 0.1 \
    --feature_dropout 0.1 \
    --hidden_dims 1024 512 256 \
    --patch_dim 128 \
    --seed 42 \
    2>&1 | tail -25
echo "--- exp14m5 done ---"
echo ""

echo "=== All 128d training complete! ==="
