#!/bin/bash
# Feature Whitening Experiments (128d, SPP pooling)
# ==================================================
# PCA whitening decorrelates pooled features before MLP training.
# Hypothesis: LayerNorm only normalizes per-sample, whitening decorrelates
# across feature dimensions using training set statistics.
set -e

GPU=3
FEAT_DIR="output/feature_extract/features_radio_dual_128/OldHospital_pilot"
DATASET_DIR="dataset/OldHospital"
BASE="python -m feature_retrieval.patch_regressor_v7 --gpu ${GPU} --feature_dir ${FEAT_DIR} --dataset_dir ${DATASET_DIR} --patch_dim 128 --feat both+sum --epochs 5000 --lr 0.001 --weight_decay 0.0001 --precompute_pool --whiten"

echo "============================================================"
echo "Feature Whitening Experiments"
echo "============================================================"
echo ""

# --- SPP + Whitening (best pooling + whitening) ---
for SEED in 314 123 456 789; do
    echo "=== exp17a_128d_spp_whiten_seed${SEED} ==="
    ${BASE} --pool spp --hidden_dims 2048 1024 512 --dropout 0.15 --feature_dropout 0.15 \
        --seed ${SEED} --exp_name exp17a_128d_spp_whiten_seed${SEED}
    echo "--- exp17a_128d_spp_whiten_seed${SEED} done ---"
    echo ""
done

# --- GAP + Whitening (smaller input with whitening) ---
for SEED in 314 123; do
    echo "=== exp17b_128d_gap_whiten_seed${SEED} ==="
    ${BASE} --pool gap --hidden_dims 1024 512 256 --dropout 0.15 --feature_dropout 0.15 \
        --seed ${SEED} --exp_name exp17b_128d_gap_whiten_seed${SEED}
    echo "--- exp17b_128d_gap_whiten_seed${SEED} done ---"
    echo ""
done

# --- GeM + Whitening ---
for SEED in 314 123; do
    echo "=== exp17c_128d_gem_whiten_seed${SEED} ==="
    ${BASE} --pool gem --hidden_dims 1024 512 256 --dropout 0.15 --feature_dropout 0.15 \
        --seed ${SEED} --exp_name exp17c_128d_gem_whiten_seed${SEED}
    echo "--- exp17c_128d_gem_whiten_seed${SEED} done ---"
    echo ""
done

echo ""
echo "=== ALL WHITENING EXPERIMENTS DONE ==="
