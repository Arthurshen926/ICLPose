#!/bin/bash
# New architecture experiments on GPU 5 with 128d features
# Tests: residual MLP, mini-batch training, translation noise
set -e
cd /root/ICLPose-loc

FEATURE_DIR="output/feature_extract/features_radio_dual_128/OldHospital_pilot"
GPU=5

echo "=== New Architecture Experiments ==="

# 1. Residual MLP (4 blocks, 1024d hidden)
for SEED in 123 456 789; do
    EXP_NAME="exp14t_128d_residual_seed${SEED}"
    echo "=== $EXP_NAME ==="
    
    CUDA_VISIBLE_DEVICES=$GPU python -m feature_retrieval.patch_regressor_v7 \
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
        --hidden_dims 1024 \
        --patch_dim 128 \
        --seed "$SEED" \
        --model_type residual \
        --n_res_blocks 4 \
        2>&1 | tail -40
    
    echo "--- $EXP_NAME done ---"
    echo ""
done

# 2. Residual MLP wider (2048d hidden, 6 blocks)
for SEED in 123 456; do
    EXP_NAME="exp14t2_128d_residual_wide_seed${SEED}"
    echo "=== $EXP_NAME ==="
    
    CUDA_VISIBLE_DEVICES=$GPU python -m feature_retrieval.patch_regressor_v7 \
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
        --hidden_dims 2048 \
        --patch_dim 128 \
        --seed "$SEED" \
        --model_type residual \
        --n_res_blocks 6 \
        2>&1 | tail -40
    
    echo "--- $EXP_NAME done ---"
    echo ""
done

# 3. Mini-batch training (batch_size=256 for SGD noise)
for SEED in 123 456 789; do
    EXP_NAME="exp14u_128d_minibatch_seed${SEED}"
    echo "=== $EXP_NAME ==="
    
    CUDA_VISIBLE_DEVICES=$GPU python -m feature_retrieval.patch_regressor_v7 \
        --pool spp \
        --feat both+sum \
        --exp_name "$EXP_NAME" \
        --gpu 0 \
        --feature_dir "$FEATURE_DIR" \
        --epochs 15000 \
        --lr 0.001 \
        --weight_decay 0.0001 \
        --dropout 0.15 \
        --feature_dropout 0.15 \
        --hidden_dims 2048 1024 512 \
        --patch_dim 128 \
        --seed "$SEED" \
        --batch_size 256 \
        2>&1 | tail -40
    
    echo "--- $EXP_NAME done ---"
    echo ""
done

# 4. Translation noise (sigma=0.3m and 0.5m)
for NOISE in 0.3 0.5; do
    for SEED in 123 456; do
        EXP_NAME="exp14v_128d_tnoise${NOISE}_seed${SEED}"
        echo "=== $EXP_NAME ==="
        
        CUDA_VISIBLE_DEVICES=$GPU python -m feature_retrieval.patch_regressor_v7 \
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
            --trans_noise "$NOISE" \
            2>&1 | tail -40
        
        echo "--- $EXP_NAME done ---"
        echo ""
    done
done

# 5. Combo: residual + mini-batch + trans_noise
for SEED in 123 456; do
    EXP_NAME="exp14w_128d_combo_seed${SEED}"
    echo "=== $EXP_NAME ==="
    
    CUDA_VISIBLE_DEVICES=$GPU python -m feature_retrieval.patch_regressor_v7 \
        --pool spp \
        --feat both+sum \
        --exp_name "$EXP_NAME" \
        --gpu 0 \
        --feature_dir "$FEATURE_DIR" \
        --epochs 15000 \
        --lr 0.001 \
        --weight_decay 0.0001 \
        --dropout 0.15 \
        --feature_dropout 0.15 \
        --hidden_dims 1024 \
        --patch_dim 128 \
        --seed "$SEED" \
        --model_type residual \
        --n_res_blocks 6 \
        --batch_size 256 \
        --trans_noise 0.3 \
        2>&1 | tail -40
    
    echo "--- $EXP_NAME done ---"
    echo ""
done

echo "=== ALL DONE ==="
