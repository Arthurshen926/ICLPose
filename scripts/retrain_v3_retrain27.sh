#!/bin/bash
# Retrain27: retrain17 base + MEAN-CENTERING appearance regularization
#
# ═══ Key insight ═══
# retrain25 (bounded [0.8,1.2] + individual identity reg=0.05) only reached
# 16.44 @5k vs retrain17's 16.78. Individual reg restricts per-image freedom
# too much, hurting convergence.
#
# NEW APPROACH: Mean-centering regularization
# - Allow each training image to have its OWN large appearance correction
#   (e.g., dark image → scale=0.7, bright image → scale=1.3)
# - But constrain the POPULATION MEAN of all corrections to be near identity
#   (mean scale ≈ 1.0, mean bias ≈ 0.0)
# - This means: at test time, the raw 2DGS output (no appearance correction)
#   IS the "average" appearance → good raw test PSNR
#
# Also: WIDER bounds (scale_range=0.8 → [0.6, 1.4]) to allow bigger per-image
# corrections while the mean stays centered via the regularization.
#
# Settings:
#   - retrain17 base (NO opacity reset, densify 25k, NO depth loss)
#   - appearance_reg=0 (no individual reg)
#   - appearance_mean_reg=1.0 (strong mean-centering)
#   - scale_range=0.8 ([0.6, 1.4]), bias_range=0.1 (±0.1)
#   - 70k iterations, lr_decay_factor=1.0

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=GPU_ID nohup python -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir output/2dgs_models/OldHospital/v3_retrain27 \
    --iterations 70000 \
    --batch_size 4 \
    --longest_edge 0 \
    \
    --random_background \
    --use_appearance \
    --use_mask \
    --appearance_lr_init 1e-3 \
    --appearance_lr_final 1e-5 \
    --appearance_reg 0.0 \
    --appearance_mean_reg 1.0 \
    --appearance_scale_range 0.8 \
    --appearance_bias_range 0.1 \
    \
    --sh_degree 3 \
    --position_lr_init 0.00005 \
    --position_lr_final 0.0000005 \
    --feature_lr 0.005 \
    --f_rest_lr_divisor 5.0 \
    --opacity_lr 0.1 \
    --scaling_lr 0.002 \
    --rotation_lr 0.002 \
    --lr_decay_factor 1.0 \
    \
    --lambda_dssim 0.2 \
    --lambda_dist 0.0 \
    --lambda_normal 0.0 \
    --normal_start_iter 999999 \
    --dist_start_iter 999999 \
    --lambda_depth 0.0 \
    --lambda_scale 0.0 \
    \
    --densify_from_iter 500 \
    --densify_until_iter 25000 \
    --densification_interval 100 \
    --densify_grad_threshold 0.00015 \
    --opacity_reset_interval 999999 \
    --opacity_reset_value 0.01 \
    --prune_dead_threshold 0.005 \
    --percent_dense 0.005 \
    \
    --save_iterations 3000 5000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 \
    --test_iterations 3000 5000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 \
    > output/retrain27.log 2>&1 &

echo "PID: $!"
echo "Log: output/retrain27.log"
echo "Monitor: tail -f output/retrain27.log"
echo "PSNR: grep 'Test PSNR' output/retrain27.log"
