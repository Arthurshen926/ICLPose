#!/bin/bash
# Retrain23: Optimal balance — moderate identity-preserving reg
#
# Previous experiments established:
#   - retrain17 (no reg): peaked 16.99 at 15k, then declined to 15.70
#   - retrain22 (identity reg=0.1): stable but slow (16.61 stagnating at 5-7k)
#   - retrain21 (embedding L2 reg=0.01): raw crashed but corrected reached 16.65
#
# The key insight: identity-preserving reg works, but 0.1 is too strong.
# It over-constrains the appearance, preventing it from helping Gaussians
# optimize multi-view color disagreements.
#
# Solution: reg=0.01 gives MORE appearance freedom while still bounding
# the MLP's output deviation from identity. Combined with mean-embedding
# eval-time correction, this should:
#   1. Match retrain17's convergence speed (16.55→16.78→16.99)
#   2. Not degrade after 15k (thanks to reg + appearance LR decay)
#   3. Get even higher corrected PSNR via mean embedding correction
#
# Target: Raw PSNR ≥ 17.0 and/or Corrected PSNR ≥ 17.5

set -e

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0

nohup python -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir output/2dgs_models/OldHospital/v3_retrain23 \
    --iterations 30000 \
    --batch_size 4 \
    --longest_edge 0 \
    --random_background \
    --use_mask \
    --use_appearance \
    --appearance_lr_init 1e-3 \
    --appearance_lr_final 1e-5 \
    --appearance_reg 0.01 \
    --sh_degree 3 \
    \
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
    --lambda_depth 0.03 \
    --depth_start_iter 3000 \
    --mono_depth_dir dataset/OldHospital/mono_depth \
    --lambda_scale 0.0 \
    \
    --densify_from_iter 500 \
    --densify_until_iter 15000 \
    --densification_interval 100 \
    --densify_grad_threshold 0.00015 \
    --opacity_reset_interval 5000 \
    --opacity_reset_value 0.01 \
    --prune_dead_threshold 0.005 \
    --percent_dense 0.005 \
    \
    --save_iterations 3000 5000 7000 10000 12000 15000 18000 20000 22000 25000 30000 \
    --test_iterations 3000 5000 7000 10000 12000 15000 18000 20000 22000 25000 30000 \
    > output/retrain23.log 2>&1 &

echo "Retrain23 started on GPU 0. PID: $!"
echo "Log: output/retrain23.log"
echo "Monitor: tail -f output/retrain23.log"
