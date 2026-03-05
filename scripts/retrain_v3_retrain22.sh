#!/bin/bash
# Retrain22: Identity-preserving appearance regularization
#
# Key insight from retrain21: L2 reg on embeddings doesn't prevent the
# MLP from learning systematic color shifts (scale drifted to 0.93 at 7k).
# This caused raw PSNR to drop from 16.49 to 16.02, with the appearance
# network "stealing" base color from Gaussians.
#
# Fix: Identity-preserving regularization that penalizes actual affine
# output (scale, bias) deviating from identity (1, 0). This directly
# bounds how much the appearance network can shift colors.
#
# Settings:
#   1. lr_decay_factor=1.0 — NO global LR decay (standard gsplat schedule)
#   2. appearance_lr: 1e-3 → 1e-5 — aggressive decay on appearance ONLY
#   3. appearance_reg=0.1 — STRONG identity-preserving reg (new formulation)
#   4. eval-time mean-embedding correction
#   5. 30k iterations
#
# The reg weight is 0.1 (higher than retrain21's 0.01) because identity-reg
# is already bounded by sigmoid — it can never produce extreme gradients.

set -e

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=1

nohup python -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir output/2dgs_models/OldHospital/v3_retrain22 \
    --iterations 30000 \
    --batch_size 4 \
    --longest_edge 0 \
    --random_background \
    --use_mask \
    --use_appearance \
    --appearance_lr_init 1e-3 \
    --appearance_lr_final 1e-5 \
    --appearance_reg 0.1 \
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
    > output/retrain22.log 2>&1 &

echo "Retrain22 started on GPU 1. PID: $!"
echo "Log: output/retrain22.log"
echo "Monitor: tail -f output/retrain22.log"
