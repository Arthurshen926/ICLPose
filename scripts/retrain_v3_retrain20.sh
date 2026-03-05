#!/bin/bash
# Retrain20: Appearance network WITH strong regularization + LR decay
#
# Hypothesis: Appearance network helps handle exposure variation, but
#   the unregulated LR=1e-3 causes it to overfit and corrupt base colors.
#   Solution: Keep appearance but with aggressive LR decay + L2 reg.
#
# Key changes from retrain17:
#   1. Appearance network with LR decay: 1e-3 → 1e-5
#   2. Appearance embedding L2 regularization (appearance_reg=0.01)
#   3. Short training: 25k iterations
#   4. All-parameter LR decay (lr_decay_factor=0.1)
#   5. Mild depth supervision (lambda_depth=0.03)
#   6. Opacity reset every 5000 iters
#
# Compare with retrain19 (no appearance) to understand the effect of
# the regularized appearance network.

set -e

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=1

nohup python -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir output/2dgs_models/OldHospital/v3_retrain20 \
    --iterations 25000 \
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
    --lr_decay_factor 0.1 \
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
    --save_iterations 3000 5000 7000 10000 12000 15000 18000 20000 22000 25000 \
    --test_iterations 3000 5000 7000 10000 12000 15000 18000 20000 22000 25000 \
    > output/retrain20.log 2>&1 &

echo "Retrain20 started on GPU 1. PID: $!"
echo "Log: output/retrain20.log"
echo "Monitor: tail -f output/retrain20.log"
