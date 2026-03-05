#!/bin/bash
# Retrain28v2: No appearance + retrain17 base (NO gradient clipping!)
#
# CRITICAL FIX: Same as retrain30v2 — previous retrain28 had gradient
# clipping (max_norm=1.0) that killed convergence. Without this clipping,
# the no-appearance baseline should be a fair comparison to retrain17.
#
# If this exceeds retrain17's 16.99 dB, then appearance is harmful.
# If this peaks around 16.68 (like retrain28), then grad clip WAS the issue.
# If this peaks even higher, then no-appearance + no-clip is the best combo.

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=0 nohup python -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir output/2dgs_models/OldHospital/v3_retrain28v2 \
    --iterations 70000 \
    --batch_size 4 \
    --longest_edge 0 \
    \
    --random_background \
    --use_mask \
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
    --save_iterations 3000 5000 7000 10000 12000 15000 20000 25000 30000 40000 50000 60000 70000 \
    --test_iterations 3000 5000 7000 10000 12000 15000 20000 25000 30000 40000 50000 60000 70000 \
    > output/retrain28v2.log 2>&1 &

echo "PID: $!"
echo "Log: output/retrain28v2.log"
echo "Monitor: tail -f output/retrain28v2.log"
