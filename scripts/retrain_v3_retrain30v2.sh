#!/bin/bash
# Retrain30v2: Appearance + FREEZE @15k (NO gradient clipping!)
#
# CRITICAL FIX: Previous retrain28-30 all had clip_grad_norm_(max_norm=1.0)
# which was NOT in retrain17's code. With 200k+ Gaussians, the parameter
# gradient norm easily exceeds 1.0, so clipping effectively reduced all
# learning rates by 100x+. This is likely why retrain28's no-appearance
# baseline peaked lower (16.68) vs retrain17 (16.99).
#
# This run removes the gradient clipping and should replicate retrain17's
# trajectory in the first 15k, then freeze the appearance network.
#
# Settings: IDENTICAL to retrain17 + appearance_freeze_iter=15000
# Wide appearance bounds [0, 2] to match retrain17's old architecture.

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=1 nohup python -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir output/2dgs_models/OldHospital/v3_retrain30v2 \
    --iterations 70000 \
    --batch_size 4 \
    --longest_edge 0 \
    \
    --random_background \
    --use_appearance \
    --use_mask \
    --appearance_freeze_iter 15000 \
    --appearance_scale_range 2.0 \
    --appearance_bias_range 0.5 \
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
    --save_iterations 3000 5000 7000 10000 12000 15000 17000 20000 25000 30000 40000 50000 60000 70000 \
    --test_iterations 3000 5000 7000 10000 12000 15000 17000 20000 25000 30000 40000 50000 60000 70000 \
    > output/retrain30v2.log 2>&1 &

echo "PID: $!"
echo "Log: output/retrain30v2.log"
echo "Monitor: tail -f output/retrain30v2.log"
