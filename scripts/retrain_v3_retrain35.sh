#!/bin/bash
# Retrain35: Geometric regularization + freeze@5k
#
# ═══ Analysis ═══
# retrain31 post-freeze: Gaussians optimize freely but without geometric
# constraints. Gaussian splats may become overly degenerate (large flat discs),
# causing artifacts from novel views.
#
# HYPOTHESIS: Adding geometric regularization (lambda_normal + lambda_dist)
# constrains Gaussian shapes to be more physically plausible, improving
# generalization to test views. Start regularization at 7000 (after freeze
# recovery period, to not interfere with post-freeze adaptation).
#
# lambda_normal=0.01: Encourages Gaussian normals to align with rendered normals
# lambda_dist=0.01: Penalizes degenerate (too spread out) Gaussians
# Both start at iter 7000 (2k after freeze@5k, giving time to adapt first)

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=GPU_ID nohup python -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir output/2dgs_models/OldHospital/v3_retrain35 \
    --iterations 70000 \
    --batch_size 4 \
    --longest_edge 0 \
    \
    --random_background \
    --use_appearance \
    --use_mask \
    --appearance_freeze_iter 5000 \
    --appearance_scale_range 2.0 \
    --appearance_bias_range 0.5 \
    --grad_clip_max_norm 1.0 \
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
    --lambda_dist 0.01 \
    --lambda_normal 0.01 \
    --normal_start_iter 7000 \
    --dist_start_iter 7000 \
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
    > output/retrain35.log 2>&1 &

echo "PID: $!"
echo "Log: output/retrain35.log"
