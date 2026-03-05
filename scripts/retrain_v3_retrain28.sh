#!/bin/bash
# Retrain28: retrain17 base WITHOUT appearance (ablation)
#
# retrain19 tested no-appearance but with BAD base settings:
#   - opacity_reset_interval=5000 (harmful!)
#   - lambda_depth=0.03 
#   - densify_until_iter=15000
# Result: 3k=16.41, 5k=16.42, 7k=16.36 (stagnating/declining)
#
# Question: Is the appearance network actually needed when we use retrain17's
# GOOD base settings (no opacity reset, densify 25k, no depth loss)?
#
# If PSNR is similar to retrain17, then appearance is causing instability
# without providing benefit → simplest approach is best.

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=GPU_ID nohup python -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir output/2dgs_models/OldHospital/v3_retrain28 \
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
    --save_iterations 3000 5000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 \
    --test_iterations 3000 5000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 \
    > output/retrain28.log 2>&1 &

echo "PID: $!"
echo "Log: output/retrain28.log"
echo "PSNR: grep 'Test PSNR' output/retrain28.log"
