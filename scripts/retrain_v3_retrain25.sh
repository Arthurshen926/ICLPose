#!/bin/bash
# Retrain25: retrain17 settings + bounded appearance (crash prevention)
#
# ═══ Key insight ═══
# retrain17 reached 16.99 dB @15k but CRASHED to 16.06 by 40k.
# Root cause: unbounded appearance network drifted (scale→0.93, bias→0.03)
# stealing base color from Gaussians. The train appearance correction creates
# a train/test distribution mismatch that destroys test PSNR.
#
# retrain22-24 tried to fix this but used wrong base settings:
# - opacity_reset_interval=5000 (devastating for large outdoor scene!)
# - densify_until_iter=15000 (too short)
# - lambda_depth=0.03 (hurts PSNR)
# retrain17 had: NO opacity reset, densify 25k, NO depth loss → MUCH better
#
# ═══ This experiment ═══
# Exact retrain17 settings + bounded appearance architecture + identity reg
# - Scale bounded to [0.8, 1.2] (cannot drift further)
# - Bias bounded to [-0.05, +0.05]
# - Identity reg weight 0.05 (moderate: between retrain22's 0.1 and 23's 0.01)
# - Everything else IDENTICAL to retrain17
#
# Expected: Should match retrain17's 16.99 at 15k AND keep climbing past 15k

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=0 nohup python -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir output/2dgs_models/OldHospital/v3_retrain25 \
    --iterations 70000 \
    --batch_size 4 \
    --longest_edge 0 \
    \
    --random_background \
    --use_appearance \
    --use_mask \
    --appearance_lr_init 1e-3 \
    --appearance_lr_final 1e-5 \
    --appearance_reg 0.05 \
    --appearance_scale_range 0.4 \
    --appearance_bias_range 0.05 \
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
    > output/retrain25.log 2>&1 &

echo "PID: $!"
echo "Log: output/retrain25.log"
echo "Monitor: tail -f output/retrain25.log"
echo "PSNR: grep 'Test PSNR' output/retrain25.log"
