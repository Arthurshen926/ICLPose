#!/bin/bash
# Retrain29: THE BEST OF BOTH WORLDS
#
# ═══ Key synthesis ═══
# retrain24 had the HIGHEST 3k PSNR ever (16.69!) with bounded appearance
# [0.8, 1.2] + identity reg=0.01. But it CRASHED at 5k because of opacity
# reset (opacity_reset_interval=5000). 
#
# retrain17 had the HIGHEST overall PSNR (16.99) thanks to good base settings:
# NO opacity reset, densify until 25k, NO depth loss. But its unbounded
# appearance network crashed it after 15k.
#
# THIS EXPERIMENT combines:
#   retrain24's appearance: bounded [0.8,1.2], reg=0.01 (light)
#   retrain17's base: NO opacity reset, densify 25k, NO depth loss
#
# Expected: 16.69+ at 3k, stable growth to 17.0+ (no crash at 5k or 15k)

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=1 nohup python -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir output/2dgs_models/OldHospital/v3_retrain29 \
    --iterations 70000 \
    --batch_size 4 \
    --longest_edge 0 \
    \
    --random_background \
    --use_appearance \
    --use_mask \
    --appearance_lr_init 1e-3 \
    --appearance_lr_final 1e-5 \
    --appearance_reg 0.01 \
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
    > output/retrain29.log 2>&1 &

echo "PID: $!"
echo "Log: output/retrain29.log"
echo "Monitor: tail -f output/retrain29.log"
echo "PSNR: grep 'Test PSNR' output/retrain29.log"
