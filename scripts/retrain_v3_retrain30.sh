#!/bin/bash
# Retrain30: Appearance with FREEZE at 15k
#
# ═══ Core insight ═══  
# retrain17 proved: appearance network helps Gaussians converge faster and
# better during early training (16.99 @15k). But the train/test mismatch
# from appearance drift causes post-15k crash (15.70 @70k).
#
# SOLUTION: Use appearance for the first 15k iterations (fast convergence),
# then FREEZE/disable it. After freezing, Gaussians continue training
# WITHOUT appearance correction → they adapt their SH colors to encode
# the "correct" canonical appearance directly.
#
# No need for bounded architecture or regularization — just disable at
# the right time and let Gaussians self-correct.
#
# Expected trajectory:
#   0-15k: ~retrain17 performance (16.99 @15k)  
#   15k+: Brief dip after freeze, then recovery as Gaussians adapt
#   Final: Stable PSNR matching or exceeding 16.99

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=1 nohup python -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir output/2dgs_models/OldHospital/v3_retrain30 \
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
    > output/retrain30.log 2>&1 &

echo "PID: $!"
echo "Log: output/retrain30.log"
echo "Monitor: tail -f output/retrain30.log"
echo "PSNR: grep 'Test PSNR' output/retrain30.log"
