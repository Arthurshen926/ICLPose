#!/bin/bash
# Retrain26: retrain25 + lambda_dssim=0.5 + aggressive densification
#
# ═══ Changes from retrain25 ═══
# 1. lambda_dssim=0.5 (vs 0.2): Many papers (3DGS, 2DGS variants) report
#    better PSNR with higher SSIM weight. SSIM captures structural similarity
#    that L1 alone misses, leading to sharper + more accurate reconstructions.
#
# 2. More aggressive densification:
#    - densify_grad_threshold=0.0001 (vs 0.00015): triggers more splits
#    - percent_dense=0.01 (vs 0.005): larger clones for big splats
#    - densify_until_iter=30000 (vs 25000): longer densification period
#    Goal: More Gaussians = better coverage of test view regions.
#    retrain17 had only ~210k Gaussians — likely too few for 1920x1080 scene.
#
# All other settings identical to retrain25 (= retrain17 base)

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=1 nohup python -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir output/2dgs_models/OldHospital/v3_retrain26 \
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
    --lambda_dssim 0.5 \
    --lambda_dist 0.0 \
    --lambda_normal 0.0 \
    --normal_start_iter 999999 \
    --dist_start_iter 999999 \
    --lambda_depth 0.0 \
    --lambda_scale 0.0 \
    \
    --densify_from_iter 500 \
    --densify_until_iter 30000 \
    --densification_interval 100 \
    --densify_grad_threshold 0.0001 \
    --opacity_reset_interval 999999 \
    --opacity_reset_value 0.01 \
    --prune_dead_threshold 0.005 \
    --percent_dense 0.01 \
    \
    --save_iterations 3000 5000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 \
    --test_iterations 3000 5000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 \
    > output/retrain26.log 2>&1 &

echo "PID: $!"
echo "Log: output/retrain26.log"
echo "Monitor: tail -f output/retrain26.log"
echo "PSNR: grep 'Test PSNR' output/retrain26.log"
