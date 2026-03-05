#!/bin/bash
# Retrain31: Appearance freeze at 5k (early freeze)
#
# ═══ Analysis ═══
# In retrain30v2, appearance was active 0-15k and frozen at 15k:
#   - 5k: raw PSNR 16.81 (appearance peak)
#   - 7k-15k: declining to 16.61 (appearance drift)
#   - 17k-20k: recovering to 16.85 (post-freeze adaptation)
#
# The key observation: by 15k, appearance had already "damaged" the Gaussians
# by shifting their learned colors away from canonical. The deeper the damage,
# the harder to recover.
#
# HYPOTHESIS: Freezing at 5k (appearance peak, minimal drift) gives Gaussians
# a better starting point for post-freeze optimization. The raw rendering at
# 5k is already good (16.81), so disabling appearance lets Gaussians continue
# improving from that strong foundation.
#
# retrain17 hit 16.99 at 15k with UNfreezed appearance — but here we freeze
# early and let the Gaussians have 65k more iterations to improve WITHOUT the
# train/test mismatch. This should yield higher final PSNR.

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=GPU_ID nohup python -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir output/2dgs_models/OldHospital/v3_retrain31 \
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
    > output/retrain31.log 2>&1 &

echo "PID: $!"
echo "Log: output/retrain31.log"
