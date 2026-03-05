#!/bin/bash
# Retrain16b: Aggressive densification variant of retrain16
#
# ═══ Retrain16 results (BEST SO FAR) ═══
# 3k=16.33, 5k=16.66, 7k=16.69, 10k=16.69, 15k=16.77 (still rising!)
# Key: no opacity reset, no regularization, lambda_dssim=0.2
# Gaussian count: 231k @ 15k (growing ~5.7k per 1k iters)
#
# ═══ Hypothesis: PSNR limited by Gaussian count (231k for 2M pixels) ═══
# This experiment: same as retrain16 but with more aggressive densification
#   - densify_grad_threshold: 0.00015 → 0.0001 (33% lower = more splits/clones)
#   - densify_until_iter: 25000 → 35000 (40% longer densification period)
#   - percent_dense: 0.005 → 0.01 (2x larger = more cloning)
#
# Expected: More Gaussians → better coverage of trees/fine structures → higher PSNR

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=1 python -u feature_3dgs/train_2dgs_geometry.py \
  --source_dir dataset/OldHospital \
  --model_dir output/2dgs_models/OldHospital/v3_retrain16b \
  --iterations 70000 \
  --batch_size 4 \
  --longest_edge 0 \
  \
  --position_lr_init 0.00005 \
  --position_lr_final 0.0000005 \
  --feature_lr 0.005 \
  --f_rest_lr_divisor 5.0 \
  --opacity_lr 0.1 \
  --scaling_lr 0.002 \
  --rotation_lr 0.002 \
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
  --densify_until_iter 35000 \
  --densification_interval 100 \
  --densify_grad_threshold 0.0001 \
  --opacity_reset_interval 999999 \
  --opacity_reset_value 0.01 \
  --prune_dead_threshold 0.005 \
  --percent_dense 0.01 \
  \
  --save_iterations 3000 5000 7000 10000 15000 20000 30000 40000 50000 60000 70000 \
  --test_iterations 3000 5000 7000 10000 15000 20000 30000 40000 50000 60000 70000
