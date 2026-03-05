#!/bin/bash
# Retrain18: Fix batch gradient dilution + extended densification + masked eval
#
# ═══ Three critical improvements over retrain17 ═══
#
# 1. BATCH GRADIENT DILUTION FIX (code change):
#    batch_size=4 divided gradients by 4 before densification stats,
#    making the effective threshold 4x higher (0.0006 vs intended 0.00015).
#    Fix: multiply grad_data by batch_size before add_densification_stats().
#    This means the Gaussians "under-densified" in ALL previous experiments!
#
# 2. MASKED EVAL PSNR (code change):
#    Training masks out dynamic objects & distortion edges, but eval computed
#    PSNR on ALL pixels → unfair penalty. Now reports both full & masked PSNR.
#
# 3. EXTENDED DENSIFICATION (this script):
#    25k → 45k iterations of densification (was stopping too early)
#    Lower grad_threshold: 0.00015 → 0.0001 (proper threshold now that
#    gradient scaling is fixed)
#    100k total iterations (more refinement time after densification)
#
# ═══ Kept from retrain17 ═══
# - Random background (encourages opaque Gaussians)
# - Appearance embedding (handles exposure variation)
# - Mask filtering (sky/dynamic/distortion)
# - No regularization (confirmed harmful)
# - No opacity reset (confirmed harmful)
# - Our LR config (confirmed better than standard 3DGS)
# - lambda_dssim=0.2
#
# retrain17 trajectory: 3k=16.55 → 5k=16.78 → 7k=16.88 → 10k=16.94 → 15k=16.99
# Expected: Significantly more Gaussians (fix gives ~4x more densification)
#           and higher PSNR from better scene coverage

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=0 python -u feature_3dgs/train_2dgs_geometry.py \
  --source_dir dataset/OldHospital \
  --model_dir output/2dgs_models/OldHospital/v3_retrain18 \
  --iterations 100000 \
  --batch_size 4 \
  --longest_edge 0 \
  \
  --random_background \
  --use_appearance \
  --use_mask \
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
  --densify_until_iter 45000 \
  --densification_interval 100 \
  --densify_grad_threshold 0.0001 \
  --opacity_reset_interval 999999 \
  --opacity_reset_value 0.01 \
  --prune_dead_threshold 0.005 \
  --percent_dense 0.005 \
  \
  --save_iterations 3000 5000 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
  --test_iterations 3000 5000 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000
