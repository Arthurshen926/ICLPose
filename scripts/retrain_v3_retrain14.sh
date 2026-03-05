#!/bin/bash
# Retrain14: No depth supervision — testing if DPT depth is harmful
#
# ═══ Evidence depth is harmful ═══
# retrain13b @5k (before depth): 16.68 dB (PEAK)
# retrain13b @10k (after depth 5k-10k): 16.66 dB (flat)
# retrain13b @15k (after depth 5k-15k): 16.41 dB (DEGRADED -0.27)
#
# All previous runs show same pattern: PSNR degrades after depth starts
# DPT is unreliable on: trees, sky, transparent objects, repeated textures
# Pearson correlation loss forces Gaussians to match bad depth → geometry distortion
#
# ═══ This experiment ═══
# Same as retrain13b (no mask) but lambda_depth=0
# Keep delayed normal/dist regularization (helped: +0.43 dB vs retrain11)
# Expected: PSNR should keep rising past 5k instead of degrading
#
# Comparison baseline: retrain13b (same params but lambda_depth=0.05)

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=0 python -u feature_3dgs/train_2dgs_geometry.py \
  --source_dir dataset/OldHospital \
  --model_dir output/2dgs_models/OldHospital/v3_retrain14 \
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
  --lambda_dssim 0.5 \
  --lambda_dist 0.01 \
  --lambda_normal 0.02 \
  --normal_start_iter 15000 \
  --dist_start_iter 12000 \
  --lambda_depth 0.0 \
  --lambda_scale 0.01 \
  --scale_reg_threshold 1.0 \
  \
  --densify_from_iter 500 \
  --densify_until_iter 25000 \
  --densification_interval 100 \
  --densify_grad_threshold 0.00015 \
  --opacity_reset_interval 5000 \
  --opacity_reset_value 0.05 \
  --prune_dead_threshold 0.005 \
  --percent_dense 0.005 \
  \
  --save_iterations 5000 10000 15000 20000 30000 40000 50000 60000 70000 \
  --test_iterations 5000 10000 15000 20000 30000 40000 50000 60000 70000
