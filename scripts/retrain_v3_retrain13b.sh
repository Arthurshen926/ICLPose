#!/bin/bash
# Retrain13b: NO-MASK control experiment (compare with retrain13 which uses masks)
#
# Purpose: Isolate the effect of dynamic object masking
#   - Same parameters as retrain13 EXCEPT no --use_mask
#   - Same delayed regularization (learned from retrain11 failures)
#   - If retrain13 >> retrain13b → masks help significantly
#   - If retrain13 ≈ retrain13b → need different improvements
#
# Key differences from retrain11:
#   + delayed normal_start_iter: 7000 → 15000
#   + delayed dist_start_iter: 7000 → 12000
#   + reduced lambda_normal: 0.05 → 0.02
#   + reduced lambda_depth: 0.1 → 0.05
#   + delayed depth_start_iter: 3000 → 5000
#   + slightly more aggressive pruning: prune_dead_threshold 0.003 → 0.005
#   + densify_until_iter 30000 → 25000

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=1 python -u feature_3dgs/train_2dgs_geometry.py \
  --source_dir dataset/OldHospital \
  --model_dir output/2dgs_models/OldHospital/v3_retrain13b \
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
  --lambda_depth 0.05 \
  --mono_depth_dir dataset/OldHospital/mono_depth \
  --depth_start_iter 5000 \
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
