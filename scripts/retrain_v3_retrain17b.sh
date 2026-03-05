#!/bin/bash
# Retrain17b: CONTROL - Random background ONLY (no mask, no appearance) 
#
# Purpose: Isolate the effect of random background from mask + appearance
# Compare with:
#   retrain16: no random bg, no mask, no appearance  → 16.77 @15k
#   retrain17: random bg + mask + appearance          → running
#   retrain17b: random bg ONLY                        → this experiment
#
# If retrain17b > retrain16 → random bg helps
# If retrain17 > retrain17b → mask+appearance add value beyond random bg

cd /home/yons/Projects/ICLPose

# NOTE: Run this on GPU 0 after retrain16 finishes or is killed.
# For now, this is a standby script.
CUDA_VISIBLE_DEVICES=0 python -u feature_3dgs/train_2dgs_geometry.py \
  --source_dir dataset/OldHospital \
  --model_dir output/2dgs_models/OldHospital/v3_retrain17b \
  --iterations 70000 \
  --batch_size 4 \
  --longest_edge 0 \
  \
  --random_background \
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
  --densify_until_iter 25000 \
  --densification_interval 100 \
  --densify_grad_threshold 0.00015 \
  --opacity_reset_interval 999999 \
  --opacity_reset_value 0.01 \
  --prune_dead_threshold 0.005 \
  --percent_dense 0.005 \
  \
  --save_iterations 3000 5000 7000 10000 15000 20000 30000 40000 50000 60000 70000 \
  --test_iterations 3000 5000 7000 10000 15000 20000 30000 40000 50000 60000 70000
