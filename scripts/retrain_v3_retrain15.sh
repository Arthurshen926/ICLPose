#!/bin/bash
# Retrain15: Standard 3DGS LR + NO regularization
#
# ═══ Key findings from retrain13/13b/14 ═══
# ALL regularization is harmful for this outdoor scene:
#   - depth supervision (DPT): -0.34 dB after 10k iters
#   - normal/dist reg: -0.51 dB after 3k iters
# Best PSNR achieved: 16.68 dB @ 5k (retrain13b, before any reg kicked in)
#
# ═══ Hypothesis: Learning rate is limiting PSNR ═══
# Our position_lr_init=0.00005 vs standard 3DGS=0.00016 (3.2x lower!)
# Gaussians can't move fast enough to optimal positions
#
# ═══ Changes in this experiment ═══
# 1. STANDARD 3DGS learning rates:
#    position_lr_init: 0.00005 → 0.00016 (3.2x up)
#    position_lr_final: 0.0000005 → 0.0000016 (3.2x up)
#    scaling_lr: 0.002 → 0.005 (2.5x up, standard)
#    opacity_lr: 0.1 → 0.05 (standard)
#    rotation_lr: 0.002 → 0.001 (standard)
#    feature_lr: 0.005 → 0.0025 (standard)
# 2. NO regularization of any kind:
#    lambda_depth=0, lambda_normal=0, lambda_dist=0, lambda_scale=0
# 3. More Gaussians via lower densify threshold:
#    densify_grad_threshold: 0.00015 → 0.0001
#    densify_until_iter: 25000 → 30000
# 4. Keep opacity reset but with more standard interval
#
# Expected: 17-18+ dB (removing all limiting factors)

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=0 python -u feature_3dgs/train_2dgs_geometry.py \
  --source_dir dataset/OldHospital \
  --model_dir output/2dgs_models/OldHospital/v3_retrain15 \
  --iterations 70000 \
  --batch_size 4 \
  --longest_edge 0 \
  \
  --position_lr_init 0.00016 \
  --position_lr_final 0.0000016 \
  --feature_lr 0.0025 \
  --f_rest_lr_divisor 20.0 \
  --opacity_lr 0.05 \
  --scaling_lr 0.005 \
  --rotation_lr 0.001 \
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
  --densify_until_iter 30000 \
  --densification_interval 100 \
  --densify_grad_threshold 0.0001 \
  --opacity_reset_interval 3000 \
  --opacity_reset_value 0.01 \
  --prune_dead_threshold 0.005 \
  --percent_dense 0.01 \
  \
  --save_iterations 5000 7000 10000 15000 20000 30000 40000 50000 60000 70000 \
  --test_iterations 5000 7000 10000 15000 20000 30000 40000 50000 60000 70000
