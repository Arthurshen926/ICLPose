#!/bin/bash
# Retrain18b: Gradient fix + moderate densification threshold
#
# ═══ Changes from retrain18 ═══
# retrain18 had grad_threshold=0.0001 → 1.4M Gaussians (TOO MANY, slower, worse PSNR)
# retrain18b uses 0.0003 — between standard 3DGS (0.0002) and pre-fix effective (0.0006)
# Target: 300-500k Gaussians (vs retrain17's 200k and retrain18's 1.4M)
#
# ═══ Changes from retrain17 (code-level) ═══
# 1. Batch gradient dilution fix: grad_data * batch_size in densification stats
# 2. Masked eval PSNR: reports both full and masked PSNR
#
# ═══ retrain17 trajectory ═══
# 3k=16.55 → 5k=16.78 → 7k=16.88 → 10k=16.94 → 15k=16.99 → 20k=16.82 (degraded!)
#
# ═══ retrain18 trajectory ═══
# 3k=16.44 (WORSE, 1.4M Gaussians too many, 1.69 it/s too slow)
#
# ═══ Hypothesis ═══
# retrain17's batch gradient bug made densification 4x too conservative
# retrain18 overcorrected (threshold too low). retrain18b finds the sweet spot.
# Extended densification (40k) + proper gradient scaling → better coverage of
# under-reconstructed areas without overwhelming the model.

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=0 python -u feature_3dgs/train_2dgs_geometry.py \
  --source_dir dataset/OldHospital \
  --model_dir output/2dgs_models/OldHospital/v3_retrain18b \
  --iterations 70000 \
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
  --densify_until_iter 40000 \
  --densification_interval 100 \
  --densify_grad_threshold 0.0003 \
  --opacity_reset_interval 999999 \
  --opacity_reset_value 0.01 \
  --prune_dead_threshold 0.005 \
  --percent_dense 0.005 \
  \
  --save_iterations 3000 5000 7000 10000 15000 20000 30000 40000 50000 60000 70000 \
  --test_iterations 3000 5000 7000 10000 15000 20000 30000 40000 50000 60000 70000
