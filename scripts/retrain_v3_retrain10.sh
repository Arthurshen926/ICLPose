#!/bin/bash
# Retrain10: Absgrad + balanced improvements WITHOUT Appearance Network
#
# Lessons from retrain9:
#   - Appearance Network caused PSNR REGRESSION (14.30→14.02 dB from 10k→20k)
#     The per-image color correction absorbed too much variation,
#     leaving base Gaussian colors in poor "average" state at test time
#   - Over-densification (508k Gaussians) didn't help quality
#
# Changes from retrain8 (v3_retrain8 baseline: 16.20 dB avg on 4 views):
#   1. absgrad=True       — proven effective for densification (AbsGS)
#   2. lambda_dssim=0.35  — moderate increase from 0.2 (not 0.5 as retrain9)  
#   3. 50k iterations     — more convergence time
#   4. densify_grad_threshold=0.0002  — same as retrain8 (avoid over-densification)
#   5. densify_until=20000 — same as retrain8
#   6. NO appearance network
#   7. NO opacity entropy regularization
#
# Expected improvement: absgrad + higher SSIM + longer training should boost PSNR
#   while avoiding the appearance network regression

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=1 python -u feature_3dgs/train_2dgs_geometry.py \
  --source_dir dataset/OldHospital \
  --model_dir output/2dgs_models/OldHospital/v3_retrain10 \
  --iterations 50000 \
  --batch_size 4 \
  --longest_edge 0 \
  --lambda_dssim 0.35 \
  --lambda_dist 0.01 \
  --lambda_normal 0.05 \
  --lambda_depth 0.1 \
  --mono_depth_dir dataset/OldHospital/mono_depth \
  --depth_start_iter 2000 \
  --lambda_scale 0.05 \
  --densify_from_iter 500 \
  --densify_until_iter 20000 \
  --densification_interval 100 \
  --densify_grad_threshold 0.0002 \
  --opacity_reset_interval 3000 \
  --save_iterations 10000 20000 30000 40000 50000 \
  --test_iterations 10000 20000 30000 40000 50000
