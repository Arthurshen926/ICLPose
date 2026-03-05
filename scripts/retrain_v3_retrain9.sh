#!/bin/bash
# Retrain9: SOTA-inspired comprehensive improvements
#
# Key changes from retrain8 (all in train_2dgs_geometry.py):
#   1. absgrad=True        — gsplat absolute gradients for better densification (AbsGS)
#   2. AppearanceNetwork   — Per-image affine color correction (Gaussian in the Wild)
#   3. lambda_dssim=0.5    — Higher SSIM weight for sharper results (was 0.2)
#   4. Opacity entropy reg — Encourages binary opacity, reduces semi-transparent floaters
#   5. More densification  — Lower grad threshold, longer densify period
#   6. 50k iterations      — More convergence time (was 30k)
#
# Config changes vs retrain8:
#   - iterations: 30000 → 50000
#   - lambda_dssim: 0.2 → 0.5
#   - densify_grad_threshold: 0.0002 → 0.00015
#   - densify_until_iter: 15000 → 25000
#   - lambda_scale: 0.1 → 0.05 (slightly relaxed)
#   - lambda_opacity_entropy: 0.001 (NEW)
#   - use_appearance: enabled (NEW)
#   - save/test at 10k, 20k, 30k, 40k, 50k

set -e

export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')"):$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0 python -m feature_3dgs.train_2dgs_geometry \
  --source_dir dataset/OldHospital \
  --model_dir output/2dgs_models/OldHospital/v3_retrain9 \
  --images "" \
  --iterations 50000 \
  --batch_size 4 \
  --longest_edge 0 \
  --lambda_dssim 0.5 \
  --lambda_dist 0.01 \
  --lambda_normal 0.05 \
  --lambda_depth 0.1 \
  --lambda_scale 0.05 \
  --lambda_opacity_entropy 0.001 \
  --use_appearance \
  --mono_depth_dir dataset/OldHospital/mono_depth \
  --dist_start_iter 3000 \
  --depth_start_iter 2000 \
  --normal_start_iter 2000 \
  --densify_until_iter 25000 \
  --densify_grad_threshold 0.00015 \
  --opacity_reset_interval 3000 \
  --save_iterations 10000 20000 30000 40000 50000 \
  --test_iterations 10000 20000 30000 40000 50000
