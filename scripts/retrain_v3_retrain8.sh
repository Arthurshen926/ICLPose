#!/bin/bash
# Retrain8: Comprehensive stability fixes + longer training
#
# Fixes applied to train_2dgs_geometry.py:
#   1. Stable Pearson depth loss (std guard, output clamping, larger epsilon)
#   2. Loss clipping: skip backward when loss > 10.0
#   3. Gradient clipping: max_norm=1.0 before optimizer step
#   4. far_plane: 10000 → 500 (OldHospital max depth ~68)
#   5. Mono depth pre-caching (avoid disk I/O per iteration)
#   6. Tighter opacity pruning (0.005 → 0.01 threshold)
#
# Training config changes vs retrain7:
#   - iterations: 15000 → 30000 (was undertrained)
#   - densify_until_iter: 9000 → 15000 (more densification time)
#   - opacity_reset_interval: 3000 (standard)
#   - save checkpoints at 7k, 15k, 25k, 30k
#   - test at same iterations

set -e

export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')"):$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0 python -m feature_3dgs.train_2dgs_geometry \
  --source_dir dataset/OldHospital \
  --model_dir output/2dgs_models/OldHospital/v3_retrain8 \
  --images "" \
  --iterations 30000 \
  --batch_size 4 \
  --longest_edge 0 \
  --lambda_dist 0.01 \
  --lambda_normal 0.05 \
  --lambda_depth 0.1 \
  --lambda_scale 0.1 \
  --mono_depth_dir dataset/OldHospital/mono_depth \
  --dist_start_iter 3000 \
  --depth_start_iter 2000 \
  --normal_start_iter 2000 \
  --densify_until_iter 15000 \
  --densify_grad_threshold 0.0002 \
  --opacity_reset_interval 3000 \
  --save_iterations 7000 15000 25000 30000 \
  --test_iterations 7000 15000 25000 30000
