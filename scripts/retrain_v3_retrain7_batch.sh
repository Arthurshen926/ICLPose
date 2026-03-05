#!/bin/bash
# Retrain7 with batch parallel rendering (batch_size=4)
# All 5 improvements + GPU acceleration:
#   1. Distortion loss calibrated (lambda_dist=0.01 for gsplat)
#   2. No mask filtering (localization has no masks)  
#   3. Full resolution (1920x1080, no downsampling)
#   4. Monocular depth supervision (DPT-Large, Pearson loss)
#   5. Normal supervision (already working)
#   6. NEW: Batch parallel rendering (4 views/step, single gsplat call)
#   7. NEW: Image pre-caching (CPU RAM, no disk I/O)

set -e

export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')"):$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0 python -m feature_3dgs.train_2dgs_geometry \
  --source_dir dataset/OldHospital \
  --model_dir output/2dgs_models/OldHospital/v3_retrain7 \
  --images "" \
  --iterations 35000 \
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
  --densify_until_iter 20000 \
  --densify_grad_threshold 0.00018 \
  --save_iterations 7000 14500 20500 35000 \
  --test_iterations 7000 14500 20500 35000
