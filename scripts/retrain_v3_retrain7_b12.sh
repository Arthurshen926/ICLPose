#!/bin/bash
# Retrain7 optimized: batch=12, 12000 iters
# GPU compute-bound at ~18 views/s → batch=12 gives best gradient quality
# Total views: 12*12000 = 144,000 (equivalent to 4*35000=140,000)
# Estimated time: ~2 hours
#
# All 5 improvements:
#   1. Distortion loss calibrated (lambda_dist=0.01 for gsplat)
#   2. No mask filtering
#   3. Full resolution (1920x1080)
#   4. Monocular depth supervision (DPT-Large, Pearson loss)
#   5. Normal supervision
#   + batch=12 parallel rendering + image pre-caching

set -e

export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')"):$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0 python -m feature_3dgs.train_2dgs_geometry \
  --source_dir dataset/OldHospital \
  --model_dir output/2dgs_models/OldHospital/v3_retrain7 \
  --images "" \
  --iterations 12000 \
  --batch_size 12 \
  --longest_edge 0 \
  --lambda_dist 0.01 \
  --lambda_normal 0.05 \
  --lambda_depth 0.1 \
  --lambda_scale 0.1 \
  --mono_depth_dir dataset/OldHospital/mono_depth \
  --dist_start_iter 1000 \
  --depth_start_iter 700 \
  --normal_start_iter 700 \
  --densify_until_iter 7000 \
  --densify_grad_threshold 0.00018 \
  --save_iterations 3000 6000 9000 12000 \
  --test_iterations 3000 6000 9000 12000
