#!/bin/bash
# Retrain7 FINAL: batch=4, 15000 iters
# Profiling proved batch=4 is optimal: 17.4 views/s vs batch=12 13.1 views/s
# batch=4 uses only 5GB → leaves headroom for densification
# Total views: 4*15000 = 60,000 (2x standard 30k, more than enough)
# 估计时间: 15000/4.35 ≈ 57 分钟
#
# All 5 improvements:
#   1. Distortion loss calibrated (lambda_dist=0.01 for gsplat)
#   2. No mask filtering
#   3. Full resolution (1920x1080)
#   4. Monocular depth supervision (DPT-Large, Pearson loss)
#   5. Normal supervision

set -e

export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')"):$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0 python -m feature_3dgs.train_2dgs_geometry \
  --source_dir dataset/OldHospital \
  --model_dir output/2dgs_models/OldHospital/v3_retrain7 \
  --images "" \
  --iterations 15000 \
  --batch_size 4 \
  --longest_edge 0 \
  --lambda_dist 0.01 \
  --lambda_normal 0.05 \
  --lambda_depth 0.1 \
  --lambda_scale 0.1 \
  --mono_depth_dir dataset/OldHospital/mono_depth \
  --dist_start_iter 2000 \
  --depth_start_iter 1500 \
  --normal_start_iter 1500 \
  --densify_until_iter 9000 \
  --densify_grad_threshold 0.00018 \
  --save_iterations 5000 10000 15000 \
  --test_iterations 5000 10000 15000
