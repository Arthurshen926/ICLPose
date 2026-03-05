#!/bin/bash
# Retrain OldHospital v3 2DGS — retrain7 (comprehensive improvements)
#
# Changes vs retrain6:
#   1. NO mask filtering (--use_mask not set) — because localization has no masks
#   2. Full resolution (longest_edge=0) — 1920×1080, no downsampling
#   3. Calibrated distortion loss (lambda_dist=0.01) — gsplat dist values ~0.6
#   4. Monocular depth supervision (lambda_depth=0.1) — DPT-Large Pearson loss
#   5. Normal consistency (lambda_normal=0.05) — from iter 2000
#   6. Scale regularization (lambda_scale=0.1) — from iter 3000
#   7. dist_start_iter=3000 (earlier, since properly calibrated now)
#
# Expected: better geometry, fewer floaters, no smearing artifacts
# Full res 1920x1080 needs ~15-20GB VRAM → use GPU 0 (24GB)

set -e
cd "$(dirname "$0")/.."

MODEL_DIR=output/2dgs_models/OldHospital/v3_retrain7
LOG_FILE=logs/oldhospital_v3_retrain7.log

echo "=== OldHospital v3 retrain7: full-res + dist + depth + no mask ==="
echo "  Model dir: $MODEL_DIR"
echo "  Log: $LOG_FILE"

rm -rf "$MODEL_DIR"
mkdir -p logs

export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')"):$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0 python -m feature_3dgs.train_2dgs_geometry \
  --source_dir dataset/OldHospital \
  --model_dir "$MODEL_DIR" \
  --images . \
  --iterations 35000 \
  --longest_edge 0 \
  --lambda_dist 0.01 \
  --lambda_normal 0.05 \
  --lambda_depth 0.1 \
  --mono_depth_dir dataset/OldHospital/mono_depth \
  --lambda_scale 0.1 \
  --scale_reg_threshold 1.0 \
  --dist_start_iter 3000 \
  --normal_start_iter 2000 \
  --depth_start_iter 2000 \
  --densify_until_iter 20000 \
  --densify_grad_threshold 0.00018 \
  --position_lr_init 0.000016 \
  --scaling_lr 0.001 \
  --opacity_lr 0.05 \
  --save_iterations 7000 14500 20500 35000 \
  --test_iterations 7000 14500 20500 35000 \
  2>&1 | tee "$LOG_FILE"

echo "=== Training complete ==="
