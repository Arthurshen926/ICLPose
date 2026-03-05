#!/bin/bash
# Retrain OldHospital v3 2DGS with mild distortion loss (lambda_dist=1.0)
# to eliminate floater artifacts while avoiding the collapse issue.
#
# Key changes vs retrain4 (lambda_dist=0):
#   lambda_dist = 1.0  (mild regularization, was 0; 2DGS paper used 100)
#   dist_start_iter = 3500  (after opacity_reset at 3000 + 500 cooldown)
#   All other params unchanged from retrain4

set -e
cd "$(dirname "$0")/.."

MODEL_DIR=output/2dgs_models/OldHospital/v3_dist1
LOG_FILE=logs/oldhospital_v3_retrain5.log

echo "=== OldHospital v3 retrain5: lambda_dist=1.0 ==="
echo "  Model dir: $MODEL_DIR"
echo "  Log: $LOG_FILE"

# Clean any previous attempt
rm -rf "$MODEL_DIR"

export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')"):$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0 python -m feature_3dgs.train_2dgs_geometry \
  --source_dir dataset/OldHospital \
  --model_dir "$MODEL_DIR" \
  --images . \
  --iterations 30000 \
  --longest_edge 1280 \
  --lambda_dist 1.0 \
  --lambda_normal 0.05 \
  --dist_start_iter 3500 \
  --normal_start_iter 2000 \
  --position_lr_init 0.000016 \
  --scaling_lr 0.001 \
  --opacity_lr 0.05 \
  --save_iterations 7000 15000 30000 \
  --test_iterations 7000 15000 30000 \
  2>&1 | tee "$LOG_FILE"

echo "=== Training complete ==="
