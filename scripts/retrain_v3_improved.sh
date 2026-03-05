#!/bin/bash
# Retrain OldHospital v3 2DGS — improved v3 (retrain6)
# 
# Key learnings from previous attempts:
#   - lambda_dist>0 hurts appearance quality (PSNR drops after distortion activates)
#   - lambda_scale=0.1 effectively prevents floaters
#   - Post-densification periodic pruning removes remaining large Gaussians
#   - Save checkpoints BEFORE opacity reset (code fix applied)
#   - opacity_reset_interval=3000 resets happen only during densification
#
# This run: scale_reg only (no distortion loss) + post-densify pruning
#   lambda_dist = 0         — NO distortion loss (preserves appearance)
#   lambda_scale = 0.1      — gentle scale regularization for floater suppression
#   scale_reg_threshold = 1.0 — penalize only truly huge Gaussians
#   densify_grad_threshold = 0.00018 — slightly more aggressive for thin structures
#   densify_until_iter = 20000 — standard duration
#   iterations = 35000      — enough refinement after densification ends

set -e
cd "$(dirname "$0")/.."

MODEL_DIR=output/2dgs_models/OldHospital/v3_improved_v2
LOG_FILE=logs/oldhospital_v3_retrain6.log

echo "=== OldHospital v3 retrain6: scale_reg + pruning (no dist) ==="
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
  --longest_edge 1280 \
  --lambda_dist 0 \
  --lambda_normal 0.05 \
  --lambda_scale 0.1 \
  --scale_reg_threshold 1.0 \
  --dist_start_iter 7000 \
  --normal_start_iter 2000 \
  --densify_until_iter 20000 \
  --densify_grad_threshold 0.00018 \
  --position_lr_init 0.000016 \
  --scaling_lr 0.001 \
  --opacity_lr 0.05 \
  --save_iterations 7000 14500 20500 35000 \
  --test_iterations 7000 14500 20500 35000 \
  2>&1 | tee "$LOG_FILE"

echo "=== Training complete ==="
