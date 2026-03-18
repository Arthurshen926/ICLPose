#!/bin/bash
# Restart room_0 2DGS training with fixes for previous divergence:
#   1. Subsampled initial point cloud (depth-backprojected, was 153K → now 50K)
#   2. Tighter max_gaussians (100K vs original 300K)
#   3. Disable opacity reset (interval > iterations → never resets)
#   4. No normal/dist regularization (synthetic scene with GT depth)
#   5. Loss guard in code: negative loss detection + optimizer state reset
#
# Usage: bash scripts/restart_room0.sh [GPU_ID]
#   GPU_ID defaults to first available GPU with <2GB used

set -e
cd /root/ICLPose

export CUDA_HOME=/usr/local/cuda-11.6
export PATH=/root/miniconda3/envs/iclpose/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0+PTX"
PYTHON=/root/miniconda3/envs/iclpose/bin/python

# Auto-detect or use specified GPU
GPU_ID=${1:-""}
if [ -z "$GPU_ID" ]; then
    echo "No GPU specified, checking availability..."
    # Find a GPU with low memory usage (< 2GB used) among GPUs 2-5
    GPU_ID=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
        awk -F', ' '$1 >= 2 && $2 < 2000 {print $1; exit}')
    if [ -z "$GPU_ID" ]; then
        echo "ERROR: No free GPU found (GPUs 2-5 all occupied). Retry later or specify GPU_ID."
        echo "Usage: bash scripts/restart_room0.sh <GPU_ID>"
        exit 1
    fi
fi

echo "[$(date)] Restarting room_0 on GPU $GPU_ID"

MODEL_DIR=output/2dgs_models/room_0/v2_fixed
mkdir -p "$MODEL_DIR"

CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/room_0 \
    --model_dir "$MODEL_DIR" \
    --iterations 30000 --batch_size 32 --random_background \
    --lambda_dssim 0.2 \
    --lambda_normal 0 --lambda_dist 0 --lambda_scale 0 --lambda_depth 0 \
    --densify_until_iter 15000 --densify_grad_threshold 0.0002 \
    --max_gaussians 100000 \
    --max_init_points 50000 \
    --opacity_reset_interval 30001 \
    --grad_clip_max_norm 1.0 \
    --test_iterations 1000 3000 5000 10000 15000 20000 25000 30000 \
    --save_iterations 10000 20000 30000 \
    > "$MODEL_DIR/train.log" 2>&1 &

PID=$!
echo "[$(date)] room_0 v2_fixed launched on GPU $GPU_ID, PID=$PID"
echo "  Model dir: $MODEL_DIR"
echo "  Key params: max_init_points=50K, max_gaussians=100K, no opacity reset, no normal reg"
echo "  Monitor: tail -f $MODEL_DIR/train.log"
echo "  PSNR check: grep PSNR $MODEL_DIR/train.log"
