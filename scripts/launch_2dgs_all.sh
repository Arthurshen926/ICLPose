#!/bin/bash
# Launch 3 parallel 2DGS training jobs on GPUs 0, 1, 3
# Batch sizes tuned for RTX 4090 (24GB): maximize VRAM utilization
set -e
cd /root/ICLPose

# CUDA environment: use system CUDA 11.6 for headers, conda nvcc 11.8 for compilation
export CUDA_HOME=/usr/local/cuda-11.6
export PATH=/root/miniconda3/envs/iclpose/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0+PTX"
PYTHON=/root/miniconda3/envs/iclpose/bin/python

# Logs in local output/logs/ directory (not /tmp)
LOG_DIR=output/logs
mkdir -p $LOG_DIR
mkdir -p output/2dgs_models/OldHospital/v3_retrain38g
mkdir -p output/2dgs_models/stairs/v1_baseline
mkdir -p output/2dgs_models/room_0/v1_baseline

echo "[$(date)] Launching 3 parallel 2DGS training jobs..."

# GPU 0: OldHospital (WildGaussians, 256x3 MLP, output_scale=0.5, 64d embed)
# 1920x1080 images + WG MLP → batch_size=16 targets ~18GB VRAM
CUDA_VISIBLE_DEVICES=0 $PYTHON -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir output/2dgs_models/OldHospital/v3_retrain38g \
    --iterations 30000 --batch_size 16 --use_mask --random_background \
    --use_appearance --wildgaussians --no_dino_uncertainty \
    --wg_output_scale 0.5 --wg_hidden_dim 256 --wg_n_hidden 3 --wg_image_embed_dim 64 \
    --lambda_dssim 0.2 --lambda_normal 0 --lambda_dist 0 --lambda_scale 0 --lambda_depth 0 \
    --densify_until_iter 15000 --densify_grad_threshold 0.0002 --max_gaussians 200000 \
    --opacity_reset_interval 30001 \
    --appearance_lr_init 5e-4 --gaussian_emb_lr 5e-3 --image_emb_lr 1e-3 --appearance_reg 0.001 \
    --grad_clip_max_norm 1.0 \
    --test_iterations 1000 3000 5000 10000 15000 20000 25000 30000 \
    --save_iterations 10000 20000 30000 --wg_test_opt_steps 0 \
    > output/2dgs_models/OldHospital/v3_retrain38g/train.log 2>&1 &
PID_OH=$!
echo "[$(date)] OldHospital launched on GPU 0, PID=$PID_OH (batch_size=16)"

# GPU 1: stairs (standard 2DGS with geometry losses)
# 640x480 images → batch_size=32 targets ~14GB VRAM
CUDA_VISIBLE_DEVICES=1 $PYTHON -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/stairs \
    --model_dir output/2dgs_models/stairs/v1_baseline \
    --iterations 30000 --batch_size 32 --random_background \
    --lambda_dssim 0.2 --lambda_normal 0.05 --lambda_dist 0.01 --lambda_scale 0 --lambda_depth 0 \
    --densify_until_iter 15000 --densify_grad_threshold 0.0002 --max_gaussians 500000 \
    --opacity_reset_interval 3000 --grad_clip_max_norm 1.0 \
    --test_iterations 1000 3000 5000 10000 15000 20000 25000 30000 \
    --save_iterations 10000 20000 30000 \
    > output/2dgs_models/stairs/v1_baseline/train.log 2>&1 &
PID_ST=$!
echo "[$(date)] stairs launched on GPU 1, PID=$PID_ST (batch_size=32)"

# GPU 3: room_0 (standard 2DGS with geometry losses)
# 640x480 images → batch_size=32 targets ~14GB VRAM
CUDA_VISIBLE_DEVICES=3 $PYTHON -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/room_0 \
    --model_dir output/2dgs_models/room_0/v1_baseline \
    --iterations 30000 --batch_size 32 --random_background \
    --lambda_dssim 0.2 --lambda_normal 0.05 --lambda_dist 0.01 --lambda_scale 0 --lambda_depth 0 \
    --densify_until_iter 15000 --densify_grad_threshold 0.0002 --max_gaussians 300000 \
    --opacity_reset_interval 3000 --grad_clip_max_norm 1.0 \
    --test_iterations 1000 3000 5000 10000 15000 20000 25000 30000 \
    --save_iterations 10000 20000 30000 \
    > output/2dgs_models/room_0/v1_baseline/train.log 2>&1 &
PID_R0=$!
echo "[$(date)] room_0 launched on GPU 3, PID=$PID_R0 (batch_size=32)"

echo ""
echo "All jobs launched. Monitor with:"
echo "  tail -f output/2dgs_models/OldHospital/v3_retrain38g/train.log"
echo "  tail -f output/2dgs_models/stairs/v1_baseline/train.log"
echo "  tail -f output/2dgs_models/room_0/v1_baseline/train.log"
echo ""
echo "PIDs: OH=$PID_OH  ST=$PID_ST  R0=$PID_R0"

# Wait for all to finish
wait $PID_OH $PID_ST $PID_R0
echo "[$(date)] All 3 training jobs finished!"
