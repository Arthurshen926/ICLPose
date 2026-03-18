#!/bin/bash
# Launch fixed 2DGS depth-supervised training (v3: fixed depth bugs)
# Fixes: 1) pearson_depth_loss valid_mask cast to bool
#        2) sensor depth clamp >20m as invalid
#        3) l1_depth_loss filters sd<20m, rd<50m
#        4) Reduced lambda_sensor_depth from 0.5 to 0.1 for stability
set -e
cd /root/ICLPose

export CUDA_HOME=/usr/local/cuda-11.6
export PATH=/root/miniconda3/envs/iclpose/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0+PTX"
PYTHON=/root/miniconda3/envs/iclpose/bin/python

echo "[$(date)] Launching fixed 2DGS depth training..."

# ─────────────────────────────────────────────────────────────────────────────
# GPU 1: stairs (fresh start with fixed depth)
# Key changes: lambda_sensor_depth 0.5 -> 0.1, depth bugs fixed
# ─────────────────────────────────────────────────────────────────────────────
STAIRS_DIR=output/2dgs_models/stairs/v3_depth_fixed
mkdir -p $STAIRS_DIR

CUDA_VISIBLE_DEVICES=1 $PYTHON -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/stairs \
    --model_dir $STAIRS_DIR \
    --iterations 30000 --batch_size 32 --random_background \
    --lambda_dssim 0.2 \
    --lambda_normal 0.05 --normal_start_iter 5000 \
    --lambda_dist 0.01 --dist_start_iter 7000 \
    --lambda_scale 0.1 --scale_reg_threshold 0.5 \
    --lambda_opacity_entropy 0.01 \
    --sensor_depth_dir dataset/stairs --lambda_sensor_depth 0.1 --depth_start_iter 1000 \
    --densify_until_iter 15000 --densify_grad_threshold 0.0002 --max_gaussians 500000 \
    --opacity_reset_interval 30001 --grad_clip_max_norm 1.0 \
    --test_iterations 1000 3000 5000 10000 15000 20000 25000 30000 \
    --save_iterations 10000 20000 30000 \
    > $STAIRS_DIR/train.log 2>&1 &
PID_ST=$!
echo "[$(date)] stairs launched on GPU 1, PID=$PID_ST -> $STAIRS_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# GPU 4: OldHospital (fresh start with fixed pearson depth mask)
# Key changes: valid_mask now correctly cast to bool
# ─────────────────────────────────────────────────────────────────────────────
OH_DIR=output/2dgs_models/OldHospital/v5_depth_fixed
mkdir -p $OH_DIR

CUDA_VISIBLE_DEVICES=4 $PYTHON -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir $OH_DIR \
    --iterations 30000 --batch_size 16 --use_mask --random_background \
    --use_appearance --wildgaussians --no_dino_uncertainty \
    --wg_output_scale 0.5 --wg_hidden_dim 256 --wg_n_hidden 3 --wg_image_embed_dim 64 \
    --lambda_dssim 0.2 \
    --lambda_normal 0.05 --normal_start_iter 5000 \
    --lambda_dist 0.01 --dist_start_iter 7000 \
    --lambda_scale 0.1 --scale_reg_threshold 1.0 \
    --mono_depth_dir dataset/OldHospital/mono_depth --lambda_depth 0.05 --depth_start_iter 3000 \
    --densify_until_iter 15000 --densify_grad_threshold 0.0002 --max_gaussians 200000 \
    --opacity_reset_interval 30001 \
    --appearance_lr_init 5e-4 --gaussian_emb_lr 5e-3 --image_emb_lr 1e-3 --appearance_reg 0.001 \
    --grad_clip_max_norm 1.0 \
    --test_iterations 1000 3000 5000 10000 15000 20000 25000 30000 \
    --save_iterations 10000 20000 30000 --wg_test_opt_steps 0 \
    > $OH_DIR/train.log 2>&1 &
PID_OH=$!
echo "[$(date)] OldHospital launched on GPU 4, PID=$PID_OH -> $OH_DIR"

echo "[$(date)] All jobs launched. Monitor with:"
echo "  tail -f $STAIRS_DIR/train.log"
echo "  tail -f $OH_DIR/train.log"
