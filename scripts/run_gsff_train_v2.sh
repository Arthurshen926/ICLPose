#!/bin/bash
# Retrain GSFFs with improved fine-level losses and encoder
# Warmstarts from existing coarse features, trains fine from scratch
#
# Improvements over v1:
#   1. Fine level gets full losses: NCE + Prototypical + CE (v1 only had NCE)  
#   2. Fine encoder receives upsampled DINOv2 guidance (v1 was CNN-only)
#   3. Fine starts at iter 5000 (v1 at 10000)
#   4. Best checkpoint tracks fine cos_sim (v1 tracked total loss, which was wrong)
#   5. Extended training: 80K iters (v1: 50K)

cd /root/ICLPose

export CUDA_VISIBLE_DEVICES=0

python scripts/train_gsff.py \
    --source_dir dataset/OldHospital \
    --model_path output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply \
    --cameras_json output/2dgs_models/OldHospital/v7_depth/cameras.json \
    --output_dir output/gsff/OldHospital_v2 \
    --total_iters 80000 \
    --phase1_iters 3000 \
    --lr_triplane 5e-4 \
    --lr_encoder 2e-4 \
    --save_freq 5000 \
    --warmstart output/gsff/OldHospital/checkpoints/final.pth \
    2>&1 | tee output/gsff/OldHospital_v2/train.log
