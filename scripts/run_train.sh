#!/bin/bash
# ICPoseNetV3 Training v6 - Flow Loss + Curriculum
cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=0 python -u train_v3.py \
    --scales fine_sd fine_dino \
    --epochs 100 \
    --batch_size 1 \
    --grad_accum 1 \
    --lr 1e-3 \
    --grad_clip 10.0 \
    --num_iters 1 \
    --gamma 0.8 \
    --lambda_trans 0.5 \
    --lambda_flow 0.1 \
    --curriculum \
    --start_noise_rot 2.0 \
    --start_noise_trans 0.03 \
    --warmup_epochs 3 \
    --rampup_epochs 30 \
    --noise_rot 15.0 \
    --noise_trans 0.3 \
    --val_noise_rot 10.0 \
    --val_noise_trans 0.2 \
    --output_dir output/v3_train_v6_flow \
    --log_interval 20
