#!/bin/bash
# Retrain19: Fix overfitting — NO appearance network + LR decay + short training
#
# Root cause analysis:
#   AppearanceNetwork applies per-image affine correction during TRAINING only.
#   At TEST time, raw Gaussian colors are evaluated without correction.
#   As training progresses, the appearance net absorbs more "work", corrupting
#   the base Gaussian colors → test PSNR degrades (16.99→15.81 over 15k→60k).
#
# Key changes from retrain17:
#   1. NO appearance network (remove train/test distribution mismatch)
#   2. Short training: 25k iterations (peak was at 15k in retrain17)
#   3. All-parameter LR decay (lr_decay_factor=0.1, 10x decay)
#   4. Mild depth supervision (lambda_depth=0.03, start@3000)
#   5. Opacity reset every 5000 iters (clean up floaters)
#   6. Dense checkpoint saving to find optimal stopping point
#
# Expected: PSNR should NOT degrade after peak, and may go higher than 17 dB
#   since Gaussian colors won't be corrupted by appearance network.

set -e

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0

nohup python -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir output/2dgs_models/OldHospital/v3_retrain19 \
    --iterations 25000 \
    --batch_size 4 \
    --longest_edge 0 \
    --random_background \
    --use_mask \
    --sh_degree 3 \
    \
    --position_lr_init 0.00005 \
    --position_lr_final 0.0000005 \
    --feature_lr 0.005 \
    --f_rest_lr_divisor 5.0 \
    --opacity_lr 0.1 \
    --scaling_lr 0.002 \
    --rotation_lr 0.002 \
    --lr_decay_factor 0.1 \
    \
    --lambda_dssim 0.2 \
    --lambda_dist 0.0 \
    --lambda_normal 0.0 \
    --normal_start_iter 999999 \
    --dist_start_iter 999999 \
    --lambda_depth 0.03 \
    --depth_start_iter 3000 \
    --mono_depth_dir dataset/OldHospital/mono_depth \
    --lambda_scale 0.0 \
    \
    --densify_from_iter 500 \
    --densify_until_iter 15000 \
    --densification_interval 100 \
    --densify_grad_threshold 0.00015 \
    --opacity_reset_interval 5000 \
    --opacity_reset_value 0.01 \
    --prune_dead_threshold 0.005 \
    --percent_dense 0.005 \
    \
    --save_iterations 3000 5000 7000 10000 12000 15000 18000 20000 22000 25000 \
    --test_iterations 3000 5000 7000 10000 12000 15000 18000 20000 22000 25000 \
    > output/retrain19.log 2>&1 &

echo "Retrain19 started on GPU 0. PID: $!"
echo "Log: output/retrain19.log"
echo "Monitor: tail -f output/retrain19.log"
