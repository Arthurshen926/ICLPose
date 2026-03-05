#!/bin/bash
# Retrain21: Targeted appearance regularization — NO global LR decay
#
# Key insight from retrain20: Global lr_decay_factor=0.1 over 25k steps
# decays Gaussian parameter LRs too aggressively (63% at 5k, 25% at 15k).
# This killed Gaussian convergence — retrain20 stagnated at 16.52 dB
# while retrain17 (no global decay) reached 16.78 at 5k.
#
# Fix: The problem is only the appearance network overfitting, not
# the Gaussian params. So we:
#   1. lr_decay_factor=1.0 — NO global LR decay (standard gsplat schedule)
#   2. appearance_lr: 1e-3 → 1e-5 — aggressive decay on appearance ONLY
#   3. appearance_reg=0.01 — L2 reg on per-image embeddings
#   4. 30k iterations (longer than 25k to give more learning time)
#   5. Eval-time appearance correction using mean embedding (new feature)
#
# Expected: PSNR should follow retrain17's trajectory (16.55→16.78→16.99)
# but NOT degrade after 15k because appearance is regularized.
# The eval-time corrected PSNR should be even higher.

set -e

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0

nohup python -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir output/2dgs_models/OldHospital/v3_retrain21 \
    --iterations 30000 \
    --batch_size 4 \
    --longest_edge 0 \
    --random_background \
    --use_mask \
    --use_appearance \
    --appearance_lr_init 1e-3 \
    --appearance_lr_final 1e-5 \
    --appearance_reg 0.01 \
    --sh_degree 3 \
    \
    --position_lr_init 0.00005 \
    --position_lr_final 0.0000005 \
    --feature_lr 0.005 \
    --f_rest_lr_divisor 5.0 \
    --opacity_lr 0.1 \
    --scaling_lr 0.002 \
    --rotation_lr 0.002 \
    --lr_decay_factor 1.0 \
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
    --save_iterations 3000 5000 7000 10000 12000 15000 18000 20000 22000 25000 30000 \
    --test_iterations 3000 5000 7000 10000 12000 15000 18000 20000 22000 25000 30000 \
    > output/retrain21.log 2>&1 &

echo "Retrain21 started on GPU 0. PID: $!"
echo "Log: output/retrain21.log"
echo "Monitor: tail -f output/retrain21.log"
