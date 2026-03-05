#!/bin/bash
# Retrain24: Bounded appearance network (architectural constraint)
#
# Root cause analysis from 6 experiments:
#   - No appearance (retrain19): stagnated at 16.42 (color disagreements)
#   - Full appearance, no reg (retrain17): peaked 16.99@15k, crashed to 15.70
#   - Embedding L2 reg (retrain21): raw crashed, corrected good (16.65)
#   - Identity reg=0.1 (retrain22): stable but slow (16.65@10k)
#   - Identity reg=0.01 (retrain23): still too weak (crashed to 16.00)
#
# NEW APPROACH: Architectural constraint instead of regularization!
#   - scale range: [0.8, 1.2] (±20%) — enough for exposure variation
#   - bias range: [-0.05, +0.05] — enough for white balance
#   - Reg=0.01 is now safe because the MLP output is bounded
#   - tanh for bias (smooth gradient everywhere)
#
# Key: the HARD BOUND prevents catastrophic overfitting that no
# regularization weight could reliably prevent. The MLP can still
# optimize freely within the bounded range.
#
# Expected: match retrain17's 16.99 convergence speed + extend past 15k

set -e

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0

nohup python -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir output/2dgs_models/OldHospital/v3_retrain24 \
    --iterations 30000 \
    --batch_size 4 \
    --longest_edge 0 \
    --random_background \
    --use_mask \
    --use_appearance \
    --appearance_lr_init 1e-3 \
    --appearance_lr_final 1e-5 \
    --appearance_scale_range 0.4 \
    --appearance_bias_range 0.05 \
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
    > output/retrain24.log 2>&1 &

echo "Retrain24 started on GPU 0. PID: $!"
echo "Log: output/retrain24.log"
echo "Monitor: tail -f output/retrain24.log"
