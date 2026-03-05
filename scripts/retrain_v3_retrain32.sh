#!/bin/bash
# Retrain32: lambda_dssim=0.5 + freeze@5k
#
# ═══ Analysis ═══
# retrain31 (lambda_dssim=0.2, freeze@5k) shows the best trajectory so far:
#   3k=16.71, 5k=16.84(freeze), 7k=16.71(dip), 10k=16.82, 12k=16.89, 15k=16.91
#   Still climbing at 15k!
#
# HYPOTHESIS: Increasing lambda_dssim from 0.2 to 0.5 gives more weight to
# structural similarity loss. This encourages the model to preserve structural
# features rather than just minimizing pixel-level error. Many successful
# 3DGS implementations use lambda_dssim=0.5 for outdoor scenes.
#
# Everything else identical to retrain31.

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=GPU_ID nohup python -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir output/2dgs_models/OldHospital/v3_retrain32 \
    --iterations 70000 \
    --batch_size 4 \
    --longest_edge 0 \
    \
    --random_background \
    --use_appearance \
    --use_mask \
    --appearance_freeze_iter 5000 \
    --appearance_scale_range 2.0 \
    --appearance_bias_range 0.5 \
    \
    --sh_degree 3 \
    --position_lr_init 0.00005 \
    --position_lr_final 0.0000005 \
    --feature_lr 0.005 \
    --f_rest_lr_divisor 5.0 \
    --opacity_lr 0.1 \
    --scaling_lr 0.002 \
    --rotation_lr 0.002 \
    --lr_decay_factor 1.0 \
    \
    --lambda_dssim 0.5 \
    --lambda_dist 0.0 \
    --lambda_normal 0.0 \
    --normal_start_iter 999999 \
    --dist_start_iter 999999 \
    --lambda_depth 0.0 \
    --lambda_scale 0.0 \
    \
    --densify_from_iter 500 \
    --densify_until_iter 25000 \
    --densification_interval 100 \
    --densify_grad_threshold 0.00015 \
    --opacity_reset_interval 999999 \
    --opacity_reset_value 0.01 \
    --prune_dead_threshold 0.005 \
    --percent_dense 0.005 \
    \
    --save_iterations 3000 5000 7000 10000 12000 15000 20000 25000 30000 40000 50000 60000 70000 \
    --test_iterations 3000 5000 7000 10000 12000 15000 20000 25000 30000 40000 50000 60000 70000 \
    > output/retrain32.log 2>&1 &

echo "PID: $!"
echo "Log: output/retrain32.log"
