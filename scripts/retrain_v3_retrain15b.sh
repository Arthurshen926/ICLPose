#!/bin/bash
# Retrain15b: Control — our LRs + NO regularization
#
# Same as retrain14 but with all regularization completely disabled.
# This isolates the effect of learning rates (compare with retrain15).
#
# retrain14 had lambda_dist=0.01 and lambda_normal=0.02 from 12k/15k
# which caused degradation. Here we set them all to 0.
#
# ═══ Comparison matrix ═══
# retrain15:  standard LRs + no reg  (test: are standard LRs better?)
# retrain15b: our LRs + no reg       (control: how do our LRs performw/o reg?)

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=1 python -u feature_3dgs/train_2dgs_geometry.py \
  --source_dir dataset/OldHospital \
  --model_dir output/2dgs_models/OldHospital/v3_retrain15b \
  --iterations 70000 \
  --batch_size 4 \
  --longest_edge 0 \
  \
  --position_lr_init 0.00005 \
  --position_lr_final 0.0000005 \
  --feature_lr 0.005 \
  --f_rest_lr_divisor 5.0 \
  --opacity_lr 0.1 \
  --scaling_lr 0.002 \
  --rotation_lr 0.002 \
  \
  --lambda_dssim 0.2 \
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
  --opacity_reset_interval 5000 \
  --opacity_reset_value 0.05 \
  --prune_dead_threshold 0.005 \
  --percent_dense 0.005 \
  \
  --save_iterations 5000 7000 10000 15000 20000 30000 40000 50000 60000 70000 \
  --test_iterations 5000 7000 10000 15000 20000 30000 40000 50000 60000 70000
