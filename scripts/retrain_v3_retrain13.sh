#!/bin/bash
# Retrain13: Fixed sky/BG mismatch + dual-mask strategy
#
# ═══ Analysis of Retrain12 failure (14.94 dB @ 10k) ═══
#   CRITICAL BUG: sky GT set to white(1.0) but black BG renders sky as black(0.0)
#     → Every sky pixel: L1 error = 1.0 (catastrophic)
#     → FIX: Dual-mask strategy in code:
#       - rgb_mask = obj_mask & distort_mask  (keep sky for RGB training)
#       - geo_mask = rgb_mask & sky_mask      (exclude sky for depth/normal/dist)
#       → Gaussians learn sky appearance (good eval PSNR)
#       → Depth/normal/dist skip sky (DPT unreliable on sky)
#
# ═══ Analysis of Retrain11 plateau (16.2 dB) ═══
#   - Gaussian count 464k→1M but no PSNR gain = wasted capacity
#   - 646 extreme loss warnings from normal/dist regularization
#   - FIX: Keep delayed regularization from retrain12
#
# ═══ Changes from Retrain12 ═══
#   = Same mask/regularization parameters (code fix handles the rest)
#   + More frequent test eval: add 5000 and 15000 checkpoints
#   + Reduced densify_until_iter 30k→25k to limit Gaussian bloat
#   + Increased prune_dead_threshold 0.003→0.005 to prune floaters earlier
#
# Expected outcome: ~17-18 dB @ 10k (sky now rendered correctly)

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=0 python -u feature_3dgs/train_2dgs_geometry.py \
  --source_dir dataset/OldHospital \
  --model_dir output/2dgs_models/OldHospital/v3_retrain13 \
  --iterations 70000 \
  --batch_size 4 \
  --longest_edge 0 \
  \
  --use_mask \
  --mask_path dataset/OldHospital/masks.pkl \
  \
  --position_lr_init 0.00005 \
  --position_lr_final 0.0000005 \
  --feature_lr 0.005 \
  --f_rest_lr_divisor 5.0 \
  --opacity_lr 0.1 \
  --scaling_lr 0.002 \
  --rotation_lr 0.002 \
  \
  --lambda_dssim 0.5 \
  --lambda_dist 0.01 \
  --lambda_normal 0.02 \
  --normal_start_iter 15000 \
  --dist_start_iter 12000 \
  --lambda_depth 0.05 \
  --mono_depth_dir dataset/OldHospital/mono_depth \
  --depth_start_iter 5000 \
  --lambda_scale 0.01 \
  --scale_reg_threshold 1.0 \
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
  --save_iterations 5000 10000 15000 20000 30000 40000 50000 60000 70000 \
  --test_iterations 5000 10000 15000 20000 30000 40000 50000 60000 70000
