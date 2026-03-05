#!/bin/bash
# Retrain11: Systematic geometry + rendering quality improvement
#
# Root cause analysis from retrain10 (16.29 dB @ 20k peak):
#   1. CRITICAL BUG: Post-densification pruning starts at iter 20000, but last
#      opacity reset at 18000. Gaussians only had 2000 iters to recover from
#      opacity=0.01 → immediately pruned by dead_mask < 0.01!
#   2. 43.7% of Gaussians have opacity < 0.1 → near-invisible, wasting capacity
#   3. f_rest_lr = feature_lr/20 = 0.000125 is too low for high-order SH convergence
#   4. Only 218k Gaussians (post-pruning) for 1920×1080 is insufficient
#   5. Position LR too conservative for outdoor scene geometry
#
# ═══ Phase 1: Geometry Fixes ═══
#   - densify_until_iter: 20000 → 30000 (more time for geometry growth)
#   - grad_threshold: 0.0002 → 0.00015 (more aggressive densification)
#   - opacity_reset_interval: 3000 → 5000 (less frequent disruption)
#   - opacity_reset_value: 0.01 → 0.05 (faster opacity recovery!)
#   - Post-densification pruning starts at densify_until + opacity_reset_interval
#     (grace period: 35000 instead of 30000, giving 5000 iters recovery)
#   - prune_dead_threshold: 0.01 → 0.003 (less aggressive dead removal)
#   - percent_dense: 0.01 → 0.005 (allow more splits)
#   - position_lr_init: 1.6e-5 → 5e-5 (faster geometry convergence)
#   - Delay normal start: 2000 → 7000 (more geometry freedom early)
#   - Delay dist start: 3000 → 7000
#   - lambda_scale: 0.05 → 0.01 (less scale penalty, allow large Gaussians)
#
# ═══ Phase 2: Rendering Fixes ═══
#   - feature_lr: 0.0025 → 0.005 (2x faster SH convergence)
#   - f_rest_lr_divisor: 20 → 5 (4x faster higher-order SH!)
#   - lambda_dssim: 0.35 → 0.5 (more perceptual quality)
#   - opacity_lr: 0.05 → 0.1 (faster opacity optimization)
#   - 70k iterations (40% more convergence time)
#
# Expected: >20 dB target from fixing the pruning/opacity/feature bottlenecks

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=0 python -u feature_3dgs/train_2dgs_geometry.py \
  --source_dir dataset/OldHospital \
  --model_dir output/2dgs_models/OldHospital/v3_retrain11 \
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
  --lambda_dssim 0.5 \
  --lambda_dist 0.01 \
  --lambda_normal 0.05 \
  --normal_start_iter 7000 \
  --dist_start_iter 7000 \
  --lambda_depth 0.1 \
  --mono_depth_dir dataset/OldHospital/mono_depth \
  --depth_start_iter 3000 \
  --lambda_scale 0.01 \
  --scale_reg_threshold 1.0 \
  \
  --densify_from_iter 500 \
  --densify_until_iter 30000 \
  --densification_interval 100 \
  --densify_grad_threshold 0.00015 \
  --opacity_reset_interval 5000 \
  --opacity_reset_value 0.05 \
  --prune_dead_threshold 0.003 \
  --percent_dense 0.005 \
  \
  --save_iterations 10000 20000 30000 40000 50000 60000 70000 \
  --test_iterations 10000 20000 30000 40000 50000 60000 70000
