#!/bin/bash
# Retrain12: Key improvements over Retrain11 (16.23 dB @ 10k)
#
# ═══ Analysis of Retrain11 issues ═══
#   1. NO MASK: Dynamic objects (cars, pedestrians) not excluded from training
#      → Gaussians forced to fit inconsistent appearances → blur + artifacts
#      → masks.pkl exists (6.3GB, 1077 frames, 3-channel: obj/sky/distort)
#   2. DEPTH on dynamic objects: DPT depth unreliable on cars/trees/transparent
#      → Pearson depth loss forces Gaussians to wrong positions
#      → FIX: Now depth loss uses mask to exclude dynamic regions (code fix applied)
#   3. Normal regularization too early: lambda_normal=0.05 from iter 7000
#      → 98 extreme loss warnings all after iter 7000 (normal_start_iter)
#      → Trees/vegetation have no clear surface normals
#      → FIX: delay to 15000, reduce weight to 0.02
#   4. Depth supervision weight: lambda_depth=0.1 may be too strong
#      → DPT unreliable on tree canopy, repeated textures, far objects
#      → FIX: reduce to 0.05
#
# ═══ Changes from Retrain11 ═══
#   + --use_mask                    (NEW: exclude dynamic objects)
#   + --mask_path dataset/OldHospital/masks.pkl (explicit path)
#   Δ lambda_normal: 0.05 → 0.02   (gentler on vegetation)
#   Δ normal_start_iter: 7000 → 15000 (let Gaussians fit appearance first)
#   Δ lambda_depth: 0.1 → 0.05     (less aggressive depth constraint)
#   Δ depth_start_iter: 3000 → 5000 (delay depth supervision slightly)
#   Δ dist_start_iter: 7000 → 12000 (delay distortion loss to avoid early instability)
#   = Everything else same as retrain11 (which had good densification settings)
#
# Expected: 18+ dB (mask alone should give 2-3 dB lift)

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=1 python -u feature_3dgs/train_2dgs_geometry.py \
  --source_dir dataset/OldHospital \
  --model_dir output/2dgs_models/OldHospital/v3_retrain12 \
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
