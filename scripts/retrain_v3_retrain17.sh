#!/bin/bash
# Retrain17: Random background + Appearance embedding + Masks (gradient-corrected)
#
# ═══ Analysis of 16.77 dB ceiling (retrain16 @15k) ═══
# Per-view PSNR: min=11.97, max=22.52, spread=10.55 dB
# Distribution: <14 dB=21 views, 14-16=57, 16-18=41, 18-20=46, 20+=17
# seq4 mean=15.65 (56 views), seq8 mean=17.26 (126 views)
# Root cause: NOT exposure variation (brightness diff <0.13), NOT Gaussian count
# → Scene content complexity (trees, vegetation) + dynamic objects + lacking opacity
#
# ═══ Three key improvements ═══
# 1. Random background: Forces Gaussians to be opaque → cleaner sky/vegetation
#    Without random bg, semi-transparent Gaussians rely on black bg → wrong colors
# 2. Appearance embedding: Per-image affine color transform (scale+bias) via 
#    learned embeddings. Absorbs inter-image color/exposure variation so Gaussians
#    learn clean canonical appearance. Identity-initialized (no initial impact).
# 3. Masks + gradient-corrected loss: Exclude dynamic objects from loss with
#    proper L1 averaging (only over valid pixels, not diluted by masked zeros).
#    Combined with random bg: masked GT pixels = random bg color for consistency.
#
# ═══ Baseline comparison ═══
# retrain16 (no mask, no random bg, no appearance): PSNR=16.77 @15k (peak)
# retrain17 target: break 17+ dB

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=1 python -u feature_3dgs/train_2dgs_geometry.py \
  --source_dir dataset/OldHospital \
  --model_dir output/2dgs_models/OldHospital/v3_retrain17 \
  --iterations 70000 \
  --batch_size 4 \
  --longest_edge 0 \
  \
  --random_background \
  --use_appearance \
  --use_mask \
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
  --opacity_reset_interval 999999 \
  --opacity_reset_value 0.01 \
  --prune_dead_threshold 0.005 \
  --percent_dense 0.005 \
  \
  --save_iterations 3000 5000 7000 10000 15000 20000 30000 40000 50000 60000 70000 \
  --test_iterations 3000 5000 7000 10000 15000 20000 30000 40000 50000 60000 70000
