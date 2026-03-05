#!/bin/bash
# Retrain36: Spatial Appearance + freeze@5k
#
# ═══ Key Insight ═══
# Previous analysis revealed the core problem: different sequences captured
# at different times create spatially non-uniform appearance changes (shadows,
# specular, vegetation). The global affine appearance model (AppearanceNetwork)
# can only apply uniform scale/bias per channel, which fails for local effects.
#
# This experiment uses SpatialAppearanceNetwork — a lightweight CNN decoder
# that generates per-pixel scale/bias maps from per-image embeddings.
# The decoder runs at 128x128 internally and bilinearly upsamples to full res.
#
# Expected improvements:
# - Better handling of shadow boundaries (spatially varying)
# - Better handling of vegetation/foliage at different positions
# - Training views should reach higher PSNR (reduced conflicting signals)
# - Test views should benefit from cleaner Gaussian geometry
#
# Freeze at 5k as per retrain31 (best baseline config)

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=GPU_ID nohup python -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir output/2dgs_models/OldHospital/v3_retrain36 \
    --iterations 70000 \
    --batch_size 4 \
    --longest_edge 0 \
    \
    --random_background \
    --use_appearance \
    --spatial_appearance \
    --use_mask \
    --appearance_freeze_iter 5000 \
    --appearance_scale_range 2.0 \
    --appearance_bias_range 0.5 \
    --grad_clip_max_norm 1.0 \
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
    --save_iterations 3000 5000 7000 10000 12000 15000 20000 25000 30000 40000 50000 60000 70000 \
    --test_iterations 3000 5000 7000 10000 12000 15000 20000 25000 30000 40000 50000 60000 70000 \
    > output/retrain36.log 2>&1 &

echo "PID: $!"
echo "Log: output/retrain36.log"
