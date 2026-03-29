#!/bin/bash
# GSFFs training on OldHospital - GPU 3
export CUDA_VISIBLE_DEVICES=3
export OPENBLAS_NUM_THREADS=8

cd /root/ICLPose

python scripts/train_gsff.py \
    --source_dir dataset/OldHospital \
    --model_path output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply \
    --cameras_json output/2dgs_models/OldHospital/v7_depth/cameras.json \
    --output_dir output/gsff/OldHospital \
    --render_height 540 --render_width 960 \
    --total_iters 50000 \
    --phase1_iters 5000 \
    --lr_triplane 1e-3 --lr_encoder 1e-4 \
    --temperature 0.07 \
    --n_clusters 34 \
    --coarse_resolution 256 --fine_resolution 1024 \
    --feature_dim 16 \
    2>&1 | tee output/gsff/OldHospital/train_stdout.log
