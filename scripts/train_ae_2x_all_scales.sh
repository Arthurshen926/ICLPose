#!/bin/bash
# Train all 4 GS feature scales with 2× upsampled AE targets for OldHospital
set -e
cd /root/ICLPose

PLY=output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply
FEAT_DIR=output/features_multiscale_compressed_2x/OldHospital_indexed
TRAJ=output/features_multiscale_compressed/OldHospital_indexed/traj_w_c.txt
OUT_BASE=output/feature_3dgs/oldhospital_ae_2x_perscale
ITERS=15000

for SCALE in coarse mid fine_sd fine_dino; do
    echo "================================================================"
    echo "Training scale: $SCALE"
    echo "================================================================"
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m feature_3dgs.train_da3_perscale \
        --scale $SCALE \
        --ply_path $PLY \
        --feature_dir $FEAT_DIR \
        --traj_path $TRAJ \
        --output_dir ${OUT_BASE}/$SCALE \
        --img_height 1080 --img_width 1920 \
        --fx 1673.5 --fy 1673.5 --cx 960.0 --cy 540.0 \
        --num_iters $ITERS --grad_accum 4 --precache_gpu \
        --log_interval 1000
    echo "Done: $SCALE"
    echo ""
done

echo "All scales trained!"
