#!/bin/bash
# Train all 4 GS feature scales with diversity regularization for OldHospital
set -e
cd /root/ICLPose

PLY=output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply
FEAT_DIR=output/features_multiscale_compressed/OldHospital_indexed
TRAJ=output/features_multiscale_compressed/OldHospital_indexed/traj_w_c.txt
OUT_BASE=output/feature_3dgs/oldhospital_ae_div_perscale
ITERS=15000
DIV_W=1.0

for SCALE in coarse mid fine_sd fine_dino; do
    echo "================================================================"
    echo "Training scale: $SCALE (diversity_weight=$DIV_W)"
    echo "================================================================"
    
    # Skip fine_sd if already trained
    if [ -f "${OUT_BASE}/$SCALE/best_model.pth" ]; then
        echo "  $SCALE already trained, skipping"
        continue
    fi
    
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m feature_3dgs.train_da3_perscale \
        --scale $SCALE \
        --ply_path $PLY \
        --feature_dir $FEAT_DIR \
        --traj_path $TRAJ \
        --output_dir ${OUT_BASE}/$SCALE \
        --img_height 1080 --img_width 1920 \
        --fx 1673.5 --fy 1673.5 --cx 960.0 --cy 540.0 \
        --num_iters $ITERS --grad_accum 4 --precache_gpu \
        --diversity_weight $DIV_W \
        --log_interval 1000
    echo "Done: $SCALE"
    echo ""
done

echo "All scales trained!"
