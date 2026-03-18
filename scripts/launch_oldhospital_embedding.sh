#!/bin/bash
# Wait for OldHospital AE compression to complete, then launch embedding training

COMPRESSED_DIR="/root/ICLPose/output/features_multiscale_compressed/OldHospital"
LOG_FILE="/root/ICLPose/output/feature_3dgs/oldhospital_v1_train.log"
SCALES=("fine_sd" "fine_dino" "mid" "coarse")
EXPECTED=1084

echo "[$(date)] Waiting for OldHospital AE compression to finish..."

while true; do
    all_done=1
    for scale in "${SCALES[@]}"; do
        n=$(ls "$COMPRESSED_DIR/$scale/" 2>/dev/null | wc -l)
        if [ "$n" -lt "$EXPECTED" ]; then
            all_done=0
            echo "[$(date)] $scale: $n/$EXPECTED files compressed..."
            break
        fi
    done
    
    if [ "$all_done" -eq 1 ]; then
        echo "[$(date)] All 4 scales compressed! Launching OldHospital embedding training..."
        break
    fi
    sleep 60
done

mkdir -p /root/ICLPose/output/feature_3dgs
cd /root/ICLPose

CUDA_VISIBLE_DEVICES=4 /root/miniconda3/envs/iclpose/bin/python -u \
    -m feature_3dgs.train_multiscale_embedding_v2 \
    --ply_path output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply \
    --feature_dir output/features_multiscale_compressed/OldHospital \
    --colmap_dir dataset/OldHospital/sparse/0 \
    --output_dir output/feature_3dgs/oldhospital_v1 \
    --num_iters 20000 \
    --grad_accum 4 \
    --precache_gpu \
    --vis_interval 2000 \
    --vis_frames 0,100,300,500 \
    --log_interval 200 \
    --save_interval 5000 \
    >> "$LOG_FILE" 2>&1

echo "[$(date)] OldHospital embedding training done."
