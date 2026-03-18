#!/bin/bash
# Auto-pipeline: wait for room_0 v8 geometry, then extract features, train AE, compress, embed

set -e
LOG_BASE="/root/ICLPose"
ROOM0_TRAIN_LOG="$LOG_BASE/output/2dgs_models/room_0/v8_fixed_poses/train.log"

echo "[$(date)] Waiting for room_0 v8 geometry training to complete..."

# Wait for final iteration marker in train log
PLY_PATH="$LOG_BASE/output/2dgs_models/room_0/v8_fixed_poses/point_cloud/iteration_30000/point_cloud.ply"
while true; do
    if [ -f "$PLY_PATH" ]; then
        echo "[$(date)] room_0 v8 point_cloud.ply found - training complete!"
        break
    fi
    if strings "$ROOM0_TRAIN_LOG" 2>/dev/null | grep -q "30000/30000"; then
        echo "[$(date)] room_0 v8 log shows iter 30000 done!"
        sleep 60  # wait a bit for ply to be written
        break
    fi
    echo "[$(date)] Still training room_0 v8 (waiting for $PLY_PATH)..."
    sleep 300  # check every 5 min
done

echo "[$(date)] Step 1: Extracting room_0 multiscale features (GPU 3)..."
CUDA_VISIBLE_DEVICES=3 /root/miniconda3/envs/iclpose/bin/python -u \
    scripts/extract_multiscale_features.py \
    --input_dir dataset/room_0/images/Sequence_1/rgb \
    --output_dir output/features_multiscale/room_0 \
    --device cuda \
    >> scripts/extract_room0.log 2>&1
echo "[$(date)] Feature extraction done."

echo "[$(date)] Step 2: Training room_0 AE and compressing (GPU 3)..."
CUDA_VISIBLE_DEVICES=3 /root/miniconda3/envs/iclpose/bin/python -u \
    scripts/train_multiscale_ae.py \
    --input_dir output/features_multiscale/room_0 \
    --output_dir output/features_multiscale_compressed/room_0 \
    --ae_save_dir output/ae_models/room_0 \
    --device cuda \
    --epochs 100 --batch_size 4096 --lr 1e-3 \
    >> scripts/ae_train_room0.log 2>&1
echo "[$(date)] AE training and compression done."

echo "[$(date)] Step 3: Training room_0 feature embedding (GPU 3)..."
mkdir -p output/feature_3dgs
CUDA_VISIBLE_DEVICES=3 /root/miniconda3/envs/iclpose/bin/python -u \
    -m feature_3dgs.train_multiscale_embedding_v2 \
    --ply_path output/2dgs_models/room_0/v8_fixed_poses/point_cloud/iteration_30000/point_cloud.ply \
    --feature_dir output/features_multiscale_compressed/room_0 \
    --traj_path dataset/room_0/Sequence_1/traj_w_c.txt \
    --output_dir output/feature_3dgs/room_0_v1 \
    --fx 600.0 --fy 600.0 --cx 319.5 --cy 239.5 \
    --img_height 480 --img_width 640 \
    --num_iters 20000 \
    --grad_accum 4 \
    --precache_gpu \
    --vis_interval 2000 \
    --vis_frames 0,100,200,500 \
    --log_interval 200 \
    --save_interval 5000 \
    >> output/feature_3dgs/room_0_v1_train.log 2>&1
echo "[$(date)] room_0 embedding training done!"

echo "[$(date)] Step 4: Splitting per-scale checkpoints..."
/root/miniconda3/envs/iclpose/bin/python scripts/split_multiscale_embedding.py \
    --ckpt output/feature_3dgs/room_0_v1/best_model.pth \
    --output_dir output/feature_3dgs/room_0_v1/per_scale \
    >> scripts/split_room0.log 2>&1
echo "[$(date)] All steps complete for room_0!"
