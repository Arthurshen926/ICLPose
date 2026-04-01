#!/bin/bash
# Train RADIO-GS V11 feature fields for multiple Replica scenes
# Usage: bash radio_gs/scripts/train_all_scenes.sh [--scenes room_1,room_2] [--gpus 2,3]
#
# Prerequisites for each scene:
#   1. RADIO features extracted: output/radio_features_1280d/{scene}/Sequence_{1,2}/backbone/
#   2. 2DGS model trained: output/2dgs_models/{scene}/v8_fixed_poses/point_cloud/iteration_30000/point_cloud.ply
#   3. Config: radio_gs/configs/replica_explicit_v11_{scene}.yaml

set -e

SCENES="room_1,room_2"
GPUS="2,3"

while [[ $# -gt 0 ]]; do
    case $1 in
        --scenes) SCENES="$2"; shift 2;;
        --gpus) GPUS="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

IFS=',' read -ra SCENE_LIST <<< "$SCENES"
IFS=',' read -ra GPU_LIST <<< "$GPUS"

if [ ${#SCENE_LIST[@]} -gt ${#GPU_LIST[@]} ]; then
    echo "ERROR: Need at least as many GPUs as scenes"
    exit 1
fi

PIDS=()
for i in "${!SCENE_LIST[@]}"; do
    scene="${SCENE_LIST[$i]}"
    gpu="${GPU_LIST[$i]}"
    scene_nounderscore="${scene//_/}"
    CONFIG="radio_gs/configs/replica_explicit_v11_${scene}.yaml"
    PLY="output/2dgs_models/${scene}/v8_fixed_poses/point_cloud/iteration_30000/point_cloud.ply"

    echo "============================================================"
    echo "  Scene: $scene | GPU: $gpu"
    echo "  Config: $CONFIG"
    echo "  PLY: $PLY"
    echo "============================================================"

    if [ ! -f "$PLY" ]; then
        echo "  ERROR: PLY not found, skipping. Train 2DGS first."
        continue
    fi

    if [ ! -f "$CONFIG" ]; then
        echo "  ERROR: Config not found, skipping."
        continue
    fi

    CUDA_VISIBLE_DEVICES=$gpu python radio_gs/scripts/train_feature_field.py \
        --config "$CONFIG" \
        2>&1 | tee "output/v11_${scene}_train.log" &
    PIDS+=($!)
    echo "  Started PID: ${PIDS[-1]}"
done

echo ""
echo "Waiting for all training jobs to complete..."
echo "PIDs: ${PIDS[*]}"

for pid in "${PIDS[@]}"; do
    wait "$pid"
    echo "  PID $pid completed (exit code: $?)"
done

echo ""
echo "All training jobs complete!"
