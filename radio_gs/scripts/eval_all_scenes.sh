#!/bin/bash
# Evaluate RADIO-GS V11 across multiple Replica scenes
# Usage: bash radio_gs/scripts/eval_all_scenes.sh [--scenes room_0,room_1,room_2] [--gpu 0]
#
# Runs: eval_rendered.py (depth + segmentation) and eval_grounding.py (text grounding)
# for each scene. Results saved to output/eval_v11_{scene}.log etc.

set -e

SCENES="room_0"
GPU=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --scenes) SCENES="$2"; shift 2;;
        --gpu) GPU="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

IFS=',' read -ra SCENE_LIST <<< "$SCENES"

for scene in "${SCENE_LIST[@]}"; do
    echo "============================================================"
    echo "  Evaluating scene: $scene"
    echo "============================================================"

    scene_nounderscore="${scene//_/}"
    CONFIG="radio_gs/configs/replica_explicit_v11_${scene}.yaml"
    CHECKPOINT="output/radio_gs/${scene_nounderscore}_explicit_v11/checkpoints/best.pth"

    # Handle room_0 special case (original config name)
    if [ "$scene" = "room_0" ]; then
        CONFIG="radio_gs/configs/replica_explicit_v11.yaml"
        CHECKPOINT="output/radio_gs/room0_explicit_v11/checkpoints/best.pth"
    fi

    if [ ! -f "$CHECKPOINT" ]; then
        echo "  WARNING: checkpoint not found: $CHECKPOINT, skipping"
        continue
    fi

    GT_FEAT_DIR="output/radio_features_1280d/${scene}/Sequence_2/backbone"
    SEM_DIR="dataset/${scene}/Sequence_2/semantic_class"
    RGB_DIR="dataset/${scene}/Sequence_2/rgb"
    POSE_FILE="dataset/${scene}/Sequence_2/traj_w_c.txt"

    # 1. Depth + Segmentation eval
    echo "  [1/2] Running depth + segmentation eval..."
    CUDA_VISIBLE_DEVICES=$GPU python radio_gs/scripts/eval_rendered.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        2>&1 | tee "output/eval_v11_${scene}.log"

    # 2. Text grounding eval
    echo "  [2/2] Running text grounding eval..."
    CUDA_VISIBLE_DEVICES=$GPU python radio_gs/scripts/eval_grounding.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --gt_features "$GT_FEAT_DIR" \
        --semantic_dir "$SEM_DIR" \
        --rgb_dir "$RGB_DIR" \
        --pose_file "$POSE_FILE" \
        --vis_dir "output/vis_grounding_v11_${scene}" \
        2>&1 | tee "output/eval_v11_${scene}_grounding.log"

    echo "  Done: $scene"
    echo ""
done

echo "============================================================"
echo "  All evaluations complete!"
echo "============================================================"

# Print summary across scenes
echo ""
echo "=== Summary ==="
for scene in "${SCENE_LIST[@]}"; do
    echo ""
    echo "--- $scene ---"
    if [ -f "output/eval_v11_${scene}.log" ]; then
        grep -A5 "^Mode" "output/eval_v11_${scene}.log" | head -7
    fi
    if [ -f "output/eval_v11_${scene}_grounding.log" ]; then
        grep "^Mean" "output/eval_v11_${scene}_grounding.log" | tail -1
    fi
done
