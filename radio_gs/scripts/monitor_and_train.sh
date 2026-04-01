#!/bin/bash
# Monitor 2DGS training completion and auto-start V11 feature field training
# Usage: bash radio_gs/scripts/monitor_and_train.sh [--eval-after]
#
# This script:
# 1. Polls for 2DGS PLY files for room_1 and room_2
# 2. As soon as a PLY appears, launches V11 training on the corresponding GPU
# 3. Optionally runs evaluation after training completes

set -e

EVAL_AFTER=false
if [[ "$1" == "--eval-after" ]]; then
    EVAL_AFTER=true
fi

PLY_ROOM1="output/2dgs_models/room_1/v8_fixed_poses/point_cloud/iteration_30000/point_cloud.ply"
PLY_ROOM2="output/2dgs_models/room_2/v8_fixed_poses/point_cloud/iteration_30000/point_cloud.ply"

CONFIG_ROOM1="radio_gs/configs/replica_explicit_v11_room_1.yaml"
CONFIG_ROOM2="radio_gs/configs/replica_explicit_v11_room_2.yaml"

GPU_ROOM1=3  # Same GPU as 2DGS training (freed after completion)
GPU_ROOM2=2  # Same GPU as 2DGS training (freed after completion)

ROOM1_STARTED=false
ROOM2_STARTED=false
ROOM1_PID=""
ROOM2_PID=""

echo "=========================================="
echo "  RADIO-GS Auto-Training Monitor"
echo "=========================================="
echo "Monitoring for 2DGS completion..."
echo "  room_1 PLY: $PLY_ROOM1"
echo "  room_2 PLY: $PLY_ROOM2"
echo ""

while true; do
    # Check room_1
    if [ "$ROOM1_STARTED" = false ] && [ -f "$PLY_ROOM1" ]; then
        echo "[$(date)] ✅ room_1 2DGS complete! PLY found."
        echo "  Starting V11 training on GPU $GPU_ROOM1..."
        
        CUDA_VISIBLE_DEVICES=$GPU_ROOM1 python radio_gs/scripts/train_feature_field.py \
            --config "$CONFIG_ROOM1" \
            2>&1 | tee "output/v11_room_1_train.log" &
        ROOM1_PID=$!
        ROOM1_STARTED=true
        echo "  room_1 V11 training PID: $ROOM1_PID"
    fi
    
    # Check room_2
    if [ "$ROOM2_STARTED" = false ] && [ -f "$PLY_ROOM2" ]; then
        echo "[$(date)] ✅ room_2 2DGS complete! PLY found."
        echo "  Starting V11 training on GPU $GPU_ROOM2..."
        
        CUDA_VISIBLE_DEVICES=$GPU_ROOM2 python radio_gs/scripts/train_feature_field.py \
            --config "$CONFIG_ROOM2" \
            2>&1 | tee "output/v11_room_2_train.log" &
        ROOM2_PID=$!
        ROOM2_STARTED=true
        echo "  room_2 V11 training PID: $ROOM2_PID"
    fi
    
    # Both started? Wait for completion
    if [ "$ROOM1_STARTED" = true ] && [ "$ROOM2_STARTED" = true ]; then
        echo ""
        echo "[$(date)] Both V11 trainings launched. Waiting for completion..."
        
        FAIL=0
        if [ -n "$ROOM1_PID" ]; then
            wait $ROOM1_PID || { echo "❌ room_1 V11 training failed!"; FAIL=1; }
            echo "[$(date)] room_1 V11 training finished."
        fi
        if [ -n "$ROOM2_PID" ]; then
            wait $ROOM2_PID || { echo "❌ room_2 V11 training failed!"; FAIL=1; }
            echo "[$(date)] room_2 V11 training finished."
        fi
        
        if [ $FAIL -eq 0 ] && [ "$EVAL_AFTER" = true ]; then
            echo ""
            echo "[$(date)] Starting evaluation for all scenes..."
            bash radio_gs/scripts/eval_all_scenes.sh --scenes room_0,room_1,room_2 --gpu 5
        fi
        
        echo ""
        echo "=========================================="
        echo "  All V11 training complete!"
        echo "=========================================="
        exit 0
    fi
    
    # Status update every 5 minutes
    echo -n "."
    sleep 300
done
