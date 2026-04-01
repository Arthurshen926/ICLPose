#!/bin/bash
# Run baseline and ablation experiments sequentially on a single GPU
# Usage: bash radio_gs/scripts/run_baselines.sh [--gpu 5]
set -e

GPU=${2:-5}
SCENE="room_0"

echo "=========================================="
echo "  RADIO-GS Baseline & Ablation Runner"
echo "  GPU: $GPU, Scene: $SCENE"
echo "=========================================="

# --- 1. Wait for F3DGS baseline training to finish ---
BASELINE_PID=$(ps aux | grep "baseline_f3dgs.yaml" | grep python | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$BASELINE_PID" ]; then
    echo ""
    echo "[$(date)] Waiting for F3DGS baseline training (PID $BASELINE_PID) to complete..."
    while kill -0 "$BASELINE_PID" 2>/dev/null; do
        sleep 30
    done
    echo "[$(date)] F3DGS baseline training completed!"
fi

# --- 2. Evaluate F3DGS baseline ---
echo ""
echo "=========================================="
echo "  Evaluating F3DGS Baseline"
echo "=========================================="
BASELINE_CKPT="output/radio_gs/room0_baseline_f3dgs/checkpoints/best.pth"
if [ -f "$BASELINE_CKPT" ]; then
    echo "[$(date)] Running depth+seg evaluation..."
    CUDA_VISIBLE_DEVICES=$GPU python radio_gs/scripts/eval_rendered.py \
        --config radio_gs/configs/baseline_f3dgs.yaml \
        --checkpoint "$BASELINE_CKPT" \
        2>&1 | tee output/eval_baseline_f3dgs.log

    echo "[$(date)] Running grounding evaluation..."
    CUDA_VISIBLE_DEVICES=$GPU python radio_gs/scripts/eval_grounding.py \
        --config radio_gs/configs/baseline_f3dgs.yaml \
        --checkpoint "$BASELINE_CKPT" \
        --gt_features "output/radio_features_1280d/${SCENE}/Sequence_2/backbone" \
        --semantic_dir "dataset/${SCENE}/Sequence_2/semantic_class" \
        --rgb_dir "dataset/${SCENE}/Sequence_2/rgb" \
        --pose_file "dataset/${SCENE}/Sequence_2/traj_w_c.txt" \
        2>&1 | tee output/eval_baseline_f3dgs_grounding.log
else
    echo "WARNING: Baseline checkpoint not found at $BASELINE_CKPT"
fi

# --- 3. Train ablation (no refiner, with FeatSharp) ---
echo ""
echo "=========================================="
echo "  Training Ablation: No Refiner"
echo "=========================================="
echo "[$(date)] Starting ablation training..."
CUDA_VISIBLE_DEVICES=$GPU python radio_gs/scripts/train_feature_field.py \
    --config radio_gs/configs/ablation_no_refiner.yaml \
    2>&1 | tee output/ablation_no_refiner_train.log

# --- 4. Evaluate ablation ---
echo ""
echo "=========================================="
echo "  Evaluating Ablation: No Refiner"
echo "=========================================="
ABLATION_CKPT="output/radio_gs/room0_ablation_no_refiner/checkpoints/best.pth"
if [ -f "$ABLATION_CKPT" ]; then
    echo "[$(date)] Running depth+seg evaluation..."
    CUDA_VISIBLE_DEVICES=$GPU python radio_gs/scripts/eval_rendered.py \
        --config radio_gs/configs/ablation_no_refiner.yaml \
        --checkpoint "$ABLATION_CKPT" \
        2>&1 | tee output/eval_ablation_no_refiner.log

    echo "[$(date)] Running grounding evaluation..."
    CUDA_VISIBLE_DEVICES=$GPU python radio_gs/scripts/eval_grounding.py \
        --config radio_gs/configs/ablation_no_refiner.yaml \
        --checkpoint "$ABLATION_CKPT" \
        --gt_features "output/radio_features_1280d/${SCENE}/Sequence_2/backbone" \
        --semantic_dir "dataset/${SCENE}/Sequence_2/semantic_class" \
        --rgb_dir "dataset/${SCENE}/Sequence_2/rgb" \
        --pose_file "dataset/${SCENE}/Sequence_2/traj_w_c.txt" \
        2>&1 | tee output/eval_ablation_no_refiner_grounding.log
else
    echo "WARNING: Ablation checkpoint not found at $ABLATION_CKPT"
fi

echo ""
echo "=========================================="
echo "  All baselines & ablations complete!"
echo "  Results in output/eval_baseline_*.log"
echo "  and output/eval_ablation_*.log"
echo "=========================================="
