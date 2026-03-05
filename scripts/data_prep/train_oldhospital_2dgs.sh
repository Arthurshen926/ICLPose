#!/bin/bash
# ============================================================
# OldHospital 2DGS 重建训练 (在 stairs 训练完成后自动启动)
# GPU 0 only — 不影响 GPU 1 上的 exp012_upsampler
# ============================================================
set -e

STDLOC_DIR="/home/yons/Projects/STDLoc"
DATA_DIR="/home/yons/Projects/ICLPose/dataset/OldHospital"
LOG_FILE="/home/yons/Projects/ICLPose/logs/oldhospital_2dgs_train.log"

echo "[$(date)] Waiting for stairs 2DGS training to finish on GPU 0..."
echo "  Checking every 60 seconds..."

# 等待 stairs 训练完成 (检测 GPU 0 显存是否释放到 < 1GB)
while true; do
    GPU0_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
    if [ "$GPU0_MEM" -lt 1000 ]; then
        echo "[$(date)] GPU 0 is free (${GPU0_MEM} MiB). Starting OldHospital training..."
        break
    fi
    echo "[$(date)] GPU 0 still busy (${GPU0_MEM} MiB). Waiting..."
    sleep 60
done

# 启动 OldHospital 2DGS 训练
cd "$STDLOC_DIR"
CUDA_VISIBLE_DEVICES=0 python train.py \
  -s "$DATA_DIR" \
  -m "$OUTPUT_DIR/2dgs_models/OldHospital/v1" \
  --iterations 30000 \
  --data_device cpu \
  -f sp \
  -g 2dgs \
  --images "processed" \
  -r 1 \
  --densify_grad_threshold 0.0004 \
  --position_lr_init 0.000016 \
  --scaling_lr 0.001 \
  2>&1 | tee "$LOG_FILE"

echo "[$(date)] OldHospital 2DGS training completed!"
