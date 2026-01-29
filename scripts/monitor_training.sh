#!/bin/bash
# 监控训练进度脚本

LOG_FILE="/home/yons/Projects/SplatLoc/implicit_correspondence/training_fixed.log"

echo "========================================="
echo "训练监控 - $(date)"
echo "========================================="

# 检查进程
PID=$(ps aux | grep "train.py" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$PID" ]; then
    echo "✓ 训练进程运行中 (PID: $PID)"
else
    echo "✗ 训练进程未运行"
    exit 1
fi

echo ""
echo "最近的训练进度:"
echo "-----------------------------------------"
tail -100 "$LOG_FILE" | grep -E "Epoch [0-9]+/100:|训练.*Loss:|验证.*Loss:|最佳验证损失"
echo ""
echo "最新日志 (最后10行):"
echo "-----------------------------------------"
tail -10 "$LOG_FILE"
