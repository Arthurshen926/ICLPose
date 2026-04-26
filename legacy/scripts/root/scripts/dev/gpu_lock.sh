#!/bin/bash
# =============================================================================
# GPU 显存锁定/释放工具
# =============================================================================
# 用法:
#   bash scripts/gpu_lock.sh on 3 4       # 锁定 GPU 3 和 4
#   bash scripts/gpu_lock.sh on 3 4 --gb 20  # 每卡占 20GB
#   bash scripts/gpu_lock.sh off            # 释放所有
#   bash scripts/gpu_lock.sh status         # 查看状态
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$PROJECT_DIR/output/logs/hold_gpu.pid"
LOG_FILE="$PROJECT_DIR/output/logs/hold_gpu.log"
PYTHON="/root/miniconda3/envs/iclpose/bin/python"

mkdir -p "$(dirname "$PID_FILE")"

ACTION="${1:-status}"
shift 2>/dev/null || true

case "$ACTION" in
    on|start|lock)
        # 先检查是否已有锁
        if [ -f "$PID_FILE" ]; then
            OLD_PID=$(head -1 "$PID_FILE")
            if kill -0 "$OLD_PID" 2>/dev/null; then
                OLD_GPUS=$(sed -n '2p' "$PID_FILE")
                echo "⚠ 已有 GPU 锁运行中 (PID=$OLD_PID, GPUs=$OLD_GPUS)"
                echo "  先释放: bash scripts/gpu_lock.sh off"
                exit 1
            else
                rm -f "$PID_FILE"
            fi
        fi

        # 默认锁 GPU 3 4
        GPUS="${*:-3 4}"
        echo "锁定 GPU: $GPUS"
        nohup "$PYTHON" "$SCRIPT_DIR/hold_gpu.py" $GPUS > "$LOG_FILE" 2>&1 &
        NEW_PID=$!
        sleep 2

        if kill -0 "$NEW_PID" 2>/dev/null; then
            echo "✓ 已锁定 (PID=$NEW_PID)"
            cat "$LOG_FILE"
            echo ""
            echo "释放命令: bash scripts/gpu_lock.sh off"
        else
            echo "✗ 启动失败, 日志:"
            cat "$LOG_FILE"
            exit 1
        fi
        ;;

    off|stop|unlock|release)
        if [ ! -f "$PID_FILE" ]; then
            echo "无活跃的 GPU 锁"
            exit 0
        fi
        PID=$(head -1 "$PID_FILE")
        GPUS=$(sed -n '2p' "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "释放 GPU 锁 (PID=$PID, GPUs=$GPUS)..."
            kill "$PID"
            sleep 1
            if kill -0 "$PID" 2>/dev/null; then
                kill -9 "$PID"
            fi
            echo "✓ 已释放"
        else
            echo "进程 $PID 已不存在"
        fi
        rm -f "$PID_FILE"
        ;;

    status|info)
        if [ -f "$PID_FILE" ]; then
            PID=$(head -1 "$PID_FILE")
            GPUS=$(sed -n '2p' "$PID_FILE")
            if kill -0 "$PID" 2>/dev/null; then
                echo "✓ GPU 锁活跃 (PID=$PID, GPUs=$GPUS)"
            else
                echo "✗ PID=$PID 已不存在 (可能已崩溃)"
                rm -f "$PID_FILE"
            fi
        else
            echo "无活跃的 GPU 锁"
        fi
        # 显示 GPU 使用情况
        echo ""
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null
        ;;

    *)
        echo "用法: bash scripts/gpu_lock.sh {on|off|status} [GPU_IDs...] [--gb N]"
        echo ""
        echo "  on  3 4        锁定 GPU 3 和 4 (默认每卡 22GB)"
        echo "  on  3 4 --gb 20       每卡占 20GB"
        echo "  on  3 4 --util 90     目标利用率 90%"
        echo "  on  3 4 --gb 22 --util 80"
        echo "  off            释放"
        echo "  status         查看状态"
        ;;
esac
