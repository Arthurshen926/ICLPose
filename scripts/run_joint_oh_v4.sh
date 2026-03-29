#!/bin/bash
# 启动两组 OldHospital 联合重建实验 (DA3 fine + DINO fine)
# GPU 2: DA3 fine (64d @ 69×121)
# GPU 4: DINO fine (64d @ 35×61)
#
# 用法:
#   bash scripts/run_joint_oh_v4.sh

set -e
cd /root/ICLPose

mkdir -p output/logs

echo "========================================================"
echo "  OldHospital v4 Joint Reconstruction (两组对比实验)"
echo "========================================================"
echo ""

# ── GPU 2: DA3 fine ─────────────────────────────────────────
echo "[1/2] 启动 DA3 fine joint 训练 (GPU 2)..."
CUDA_VISIBLE_DEVICES=2 nohup python -u -m feature_3dgs.train_2dgs_joint_v3 \
    --config configs/joint_oh_v4_da3.yaml \
    > output/logs/joint_oh_v4_da3.log 2>&1 &
DA3_PID=$!
echo "  PID: $DA3_PID  |  Log: output/logs/joint_oh_v4_da3.log"

# 等待 DA3 进程完成初始化(深度+图像预缓存)再启动 DINO，避免双进程 NFS 竞争
echo "  等待 DA3 完成初始化 (最长 30 分钟)..."
WAIT_SECS=0
while [ $WAIT_SECS -lt 1800 ]; do
    sleep 30
    WAIT_SECS=$((WAIT_SECS + 30))
    if grep -q "Iter " output/logs/joint_oh_v4_da3.log 2>/dev/null; then
        echo "  DA3 已开始训练! (等待了 ${WAIT_SECS}s)"
        break
    fi
    echo "    ...仍在初始化 (${WAIT_SECS}s)..."
done

# ── GPU 4: DINO fine ────────────────────────────────────────
echo "[2/2] 启动 DINO fine joint 训练 (GPU 4)..."
CUDA_VISIBLE_DEVICES=4 nohup python -u -m feature_3dgs.train_2dgs_joint_v3 \
    --config configs/joint_oh_v4_dino.yaml \
    > output/logs/joint_oh_v4_dino.log 2>&1 &
DINO_PID=$!
echo "  PID: $DINO_PID  |  Log: output/logs/joint_oh_v4_dino.log"

echo ""
echo "========================================================"
echo "  两组训练已启动!"
echo "  监控命令:"
echo "    tail -f output/logs/joint_oh_v4_da3.log"
echo "    tail -f output/logs/joint_oh_v4_dino.log"
echo "  可视化输出:"
echo "    output/2dgs_joint/joint_oh_v4_da3/vis_iter*.png"
echo "    output/2dgs_joint/joint_oh_v4_dino/vis_iter*.png"
echo "========================================================"
