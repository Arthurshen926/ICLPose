#!/bin/bash

# 快速重启训练脚本 - 修复了初始位姿噪声过大的问题

echo "=========================================="
echo "重启训练 - EXP011"
echo "=========================================="
echo ""
echo "已修复的问题:"
echo "  ✓ 初始位姿噪声: 10°/1.0m → 3°/0.05m"
echo "  ✓ Batch size: 144 → 32"
echo "  ✓ 学习率: 5e-5 → 1e-4"
echo "  ✓ 模型简化: queries 128→64, layers 8→4"
echo "  ✓ 验证频率: 每5个epoch → 每2个epoch"
echo ""
echo "预期效果:"
echo "  - 前10个epoch loss应该降到2-5"
echo "  - 验证loss应该接近训练loss"
echo "  - 角度误差<2°, 平移误差<0.03m"
echo ""
echo "=========================================="
echo ""

# 确认
read -p "开始训练? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

# 切换到正确目录
cd "$(dirname "$0")"

# 检查是否在虚拟环境中
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "警告: 未检测到conda环境"
    echo "请先激活环境: conda activate geo-aware"
    exit 1
fi

# 启动训练
echo "启动训练..."
python train.py --config configs/train_config.yaml

echo ""
echo "训练完成!"
echo "查看结果: tensorboard --logdir output/exp011/logs --port 6006"
