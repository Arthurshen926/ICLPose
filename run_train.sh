#!/bin/bash
# 隐式对应关系位姿估计训练启动脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 配置文件路径
CONFIG_FILE="configs/train_config.yaml"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 训练命令
echo "======================================"
echo "隐式对应关系位姿估计网络训练"
echo "======================================"
echo "配置文件: $CONFIG_FILE"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "======================================"
echo ""

# 启动训练
python train.py \
    --config "$CONFIG_FILE" \
    "$@"

echo ""
echo "训练完成!"
