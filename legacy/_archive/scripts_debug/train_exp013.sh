#!/bin/bash
# 启动新的训练 - exp013（完全对齐ICL-I2PReg架构）

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 实验名称
EXP_NAME="exp013"

# 配置文件
CONFIG="configs/train_config.yaml"

echo "========================================="
echo "🚀 启动训练: $EXP_NAME"
echo "========================================="
echo "架构变更:"
echo "  ✓ Fusion Module完全对齐ICL-I2PReg"
echo "  ✓ 使用分离的query特征（img/pcd）"
echo "  ✓ 使用处理后的tokens计算heatmap"
echo "  ✓ 固定2层架构（1 img block + 1 pcd block）"
echo "========================================="

# 从头开始训练（因为模型架构变了）
python train.py \
    --config "$CONFIG" \
    --gpus 0 \
    --num_gpus 1 \
    2>&1 | tee "output/${EXP_NAME}/train.log"

echo ""
echo "✅ 训练完成！"
echo "📊 查看训练日志: output/${EXP_NAME}/train.log"
echo "📈 启动TensorBoard: tensorboard --logdir output/${EXP_NAME}/logs --port 6007"
