#!/bin/bash
# 训练脚本使用示例

# ============================================
# 示例1: 基本训练
# ============================================
echo "示例1: 基本训练"
python train.py --config configs/train_config.yaml

# ============================================
# 示例2: 指定输出目录
# ============================================
echo "示例2: 指定输出目录"
python train.py \
    --config configs/train_config.yaml \
    --output_dir output/experiment_001

# ============================================
# 示例3: 从checkpoint恢复训练
# ============================================
echo "示例3: 从checkpoint恢复训练"
python train.py \
    --config configs/train_config.yaml \
    --resume output/exp001/checkpoints/latest.pth

# ============================================
# 示例4: 使用不同的GPU
# ============================================
echo "示例4: 使用GPU 1"
CUDA_VISIBLE_DEVICES=1 python train.py \
    --config configs/train_config.yaml

# ============================================
# 示例5: 后台运行并保存日志
# ============================================
echo "示例5: 后台运行"
nohup python train.py \
    --config configs/train_config.yaml \
    > train.log 2>&1 &

echo "训练进程已启动，PID: $!"
echo "查看日志: tail -f train.log"

# ============================================
# 示例6: 使用tmux会话（推荐长时间训练）
# ============================================
echo "示例6: 在tmux会话中训练"
tmux new-session -d -s icpose_training
tmux send-keys -t icpose_training "cd $(pwd)" C-m
tmux send-keys -t icpose_training "python train.py --config configs/train_config.yaml" C-m

echo "tmux会话已创建: icpose_training"
echo "查看会话: tmux attach -t icpose_training"
echo "退出会话: Ctrl+B, 然后按D"

# ============================================
# 示例7: 查看TensorBoard
# ============================================
echo "示例7: 启动TensorBoard"
tensorboard --logdir output/exp001/logs --port 6006 &

echo "TensorBoard已启动"
echo "在浏览器打开: http://localhost:6006"
