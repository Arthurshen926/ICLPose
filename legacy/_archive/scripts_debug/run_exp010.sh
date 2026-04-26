#!/bin/bash
# EXP010启动脚本 - 完全对齐ICL-I2PReg实现

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate geo-aware

# 切换目录
cd /home/yons/Projects/SplatLoc/implicit_correspondence

# 训练
python train.py --config configs/train_config_exp010.yaml

# 查看日志
tail -f output/exp010/training.log
