#!/bin/bash

# 分布式训练启动脚本
# 使用torchrun启动2个GPU训练

# 设置可见的GPU
export CUDA_VISIBLE_DEVICES=0,1

# 使用torchrun启动分布式训练
# --nproc_per_node: 每个节点的进程数（GPU数量）
# --master_port: 主进程通信端口
torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    train.py \
    --config configs/train_config.yaml

# 如果使用老版本的PyTorch（<1.10），使用以下命令：
# python -m torch.distributed.launch \
#     --nproc_per_node=2 \
#     --master_port=29500 \
#     train.py \
#     --config configs/train_config.yaml
