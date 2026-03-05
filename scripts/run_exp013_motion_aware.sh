#!/bin/bash
# exp013: Motion-aware CorrPoseNet
# 
# 改进 vs exp012:
#   1. 深度信息注入: inverse depth → motion_encoder → 加到 correlation 特征上
#      让匹配过程感知 3D 几何 (远处/近处的 flow 幅度不同)
#   2. Flow 反馈: 上一次迭代的 flow 预测 → motion_encoder
#      提供预测历史上下文, 帮助 GRU 理解收敛方向
#   3. Correlation 加速: 预分配输出 + narrow 视图
#      消除 list + cat 开销, 训练速度 ~1.5x
#   4. 从 exp012 best_model 热启动 (motion_encoder 随机初始化)
#      利用已收敛的 correlation + GRU + flow_head 权重
#
# 新增参数: motion_encoder = 3,072 params (微不足道)
# 预期: 平移精度进一步提升 (深度感知), 训练收敛更快 (热启动)
#
# GPU: 1, BS=16, 预估 ~200-250s/epoch

set -e

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=1 python scripts/train_corr_pose.py \
    --output_dir output/corr_pose/exp013_motion_aware \
    --resume output/corr_pose/exp012_upsampler/best_model.pth \
    --scale fine_dino \
    --enc_dim 128 \
    --hidden_dim 128 \
    --corr_radius 4 \
    --num_iters 5 \
    --damping 1e-3 \
    --use_upsampler \
    --upsample_dim 64 \
    --upsample_scale 4 \
    --upsample_after_iter 2 \
    --use_motion_input \
    --render_chunk_size 256 \
    --epochs 150 \
    --batch_size 8 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --gamma 0.8 \
    --trans_weight 15.0 \
    --flow_loss_weight 1.0 \
    --grad_accum 1 \
    --curriculum "0:10,30:15,80:20" \
    --sim_data_dir output/sim_training_data/room_0 \
    --real_ratio 0.3 \
    --sim_epoch_size 1200 \
    --log_every 10 \
    --val_every 5 \
    --save_every 20 \
    --seq2_val_every 10 \
    --num_workers 2 \
    --seed 42
