#!/bin/bash
# exp016: Translation-focused fine-tuning from exp015 best model
#
# 核心改动 (vs exp015):
#   1. trans_weight=30 (from 20) — 50%更强的平移损失权重
#   2. flow_loss_weight=2.0 (from 1.0) — 2x更强的flow监督
#      更精确的flow直接 → 更精确的几何位姿求解 → 更好的平移
#   3. 从 exp015 best_model.pth 热启动 (fine-tune模式, optimizer重置)
#   4. curriculum: 0:10, 20:15, 50:20 — 更快的噪声递增
#
# 动机:
#   exp015 epoch 20 达到 Seq2 trans=28.2mm, 但 trans_weight=20 可能不够.
#   增加 trans_weight 直接让优化器更关注平移精度.
#   增加 flow_loss_weight 提供更强的flow像素级监督.
#
# 架构: 与 exp015 完全一致 (所有特性启用)
#   - Soft-argmax flow initialization
#   - Learnable per-iteration damping (7个可学习参数)
#   - K=7 iterations
#   - PoseAwareUpsampler (4x, 140×184)
#   - Motion encoder (inv_depth + flow feedback)
#
# 对比基线:
#   exp015 (epoch 20): Seq2 rot=0.44°, trans=28.2mm, <1°=82.6%
#   exp015 (epoch 40): Val rot=0.32° (record), Seq2 rot=0.49°, trans=31.3mm
#
# GPU 1 (exp013已完成, 空闲), BS=4×2=8 effective
# 预计显存: ~24GB, 速度: ~0.7s/sample

set -e

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=1 python scripts/train_corr_pose.py \
    --output_dir output/corr_pose/exp016_trans_focus \
    --resume output/corr_pose/exp015_flow_init/best_model.pth \
    --scale fine_dino \
    --enc_dim 128 \
    --hidden_dim 128 \
    --corr_radius 4 \
    --num_iters 7 \
    --damping 1e-3 \
    --use_upsampler \
    --upsample_dim 64 \
    --upsample_scale 4 \
    --upsample_after_iter 2 \
    --use_motion_input \
    --use_flow_init \
    --learnable_damping \
    --render_chunk_size 256 \
    --epochs 100 \
    --batch_size 4 \
    --lr 5e-5 \
    --weight_decay 1e-4 \
    --gamma 0.8 \
    --trans_weight 30.0 \
    --flow_loss_weight 2.0 \
    --grad_accum 2 \
    --curriculum "0:10,20:15,50:20" \
    --sim_data_dir output/sim_training_data/room_0 \
    --real_ratio 0.3 \
    --sim_epoch_size 1200 \
    --log_every 10 \
    --val_every 5 \
    --save_every 10 \
    --seq2_val_every 10 \
    --num_workers 2 \
    --seed 42
