#!/bin/bash
# exp015: Soft-argmax flow init + learnable damping + K=7 iterations
# 
# 三个架构改进:
#   1. Soft-argmax flow initialization — correlation peak → initial flow estimate
#      flow_head 只需预测 residual (sub-pixel correction), 收敛更快
#   2. Learnable per-iteration damping — 网络自己学习每次迭代的最优 LM 阻尼
#      初始 1e-3, 训练中自适应调整
#   3. K=7 iterations (from K=5) — 更多精修步骤, 尤其有利于平移精度
#
# Memory budget: K=7 增加 40% 激活缓存, 需降 BS=4 + grad_accum=2 (有效 BS=8)
#
# 对比:
#   exp012: upsampler ON, motion OFF, K=5, BS=8,  200ep, Seq2: 0.56° / 37.0mm
#   exp013: upsampler ON, motion ON,  K=5, BS=8,  150ep, Seq2: 0.50° / 33.4mm @ep80
#   exp015: upsampler ON, motion ON,  K=7, BS=4×2, + flow_init + learnable_damping
#
# GPU: 等 exp014 完成后用 GPU 0, 或 exp013 完成后用 GPU 1
# 从 exp013 best_model 热启动

set -e

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=0 python scripts/train_corr_pose.py \
    --output_dir output/corr_pose/exp015_flow_init \
    --resume output/corr_pose/exp013_motion_aware/best_model.pth \
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
    --epochs 150 \
    --batch_size 4 \
    --lr 5e-5 \
    --weight_decay 1e-4 \
    --gamma 0.8 \
    --trans_weight 20.0 \
    --flow_loss_weight 1.0 \
    --grad_accum 2 \
    --curriculum "0:10,30:15,80:20" \
    --sim_data_dir output/sim_training_data/room_0 \
    --real_ratio 0.3 \
    --sim_epoch_size 1200 \
    --log_every 10 \
    --val_every 5 \
    --save_every 10 \
    --seq2_val_every 10 \
    --num_workers 2 \
    --seed 42
