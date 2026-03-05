#!/bin/bash
# exp014: Ablation — No upsampler, motion_input only, BS=32
# 目的: 验证 motion_input 在原生 35×46 分辨率的效果
#        去掉 upsampler 后速度应该提升 ~2x (0.33→~0.15 s/sample)
#        如果精度相当，说明 motion_input 是主要贡献者
#
# 对比:
#   exp012: upsampler ON, motion OFF, BS=8,  0.33s/sample, 200ep
#   exp013: upsampler ON, motion ON,  BS=8,  0.33s/sample, 150ep
#   exp014: upsampler OFF, motion ON, BS=32, ~0.12s/sample, 200ep
#
# GPU: 0 (exp013 在 GPU 1 继续)
# 从 exp013 best_model 热启动 (strict=False, upsampler 权重被丢弃)

set -e

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=0 python scripts/train_corr_pose.py \
    --output_dir output/corr_pose/exp014_no_upsample \
    --resume output/corr_pose/exp013_motion_aware/best_model.pth \
    --scale fine_dino \
    --enc_dim 128 \
    --hidden_dim 128 \
    --corr_radius 4 \
    --num_iters 5 \
    --damping 1e-3 \
    --use_motion_input \
    --render_chunk_size 256 \
    --epochs 200 \
    --batch_size 16 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --gamma 0.8 \
    --trans_weight 10.0 \
    --flow_loss_weight 1.0 \
    --grad_accum 1 \
    --curriculum "0:5,30:10,80:15" \
    --sim_data_dir output/sim_training_data/room_0 \
    --real_ratio 0.3 \
    --sim_epoch_size 1200 \
    --log_every 5 \
    --val_every 5 \
    --save_every 20 \
    --seq2_val_every 20 \
    --num_workers 4 \
    --seed 42
