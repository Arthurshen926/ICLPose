#!/bin/bash
# exp010: 仿真+真实混合训练
# 基于 exp007 最佳配置: K=5, curriculum
# 新增: 仿真数据混合, real_ratio=0.3, epoch_size=1200
# GPU 0, 预计 200 epochs × ~20min/epoch
cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=0 python scripts/train_corr_pose.py \
  --sim_data_dir output/sim_training_data/room_0 \
  --real_ratio 0.3 \
  --sim_epoch_size 1200 \
  --num_iters 5 \
  --epochs 200 \
  --batch_size 1 \
  --grad_accum 4 \
  --lr 2e-4 \
  --weight_decay 1e-4 \
  --curriculum "0:5,50:10,100:15" \
  --corr_radius 4 \
  --gamma 0.8 \
  --trans_weight 10.0 \
  --flow_loss_weight 1.0 \
  --val_every 5 \
  --save_every 10 \
  --log_every 50 \
  --output_dir output/corr_pose/exp010_sim_mixed \
  --seed 42
