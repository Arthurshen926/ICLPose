#!/bin/bash
# exp011: 从 exp007 最佳模型 fine-tune + 仿真数据混合
# 策略: 保留已学到的强基线, 通过sim数据增加泛化
# GPU 1, 较低学习率 + 100 epochs, curriculum从10°开始
cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=1 python scripts/train_corr_pose.py \
  --resume output/corr_pose/exp007_curriculum_K5/best_model.pth \
  --sim_data_dir output/sim_training_data/room_0 \
  --real_ratio 0.5 \
  --sim_epoch_size 1000 \
  --num_iters 5 \
  --epochs 100 \
  --batch_size 1 \
  --grad_accum 4 \
  --lr 5e-5 \
  --weight_decay 1e-4 \
  --curriculum "0:10,30:15" \
  --corr_radius 4 \
  --gamma 0.8 \
  --trans_weight 10.0 \
  --flow_loss_weight 1.0 \
  --val_every 5 \
  --save_every 10 \
  --log_every 50 \
  --output_dir output/corr_pose/exp011_finetune_sim \
  --seed 42
