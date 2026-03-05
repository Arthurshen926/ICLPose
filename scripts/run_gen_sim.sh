#!/bin/bash
# 生成仿真训练数据: 3000 个样本 (float16, ~7GB)
# GPU 0, 预计 ~40 分钟
cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=0 python scripts/generate_sim_training_data.py \
  --num_poses 3000 \
  --output_dir output/sim_training_data/room_0 \
  --margin 0.3 \
  --nearby_ratio 0.7 \
  --nearby_pos_std 0.3 \
  --nearby_rot_std 15.0 \
  --min_alpha_ratio 0.5 \
  --min_depth 0.1 \
  --save_rgb \
  --rgb_interval 100
