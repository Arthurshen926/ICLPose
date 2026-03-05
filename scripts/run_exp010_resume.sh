#!/bin/bash
# exp010: 从 checkpoint_50 恢复 (from scratch, sim mixed)
# 改动: batch_size=32, chunk_size=256 (减少一半 rasterization 调用)
cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=0 python scripts/train_corr_pose.py \
    --output_dir output/corr_pose/exp010_sim_mixed \
    --resume output/corr_pose/exp010_sim_mixed/checkpoint_50.pth \
    --sim_data_dir output/sim_training_data/room_0 \
    --real_ratio 0.3 \
    --sim_epoch_size 1200 \
    --num_iters 5 \
    --batch_size 32 \
    --grad_accum 1 \
    --render_chunk_size 256 \
    --lr 0.0002 \
    --epochs 200 \
    --curriculum "0:5,50:10,100:15" \
    --val_every 5 \
    --save_every 10 \
    --seq2_val_every 20 \
    --log_every 10 \
    --num_workers 4
