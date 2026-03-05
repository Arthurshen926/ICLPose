#!/bin/bash
# exp011: 从 checkpoint_20 恢复 (fine-tune from exp007)
# 改动: batch_size=32, chunk_size=256 (减少一半 rasterization 调用)
cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=1 python scripts/train_corr_pose.py \
    --output_dir output/corr_pose/exp011_finetune_sim \
    --resume output/corr_pose/exp011_finetune_sim/checkpoint_20.pth \
    --sim_data_dir output/sim_training_data/room_0 \
    --real_ratio 0.5 \
    --sim_epoch_size 1000 \
    --num_iters 5 \
    --batch_size 32 \
    --grad_accum 1 \
    --render_chunk_size 256 \
    --lr 5e-5 \
    --epochs 100 \
    --curriculum "0:10,30:15" \
    --val_every 5 \
    --save_every 10 \
    --seq2_val_every 20 \
    --log_every 10 \
    --num_workers 4
