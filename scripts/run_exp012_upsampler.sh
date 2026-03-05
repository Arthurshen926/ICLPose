#!/bin/bash
# exp012: Task-driven feature transform (PoseAwareUpsampler)
# 768d@35×46 → 64d@140×184, 联合训练, 提升平移精度
#
# 关键改进:
#   - PoseAwareUpsampler: 降维+上采样, pose loss 端到端驱动
#   - Image Jacobian 在 140×184 (fx=92), 1cm→0.46px (vs 原来 0.115px)
#   - 预期: rotation 保持 ~0.4°, translation 从 5cm → 1-2cm
#
# GPU: 1 (释放 exp011 的 GPU)
# 预估速度: ~0.20s/sample (coarse-to-fine 比全高分辨率快)
# 前 2 次迭代在 35×46 粗定位, 后 3 次在 140×184 精修

set -e

cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=1 python scripts/train_corr_pose.py \
    --output_dir output/corr_pose/exp012_upsampler \
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
    --render_chunk_size 256 \
    --epochs 200 \
    --batch_size 8 \
    --lr 2e-4 \
    --weight_decay 1e-4 \
    --gamma 0.8 \
    --trans_weight 10.0 \
    --flow_loss_weight 1.0 \
    --grad_accum 1 \
    --curriculum "0:5,50:10,100:15" \
    --sim_data_dir output/sim_training_data/room_0 \
    --real_ratio 0.3 \
    --sim_epoch_size 1200 \
    --log_every 10 \
    --val_every 5 \
    --save_every 20 \
    --seq2_val_every 20 \
    --num_workers 2 \
    --seed 42
