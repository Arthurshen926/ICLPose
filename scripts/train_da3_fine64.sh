#!/bin/bash
# 只训练 64-dim fine 特征 (69x121)
# 这是最具代表性的 DA3 特征，不再做 coarse/mid 多尺度
#
# 数据: features_da3/OldHospital_indexed/fine/  →  (64, 69, 121)
# 模型: output/feature_3dgs/oldhospital_da3_fine64/

set -e
cd /root/ICLPose

PLY="output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply"
FEAT_DIR="output/features_da3/OldHospital_indexed"
TRAJ="output/features_da3/OldHospital_indexed/traj_w_c.txt"
OUT="output/feature_3dgs/oldhospital_da3_fine64"

# warm start from oldhospital_da3_perscale_v2/fine (64-dim, 已有基础; unified_v2 同 perscale_v2 内容)
WARMSTART="output/feature_3dgs/oldhospital_da3_perscale_v2/fine/best_model.pth"

echo "============================================"
echo "  DA3 Fine-64 Training (64-dim @ 69x121)"
echo "  全新训练 (无 warmstart)"
echo "============================================"

CUDA_VISIBLE_DEVICES=${GPU:-4} nohup python -m feature_3dgs.train_da3_perscale_v3 \
    --scale fine \
    --ply_path "$PLY" \
    --feature_dir "$FEAT_DIR" \
    --traj_path "$TRAJ" \
    --output_dir "$OUT" \
    --num_iters 30000 \
    --lr 0.003 \
    --cos_weight 2.0 \
    --grad_accum 16 \
    --warmup_iters 500 \
    --eval_interval 2000 \
    --vis_interval 5000 \
    --precache_gpu \
    > output/da3_fine64_train.log 2>&1 &

echo "PID: $!"
echo "Log: output/da3_fine64_train.log"
echo "Watch: tail -f output/da3_fine64_train.log"
