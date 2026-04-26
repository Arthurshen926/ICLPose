#!/bin/bash
# ============================================================
#  run_da3_pipeline.sh
#  Complete DA3 feature pipeline: 3DGS training → Pose training
#
#  Prerequisites:
#    - DA3 features extracted to output/features_da3/OldHospital_indexed
#    - PLY file at output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply
#
#  Usage:
#    CUDA_VISIBLE_DEVICES=4 bash legacy/feature_ablations/scripts/run_da3_pipeline.sh
# ============================================================
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
cd "$ROOT_DIR"


set -e

# Configurable paths
FEATURE_DIR="output/features_da3/OldHospital_indexed"
PLY_PATH="output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply"
TRAJ_PATH="output/features_da3/OldHospital_indexed/traj_w_c.txt"
GSPLAT_DIR="output/feature_3dgs/oldhospital_da3"
POSE_CONFIG="legacy/feature_ablations/configs/exp180_oh_da3.yaml"

# Verify features exist
echo "=== Verifying DA3 features ==="
for scale in fine mid coarse; do
    n=$(ls "${FEATURE_DIR}/${scale}/"*.pt 2>/dev/null | wc -l)
    f=$(ls "${FEATURE_DIR}/${scale}/"*.pt 2>/dev/null | head -1)
    echo "  ${scale}: ${n} files, first: $(basename "${f}" 2>/dev/null)"
done

# Step 1: Train 3DGS feature embeddings
echo ""
echo "=== Step 1: Training 3DGS feature embeddings ==="
PYTHONPATH=. python -m feature_3dgs.train_flowfeat_embedding \
    --ply_path "${PLY_PATH}" \
    --feature_dir "${FEATURE_DIR}" \
    --traj_path "${TRAJ_PATH}" \
    --output_dir "${GSPLAT_DIR}" \
    --img_height 1080 --img_width 1920 \
    --fx 1663.12 --fy 1663.12 --cx 960.0 --cy 540.0 \
    --num_iters 15000 --grad_accum 4 --precache_gpu \
    --lr 0.01 --cos_weight 1.0 \
    --w_fine 1.0 --w_mid 1.0 --w_coarse 1.0 \
    --log_interval 100 --save_interval 5000 \
    2>&1 | tee output/da3_3dgs_train.log

echo ""
echo "=== Step 1 complete. Checking render quality ==="
# Render quality will be reported in the training log

# Step 2: Train pose estimation network
echo ""
echo "=== Step 2: Training pose network ==="
python scripts/train_ms_flow.py \
    --config "${POSE_CONFIG}" \
    2>&1 | tee output/exp180_train.log

echo ""
echo "=== Pipeline complete ==="
grep '★' output/exp180_train.log | tail -5
