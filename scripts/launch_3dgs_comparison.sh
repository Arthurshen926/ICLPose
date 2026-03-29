#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# 2DGS vs 3DGS Comparison — OldHospital
# ═══════════════════════════════════════════════════════════════════
#
# 对比实验:
#   A) 2DGS (surfel) geometry — 当前方案 (rasterization_2dgs)
#   B) 3DGS (ellipsoid) geometry — STDLoc 方案 (rasterization)
#
# 前置条件:
#   1. CLAHE 预处理完成: dataset/OldHospital/clahe_images/
#   2. DA3 features 已 extract: output/features_da3/OldHospital_indexed/
#   3. 6-class semantic masks: output/features_da3_unified/OldHospital/masks/
#
# Usage:
#   bash scripts/launch_3dgs_comparison.sh [gpu_id]
# ═══════════════════════════════════════════════════════════════════

set -e
GPU=${1:-0}

echo "═══════════════════════════════════════════════════"
echo " 2DGS vs 3DGS Comparison — OldHospital"
echo "═══════════════════════════════════════════════════"

# ── Step 1: CLAHE preprocessing (if not done) ──
if [ ! -d "dataset/OldHospital/clahe_images" ]; then
    echo "[1/4] Running CLAHE preprocessing..."
    CUDA_VISIBLE_DEVICES=$GPU python scripts/preprocess_clahe.py \
        --image_dir dataset/OldHospital \
        --output_dir dataset/OldHospital/clahe_images
else
    echo "[1/4] CLAHE images already exist, skipping."
fi

# ── Step 2: 2DGS joint training (baseline) ──
echo "[2/4] Training 2DGS joint (v4_clahe)..."
CUDA_VISIBLE_DEVICES=$GPU python -m feature_3dgs.train_2dgs_joint_v3 \
    --config configs/joint_oh_v4_clahe.yaml \
    2>&1 | tee output/2dgs_joint/joint_oh_v4_clahe/train.log

# ── Step 3: 3DGS joint training (comparison) ──
# Uses FORCE_3DGS_RASTERIZATION=1 to switch feature rendering to 3DGS
# Geometry rendering also uses 3DGS via the --use_3dgs flag
echo "[3/4] Training 3DGS joint (v4_3dgs_comparison)..."
CUDA_VISIBLE_DEVICES=$GPU FORCE_3DGS_RASTERIZATION=1 \
    python -m feature_3dgs.train_2dgs_joint_v3 \
    --config configs/joint_oh_v4_3dgs.yaml \
    2>&1 | tee output/2dgs_joint/joint_oh_v4_3dgs/train.log

# ── Step 4: Comparison ──
echo ""
echo "═══════════════════════════════════════════════════"
echo " Results"
echo "═══════════════════════════════════════════════════"
echo ""
echo "2DGS results:"
grep -E '★|PSNR=' output/2dgs_joint/joint_oh_v4_clahe/train.log | tail -5
echo ""
echo "3DGS results:"
grep -E '★|PSNR=' output/2dgs_joint/joint_oh_v4_3dgs/train.log | tail -5
echo ""
echo "Compare feature quality, PSNR, and Gaussian count between 2DGS and 3DGS."
echo "Visualizations in output/2dgs_joint/joint_oh_v4_{clahe,3dgs}/visualizations/"
