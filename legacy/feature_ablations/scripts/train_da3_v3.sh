#!/bin/bash
# DA3 Per-Scale Feature Training v3
# ==================================
# Key improvements over v1/v2:
#   1. Warm start from v2 best checkpoints
#   2. Lower LR (0.003) with warmup
#   3. Heavier cosine weight (2.0) — cosine is our target metric
#   4. More gradient accumulation for coarse/mid (8→16) to compensate sparse pixels
#   5. More iterations (30K for coarse/mid, 20K for fine)
#   6. Periodic evaluation logging
#
# Current quality:
#   v2 coarse: cos=0.539 ⚠  (target > 0.7)
#   v2 mid:    cos=0.260 ❌ (target > 0.6)
#   v2 fine:   cos=0.954 ✅  (already great)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
cd "$ROOT_DIR"


set -e

PLY="output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply"
FEAT_DIR="output/features_da3/OldHospital_indexed"
TRAJ="output/features_da3/OldHospital_indexed/traj_w_c.txt"
OUT_BASE="output/feature_3dgs/oldhospital_da3_perscale_v3"

# v2 checkpoints for warm start
V2_BASE="output/feature_3dgs/oldhospital_da3_perscale_v2"

echo "============================================"
echo "  DA3 Per-Scale Feature Training v3"
echo "============================================"

# --- COARSE ---
echo ""
echo ">>> Training COARSE (32d @ 15x26) ..."
CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONPATH=. python -m feature_3dgs.train_da3_perscale_v3 \
    --scale coarse \
    --ply_path "$PLY" \
    --feature_dir "$FEAT_DIR" \
    --traj_path "$TRAJ" \
    --output_dir "${OUT_BASE}/coarse" \
    --warmstart "${V2_BASE}/coarse/best_model.pth" \
    --num_iters 30000 \
    --lr 0.003 \
    --cos_weight 2.0 \
    --grad_accum 16 \
    --warmup_iters 1000 \
    --eval_interval 2000 \
    --precache_gpu

# --- MID ---
echo ""
echo ">>> Training MID (64d @ 30x53) ..."
CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONPATH=. python -m feature_3dgs.train_da3_perscale_v3 \
    --scale mid \
    --ply_path "$PLY" \
    --feature_dir "$FEAT_DIR" \
    --traj_path "$TRAJ" \
    --output_dir "${OUT_BASE}/mid" \
    --warmstart "${V2_BASE}/mid/best_model.pth" \
    --num_iters 30000 \
    --lr 0.003 \
    --cos_weight 2.0 \
    --grad_accum 16 \
    --warmup_iters 1000 \
    --eval_interval 2000 \
    --vis_interval 5000 \
    --precache_gpu

# --- FINE ---
echo ""
echo ">>> Training FINE (64d @ 69x121) ..."
CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONPATH=. python -m feature_3dgs.train_da3_perscale_v3 \
    --scale fine \
    --ply_path "$PLY" \
    --feature_dir "$FEAT_DIR" \
    --traj_path "$TRAJ" \
    --output_dir "${OUT_BASE}/fine" \
    --warmstart "${V2_BASE}/fine/best_model.pth" \
    --num_iters 20000 \
    --lr 0.001 \
    --cos_weight 2.0 \
    --grad_accum 16 \
    --warmup_iters 500 \
    --eval_interval 2000 \
    --vis_interval 4000 \
    --precache_gpu

echo ""
echo "============================================"
echo "  v3 ALL SCALES COMPLETE"
echo "============================================"

# Generate comparison images
echo ">>> Generating comparison images..."
CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONPATH=. python legacy/feature_ablations/scripts/visualize_da3_features.py
