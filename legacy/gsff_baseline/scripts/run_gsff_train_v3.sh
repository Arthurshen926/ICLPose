#!/bin/bash
# GSFFs v3 training: warmstart from v1, 200K iterations, cosine loss, StepLR
# Key improvements over v1:
#   1. 4x longer training (200K vs 50K)
#   2. Direct cosine similarity loss (weight=0.5) for feature matching
#   3. StepLR scheduler (halve LR every 50K) instead of CosineAnnealing
#   4. Warmstart from v1 final.pth (best model so far)
#   5. No warmup phase (already trained, start with full losses)
#   6. Uses OldFineEncoder (v1-compatible, proven to generalize better)
#   7. More NCE samples (2048 vs 1024) for better gradient signal
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
cd "$ROOT_DIR"


export CUDA_VISIBLE_DEVICES=3

python legacy/gsff_baseline/scripts/train_gsff.py \
    --source_dir dataset/OldHospital \
    --model_path output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply \
    --cameras_json output/2dgs_models/OldHospital/v7_depth/cameras.json \
    --output_dir output/gsff/OldHospital_v3b \
    --warmstart output/gsff/OldHospital/checkpoints/final.pth \
    --total_iters 200000 \
    --phase1_iters 0 \
    --lr_triplane 5e-4 \
    --lr_encoder 5e-5 \
    --cosine_loss_weight 0.5 \
    --scheduler step \
    --lr_step_size 50000 \
    --nce_samples 2048 \
    --save_freq 10000 \
    --use_old_fine_encoder \
    --render_height 540 \
    --render_width 960
