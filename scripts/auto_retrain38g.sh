#!/bin/bash
# Auto-launch retrain38g when exp032 (PID 4177) finishes
# Run: nohup bash scripts/auto_retrain38g.sh &> logs/auto_retrain38g.log &

set -e

EXP032_PID=4177
LOG_PREFIX="[auto_retrain38g]"

echo "$LOG_PREFIX Waiting for exp032 (PID $EXP032_PID) to finish..."
echo "$LOG_PREFIX Started monitoring at $(date)"

# Wait for exp032 to finish
while kill -0 $EXP032_PID 2>/dev/null; do
    sleep 60
done

echo "$LOG_PREFIX exp032 finished at $(date)"
echo "$LOG_PREFIX Waiting 30 seconds for GPU memory to be released..."
sleep 30

# Check GPU 1 is free
GPU1_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')
echo "$LOG_PREFIX GPU 1 memory usage: ${GPU1_MEM}MiB"

if [ "$GPU1_MEM" -gt 2000 ]; then
    echo "$LOG_PREFIX WARNING: GPU 1 still has ${GPU1_MEM}MiB used. Aborting."
    exit 1
fi

echo "$LOG_PREFIX GPU 1 is free. Launching retrain38g..."

cd /home/yons/Projects/ICLPose
source activate geo-aware 2>/dev/null || conda activate geo-aware 2>/dev/null || true

# retrain38g: bigger MLP (256×3), output_scale=0.5, appearance_reg=0.001
# NEW: --no_dino_uncertainty (saves ~1GB, use static masks instead)
# NEW: --wg_image_embed_dim 64 (more expressive for test-time opt)
MODEL_DIR=output/2dgs_models/OldHospital/v3_retrain38g
mkdir -p "$MODEL_DIR"

CUDA_VISIBLE_DEVICES=1 python -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir "$MODEL_DIR" \
    --iterations 30000 \
    --batch_size 4 \
    --use_mask \
    --random_background \
    --use_appearance \
    --wildgaussians \
    --no_dino_uncertainty \
    --wg_output_scale 0.5 \
    --wg_hidden_dim 256 \
    --wg_n_hidden 3 \
    --wg_image_embed_dim 64 \
    --lambda_dssim 0.2 \
    --lambda_normal 0 \
    --lambda_dist 0 \
    --lambda_scale 0 \
    --lambda_depth 0 \
    --densify_until_iter 15000 \
    --densify_grad_threshold 0.0002 \
    --max_gaussians 200000 \
    --opacity_reset_interval 30001 \
    --appearance_lr_init 5e-4 \
    --gaussian_emb_lr 5e-3 \
    --image_emb_lr 1e-3 \
    --appearance_reg 0.001 \
    --grad_clip_max_norm 1.0 \
    --test_iterations 1000 5000 10000 15000 20000 25000 30000 \
    --save_iterations 10000 15000 30000 \
    --wg_test_opt_steps 0 \
    2>&1 | tee "$MODEL_DIR/train.log"

echo "$LOG_PREFIX retrain38g finished at $(date)"

# Auto-run eval with improved test-time optimization (100 steps)
echo "$LOG_PREFIX Running eval-only with 100-step test-time opt..."
CUDA_VISIBLE_DEVICES=1 python -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir "$MODEL_DIR" \
    --use_mask \
    --wildgaussians \
    --eval_only \
    --checkpoint_iter 30000 \
    --wg_test_opt_steps 100 \
    --lambda_dssim 0.2 \
    2>&1 | tee -a "$MODEL_DIR/eval_30k_100steps.log"

echo "$LOG_PREFIX All done at $(date)"
