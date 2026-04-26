#!/bin/bash
# K-neighbor sweep: accumulated LoFTR-only (outer_iters=0)
# Runs K=30, K=10, K=5 on GPUs 0, 1, 2 respectively

source /root/miniconda3/etc/profile.d/conda.sh
conda activate iclpose
cd /root/ICLPose-loc

CONFIG="pose_refine/configs/concat_loc_oh_v20k_v5l_pointwise_featsharp_full_fsm_v1_b4_bs4.yaml"
CKPT="output/pose_refine/concat_loc_oh_v20k_v5l_pointwise_featsharp_full_fsm_v1_b4_bs4/checkpoints/best.pth"

echo "Launching K=30 on GPU 0..."
python -m pose_refine.evaluate_pipeline \
  --config $CONFIG --checkpoint $CKPT \
  --gpu 0 --num_neighbors 30 --loftr_mode accumulated --outer_iters 0 --gru_iters 8 \
  > output/pipeline_eval/oracle_k30_accumulated_oi0.log 2>&1 &
PID_K30=$!

echo "Launching K=10 on GPU 1..."
python -m pose_refine.evaluate_pipeline \
  --config $CONFIG --checkpoint $CKPT \
  --gpu 1 --num_neighbors 10 --loftr_mode accumulated --outer_iters 0 --gru_iters 8 \
  > output/pipeline_eval/oracle_k10_accumulated_oi0.log 2>&1 &
PID_K10=$!

echo "Launching K=5 on GPU 2..."
python -m pose_refine.evaluate_pipeline \
  --config $CONFIG --checkpoint $CKPT \
  --gpu 2 --num_neighbors 5 --loftr_mode accumulated --outer_iters 0 --gru_iters 8 \
  > output/pipeline_eval/oracle_k5_accumulated_oi0.log 2>&1 &
PID_K5=$!

echo "Launched: K30=$PID_K30, K10=$PID_K10, K5=$PID_K5"
echo "Waiting for all to complete..."
wait $PID_K30 $PID_K10 $PID_K5
echo "All done!"
echo ""
echo "=== K=30 Results ==="
tail -30 output/pipeline_eval/oracle_k30_accumulated_oi0.log
echo ""
echo "=== K=10 Results ==="
tail -30 output/pipeline_eval/oracle_k10_accumulated_oi0.log
echo ""
echo "=== K=5 Results ==="
tail -30 output/pipeline_eval/oracle_k5_accumulated_oi0.log
