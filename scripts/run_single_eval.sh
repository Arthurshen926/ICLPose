#!/bin/bash
# Run K-sweep directly with unbuffered Python output
source /root/miniconda3/etc/profile.d/conda.sh
conda activate iclpose
cd /root/ICLPose-loc

CONFIG="pose_refine/configs/concat_loc_oh_v20k_v5l_pointwise_featsharp_full_fsm_v1_b4_bs4.yaml"
CKPT="output/pose_refine/concat_loc_oh_v20k_v5l_pointwise_featsharp_full_fsm_v1_b4_bs4/checkpoints/best.pth"

export PYTHONUNBUFFERED=1

echo "Starting K=$1 on GPU $2..."
python -u -m pose_refine.evaluate_pipeline \
  --config $CONFIG --checkpoint $CKPT \
  --gpu $2 --num_neighbors $1 --loftr_mode accumulated --outer_iters 0 --gru_iters 8
echo "K=$1 DONE with exit code $?"
