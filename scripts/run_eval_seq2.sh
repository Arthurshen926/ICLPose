#!/bin/bash
# Evaluate exp007 on Seq2 test set
cd /home/yons/Projects/ICLPose
CUDA_VISIBLE_DEVICES=1 python scripts/eval_corrpose_iters.py \
  --checkpoint output/corr_pose/exp007_curriculum_K5/best_model.pth \
  --test_seq2 \
  --iters 3 5 7 10
