#!/bin/bash
cd /home/yons/Projects/ICLPose
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$(conda run -n geo-aware python -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')"):$LD_LIBRARY_PATH
conda run --no-capture-output -n geo-aware bash scripts/retrain_v3_retrain7_final.sh
