#!/bin/bash
# DCFF v11: Joint geometry + feature training with channel-standardized loss
# Key changes from v10c:
#   1. Geometry NOT frozen (joint training)
#   2. Channel-standardized loss (lambda_channel_std=0.1)
#   3. Max batch size (32 vs 8) - fits in 24GB
#   4. Higher latent LR (0.0003 vs 0.0002)
#   5. Joint densification (from iter 100, until iter 5000)
#   6. Full geometry checkpoint saving for resume

set -e
export CUDA_VISIBLE_DEVICES=0

echo "======================================"
echo "DCFF v11: Joint Geometry Training"
echo "======================================"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Config: feature_field/configs/dcff_oldhospital_v11_joint_geo.yaml"
echo "Exp: dcff_oldhospital_v11_joint_geo"
echo "======================================"

python -m feature_field.train \
    --config feature_field/configs/dcff_oldhospital_v11_joint_geo.yaml \
    2>&1 | tee "feature_field/output/dcff_oldhospital_v11_joint_geo_launch.log"