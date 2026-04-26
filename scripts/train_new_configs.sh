#!/bin/bash
# Train new attention configs f (8-head + wider) and g (4-head + deeper)
# Plus more config e seeds (40-69) for better outliers
#
# Usage: GPU=0 CONFIG=f SEEDS="123 314 7" bash scripts/train_new_configs.sh

GPU=${GPU:-0}
CONFIG=${CONFIG:-f}
SEEDS=${SEEDS:-"123 314 7 24 38 42"}

COMMON="--pool attn --feat both+sum --patch_dim 128 --feature_dropout 0.15 --lr 0.001 --weight_decay 0.0001 --epochs 10000 --batch_size 128 --feature_dir output/feature_extract/features_radio_dual_128/OldHospital_pilot"

case $CONFIG in
  f)
    # Config f: 8 heads + wider MLP (2048-1024-512)
    # Feature dim: 128*8*2 + 2560 = 4608d, ~13M params
    EXTRA="--attn_heads 8 --hidden_dims 2048 1024 512 --dropout 0.15"
    PREFIX="exp24f_attn_h8_wider"
    ;;
  g)
    # Config g: 4 heads + even wider MLP (3072-1024-512)
    # Feature dim: 3584d, ~14M params
    EXTRA="--attn_heads 4 --hidden_dims 3072 1024 512 --dropout 0.2"
    PREFIX="exp24g_attn_h4_xwide"
    ;;
  e)
    # Config e (repeat): 4 heads + wider MLP (2048-1024-512)
    EXTRA="--attn_heads 4 --hidden_dims 2048 1024 512 --dropout 0.15"
    PREFIX="exp24e_attn_h4_wider"
    ;;
  *)
    echo "Unknown config: $CONFIG"
    exit 1
    ;;
esac

echo "[$(date)] Starting config=$CONFIG on GPU=$GPU, seeds=($SEEDS)"

for SEED in $SEEDS; do
  EXP_NAME="${PREFIX}_seed${SEED}"
  OUTDIR="output/feature_retrieval/pose_regression/${EXP_NAME}"
  
  if [ -d "$OUTDIR" ]; then
    echo "[$(date)] SKIP $EXP_NAME (already exists)"
    continue
  fi
  
  echo "[$(date)] Training $EXP_NAME on GPU $GPU ..."
  python -u feature_retrieval/patch_regressor_v7.py \
    $COMMON $EXTRA \
    --seed $SEED \
    --gpu $GPU \
    --exp_name "$EXP_NAME" \
    2>&1 | tail -5
  
  echo "[$(date)] Done $EXP_NAME"
done

echo "[$(date)] All done for config=$CONFIG on GPU=$GPU"
