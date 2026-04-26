#!/bin/bash
# Train patch-token grid classifier experiments.

GPU=${GPU:-0}
CELLS=${CELLS:-64}
SEEDS=${SEEDS:-"123"}
EXTRA_ARGS=${EXTRA_ARGS:-}

COMMON="--pool attn --feat both+sum --patch_dim 128 --feature_dir output/feature_extract/features_radio_dual_128/OldHospital_pilot --lr 0.001 --weight_decay 0.0001 --epochs 10000 --batch_size 128 --attn_heads 4 --hidden_dims 2048 1024 512 --dropout 0.15 --feature_dropout 0.15"

echo "[$(date)] Starting grid classifier on GPU=$GPU, cells=$CELLS, seeds=($SEEDS)"

for SEED in $SEEDS; do
  EXP_NAME="exp26_gridcls_k${CELLS}_seed${SEED}"
  OUTDIR="output/feature_retrieval/pose_regression/${EXP_NAME}"

  if [ -f "$OUTDIR/model_best.pt" ]; then
    echo "[$(date)] SKIP $EXP_NAME (already completed)"
    continue
  fi

  if [ -d "$OUTDIR" ] && [ ! -f "$OUTDIR/model_best.pt" ]; then
    rm -rf "$OUTDIR"
  fi

  echo "[$(date)] Training $EXP_NAME ..."
  python -u feature_retrieval/patch_grid_classifier_v1.py \
    $COMMON \
    --gpu $GPU \
    --n_cells $CELLS \
    --seed $SEED \
    --exp_name "$EXP_NAME" \
    $EXTRA_ARGS
done

echo "[$(date)] Done cells=$CELLS on GPU=$GPU"
