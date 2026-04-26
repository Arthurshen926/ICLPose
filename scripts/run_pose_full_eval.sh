#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <config> <checkpoint> <gpu> [tag]"
  exit 1
fi

CONFIG="$1"
CHECKPOINT="$2"
GPU="$3"
TAG="${4:-}"

ROOT="/root/ICLPose-loc"
cd "$ROOT"

if [ ! -f "$CONFIG" ]; then
  echo "Config not found: $CONFIG"
  exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
  echo "Checkpoint not found: $CHECKPOINT"
  exit 1
fi

EXP_NAME=$(python - <<'PY' "$CONFIG"
import sys, yaml
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f) or {}
print(cfg.get('exp_name', 'unknown_exp'))
PY
)

read_config_value() {
  local key_expr="$1"
  python - <<'PY' "$CONFIG" "$key_expr"
import sys, yaml
config_path, key_expr = sys.argv[1], sys.argv[2]
with open(config_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f) or {}
value = cfg
for key in key_expr.split('.'):
    if not isinstance(value, dict):
        value = None
        break
    value = value.get(key)
print("" if value is None else value)
PY
}

OUTER_ITERS=$(read_config_value "training.outer_iters_val")
if [ -z "$OUTER_ITERS" ]; then
  OUTER_ITERS=$(read_config_value "training.outer_iters_train")
fi
if [ -z "$OUTER_ITERS" ]; then
  OUTER_ITERS=10
fi

GRU_ITERS=$(read_config_value "model.gru_iters")
if [ -z "$GRU_ITERS" ]; then
  GRU_ITERS=8
fi

OUT_ROOT="$ROOT/output/pose_refine/$EXP_NAME"
LOG_DIR="$OUT_ROOT/post_eval_logs"
mkdir -p "$LOG_DIR"

if [ -n "$TAG" ]; then
  SUFFIX="_$TAG"
else
  SUFFIX=""
fi

EVAL_ROOT="$OUT_ROOT/eval$SUFFIX"
PIPE_OUT="$EVAL_ROOT/pipeline_eval"
REALINIT_OUT_ROOT="$EVAL_ROOT"
RERANK_OUT="$EVAL_ROOT/real_init_reranker_top10_spatial"

mkdir -p "$EVAL_ROOT"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

run_step() {
  local name="$1"
  local log_path="$2"
  shift 2
  echo "[$(date '+%F %T')] Starting $name"
  "$@" > "$log_path" 2>&1
  echo "[$(date '+%F %T')] Finished $name"
}

run_step "pipeline_eval${SUFFIX}" "$LOG_DIR/pipeline_eval${SUFFIX}.log" \
  env CUDA_VISIBLE_DEVICES="$GPU" python -m pose_refine.evaluate_pipeline \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --gpu 0 \
    --num_neighbors 3 \
    --outer_iters "$OUTER_ITERS" \
    --gru_iters "$GRU_ITERS" \
    --selection residual \
    --multi_start 1 \
    --output_dir "$PIPE_OUT"

run_step "real_init_top1${SUFFIX}" "$LOG_DIR/real_init_top1${SUFFIX}.log" \
  env CUDA_VISIBLE_DEVICES="$GPU" python -m feature_retrieval.evaluate_impl \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --gpu 0 \
    --output_dir "$REALINIT_OUT_ROOT" \
    --batch_size 1 \
    --num_workers 2 \
    --outer_iters "$OUTER_ITERS" \
    --gru_iters "$GRU_ITERS" \
    --solver default \
    --retrieval_method auto \
    --retrieval_topk 1 \
    --pose_fusion none \
    --qual_limit 12 \
    --qual_dir "$OUT_ROOT/qual_real_init${SUFFIX}"

run_step "real_init_top10_none${SUFFIX}" "$LOG_DIR/real_init_top10_none${SUFFIX}.log" \
  env CUDA_VISIBLE_DEVICES="$GPU" python -m feature_retrieval.evaluate_impl \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --gpu 0 \
    --output_dir "$REALINIT_OUT_ROOT" \
    --batch_size 1 \
    --num_workers 2 \
    --outer_iters "$OUTER_ITERS" \
    --gru_iters "$GRU_ITERS" \
    --solver default \
    --retrieval_method auto \
    --retrieval_topk 10 \
    --pose_fusion none \
    --qual_limit 12 \
    --qual_dir "$OUT_ROOT/qual_real_init_top10_none${SUFFIX}"

run_step "reranker_top10_spatial${SUFFIX}" "$LOG_DIR/reranker_top10_spatial${SUFFIX}.log" \
  env CUDA_VISIBLE_DEVICES="$GPU" python -m feature_retrieval.train_impl \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --gpu 0 \
    --batch_size 1 \
    --num_workers 2 \
    --retrieval_method auto \
    --retrieval_topk 10 \
    --outer_iters "$OUTER_ITERS" \
    --gru_iters "$GRU_ITERS" \
    --solver default \
    --refine_chunk_size 1 \
    --epochs 40 \
    --hidden_dim 128 \
    --model_type spatial \
    --output_dir "$RERANK_OUT"

echo "Completed full eval chain for $EXP_NAME$SUFFIX"
