#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <config> <checkpoint> <train_pid> <eval_gpu> [tag] [lock_file]"
  exit 1
fi

CONFIG="$1"
CHECKPOINT="$2"
TRAIN_PID="$3"
EVAL_GPU="$4"
TAG="${5:-}"
LOCK_FILE="${6:-/tmp/iclpose_pose_full_eval.lock}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXP_NAME=$(python - <<'PY' "$CONFIG"
import sys, yaml
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f) or {}
print(cfg.get('exp_name', 'unknown_exp'))
PY
)

LOG_DIR="$ROOT/output/pose_refine/$EXP_NAME/post_eval_logs"
mkdir -p "$LOG_DIR"

if [ -n "$TAG" ]; then
  SUFFIX="_$TAG"
else
  SUFFIX=""
fi

WATCH_LOG="$LOG_DIR/watcher${SUFFIX}.log"

{
  echo "[$(date '+%F %T')] Watching PID $TRAIN_PID for $EXP_NAME"
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 60
  done
  echo "[$(date '+%F %T')] Train PID $TRAIN_PID exited"

  if [ ! -f "$CHECKPOINT" ]; then
    echo "[$(date '+%F %T')] Checkpoint missing, skip eval: $CHECKPOINT"
    exit 1
  fi

  echo "[$(date '+%F %T')] Waiting for eval lock: $LOCK_FILE"
  flock "$LOCK_FILE" bash "$ROOT/scripts/run_pose_full_eval.sh" "$CONFIG" "$CHECKPOINT" "$EVAL_GPU" "$TAG"
  echo "[$(date '+%F %T')] Full eval finished for $EXP_NAME"
} >> "$WATCH_LOG" 2>&1
