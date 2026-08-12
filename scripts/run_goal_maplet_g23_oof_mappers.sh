#!/usr/bin/env bash
set -euo pipefail

repo=/root/ICLPose
base=$repo/output/vfm/2dgs_surface/StMarysChurch/full_train
root=$base/goal_maplet/g23_official_train_oof_v1
maplets=$base/bootstrap_surface/surface_maplets.npz
manifest=$repo/output/vfm_tokens/StMarysChurch/full_1024x576/train_manifest.json
export PYTHONPATH=$repo

run_fold() {
  local gpu=$1
  local fold=$2
  shift 2
  local held_count=$1
  shift
  local held=("${@:1:$held_count}")
  shift "$held_count"
  local mapping=("$@")
  local output=$root/$fold/surface_mapper.pt
  local summary=$root/$fold/surface_mapper.json
  if [[ -e $output && -e $summary && ${G23_FORCE:-0} != 1 ]]; then
    return
  fi
  local force=()
  if [[ ${G23_FORCE:-0} == 1 ]]; then
    force=(--force)
  fi
  mkdir -p "$root/$fold"
  CUDA_VISIBLE_DEVICES=$gpu python \
    "$repo/feature_extract/tools/vfm/train_surface_maplet_mapper.py" \
    --surface_maplets "$maplets" --radio_final_manifest "$manifest" \
    --output_checkpoint "$output" --summary_json "$summary" \
    --checkpoint_protocol fixed_epoch_no_selection \
    --training_trajectory_ids "${mapping[@]}" \
    --strict_holdout_trajectory_ids "${held[@]}" seq3 seq5 seq13 \
    --epochs 120 --steps_per_epoch 8 --batch_maplets 64 \
    --hidden_dim 256 --output_dim 128 --dropout 0 \
    --learning_rate 0.0002 --weight_decay 0.0001 \
    --temperature 0.07 --hard_negative_radius 0.75 \
    --hard_negative_margin 0.25 --hard_negative_weight 0.20 \
    --seed "$((2300 + ${fold#fold}))" --device cuda:0 \
    "${force[@]}" > "$root/$fold/surface_mapper.log" 2>&1
}

gpu0() {
  run_fold 0 fold0 1 seq2 seq1 seq4 seq6 seq7 seq8 seq9 seq10 seq11 seq12 seq14
  run_fold 0 fold2 3 seq1 seq8 seq14 seq2 seq4 seq6 seq7 seq9 seq10 seq11 seq12
  run_fold 0 fold4 4 seq6 seq9 seq10 seq11 seq1 seq2 seq4 seq7 seq8 seq12 seq14
}

gpu1() {
  run_fold 1 fold1 1 seq4 seq1 seq2 seq6 seq7 seq8 seq9 seq10 seq11 seq12 seq14
  run_fold 1 fold3 2 seq7 seq12 seq1 seq2 seq4 seq6 seq8 seq9 seq10 seq11 seq14
}

gpu0 &
pid0=$!
gpu1 &
pid1=$!
wait "$pid0"
wait "$pid1"
