#!/usr/bin/env bash
set -euo pipefail

repo=/root/ICLPose
root=$repo/output/vfm/2dgs_surface/StMarysChurch/full_train/goal_maplet/g23_official_train_oof_v1
manifests=$root/geometry_folds
export PYTHONPATH=$repo

manifest_force=()
if [[ ${G23_FORCE:-0} == 1 ]]; then
  manifest_force=(--force)
fi
if [[ ! -e $manifests/summary.json || ${G23_FORCE:-0} == 1 ]]; then
  python "$repo/feature_extract/tools/vfm/build_goal_maplet_geometry_oof_manifests.py" \
    --geometry_manifests \
    "$root/geometry_labels_exact_shard0/geometry_manifest.json" \
    "$root/geometry_labels_exact_shard1/geometry_manifest.json" \
    --protocol_json "$repo/configs/vfm/goal_maplet_stmarys_official_train_oof_v1.json" \
    --output_dir "$manifests" "${manifest_force[@]}"
fi

run_fold() {
  local gpu=$1
  local fold=$2
  local output=$root/$fold/geometry_head
  if [[ -e $output/radio_highres_geometry_head.pt && -e $output/geometry_head_summary.json && ${G23_FORCE:-0} != 1 ]]; then
    return
  fi
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES=$gpu python \
    "$repo/feature_extract/tools/vfm/train_vfm_highres_geometry_head.py" \
    --train_geometry_manifest "$manifests/${fold}_train.json" \
    --eval_geometry_manifest "$manifests/${fold}_eval.json" \
    --output_dir "$output" --epochs 30 \
    --checkpoint_protocol fixed_epoch_no_selection --batch_size 16 \
    --hidden_channels 128 --architecture separate_decoders --task multitask \
    --lr 0.001 --weight_decay 0.0001 --normal_weight 1.0 \
    --confidence_weight 0.1 --cache_in_memory --amp --visualize 0 \
    --seed "$((2400 + ${fold#fold}))" --device cuda:0 \
    > "$output/training.log" 2>&1
}

gpu0() {
  run_fold 0 fold0
  run_fold 0 fold2
  run_fold 0 fold4
}
gpu1() {
  run_fold 1 fold1
  run_fold 1 fold3
}

gpu0 &
pid0=$!
gpu1 &
pid1=$!
wait "$pid0"
wait "$pid1"
