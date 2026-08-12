#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <fold0..fold4|final_alltrain> <physical_gpu_index>" >&2
  exit 2
fi

repo=/root/ICLPose
fold=$1
gpu=$2
base=$repo/output/vfm/2dgs_surface/StMarysChurch/full_train
root=$base/goal_maplet/g23_strict_oof_geometry_v1
dir=$root/$fold
protocol=$repo/configs/vfm/goal_maplet_stmarys_official_train_oof_v1.json
iterations=${G23_MATCHA_GAUSSIAN_ITERATIONS:-30000}
ply=$dir/strict_geometry/free_gaussians/point_cloud/iteration_${iterations}/point_cloud.ply
contributors=$dir/contributors_official_train
physical=$dir/physical_map.npz
mapper=$dir/surface_mapper.pt
teachers=$base/mainline_v6/v8_offline_radio_teacher_regions_alltrain
export PYTHONPATH=$repo

if [[ ! -e $dir/contributors_official_train_audit.json || ! -e $physical ]]; then
  echo "strict contributors or physical map are incomplete for $fold" >&2
  exit 2
fi
mapfile -t mapping_routes < <(python - "$dir/strict_inputs/strict_mapping_inputs.json" <<'PY'
import json,sys
sys.stdout.write("\n".join(json.load(open(sys.argv[1]))["mapping_trajectories"]))
PY
)
mapfile -t held_routes < <(python - "$dir/strict_inputs/strict_mapping_inputs.json" <<'PY'
import json,sys
sys.stdout.write("\n".join(json.load(open(sys.argv[1]))["held_query_trajectories"]))
PY
)
excluded=("${held_routes[@]}" seq3 seq5 seq13)
force=()
if [[ ${G23_FORCE:-0} == 1 ]]; then
  force=(--force)
fi

label_pids=()
for shard in 0 1; do
  labels=$dir/geometry_labels_shard$shard
  if [[ ! -e $labels/geometry_manifest.json || ${G23_FORCE:-0} == 1 ]]; then
    python "$repo/feature_extract/tools/vfm/build_2dgs_geometry_labels_from_contributors.py" \
      --contributors "$contributors" --gaussian_ply "$ply" \
      --output_dir "$labels" --shard_index "$shard" --shard_count 2 \
      > "$dir/geometry_labels_shard${shard}.log" 2>&1 &
    label_pids+=("$!")
  fi
done
for pid in "${label_pids[@]}"; do
  wait "$pid"
done

manifests=$dir/geometry_manifests
if [[ ! -e $manifests/summary.json || ${G23_FORCE:-0} == 1 ]]; then
  python "$repo/feature_extract/tools/vfm/build_goal_maplet_strict_geometry_fold_manifests.py" \
    --geometry_manifests \
      "$dir/geometry_labels_shard0/geometry_manifest.json" \
      "$dir/geometry_labels_shard1/geometry_manifest.json" \
    --protocol "$protocol" --fold_id "$fold" --output_dir "$manifests" \
    "${force[@]}" > "$dir/geometry_manifests.log" 2>&1
fi

geometry_head=$dir/geometry_head
if [[ ! -e $geometry_head/radio_highres_geometry_head.pt \
      || ! -e $geometry_head/geometry_head_summary.json \
      || ${G23_FORCE:-0} == 1 ]]; then
  mkdir -p "$geometry_head"
  eval_args=()
  if [[ $fold != final_alltrain ]]; then
    eval_args=(--eval_geometry_manifest "$manifests/eval.json")
    seed=$((2450 + ${fold#fold}))
  else
    seed=2450
  fi
  CUDA_VISIBLE_DEVICES=$gpu python \
    "$repo/feature_extract/tools/vfm/train_vfm_highres_geometry_head.py" \
    --train_geometry_manifest "$manifests/train.json" "${eval_args[@]}" \
    --output_dir "$geometry_head" --epochs 30 \
    --checkpoint_protocol fixed_epoch_no_selection --batch_size 16 \
    --hidden_channels 128 --architecture separate_decoders --task multitask \
    --lr 0.001 --weight_decay 0.0001 --normal_weight 1.0 \
    --confidence_weight 0.1 --cache_in_memory --amp --visualize 0 \
    --seed "$seed" --device cuda:0 > "$geometry_head/training.log" 2>&1
fi

if [[ ! -e $dir/canonical_field.npz || ! -e $dir/canonical_field.json || ${G23_FORCE:-0} == 1 ]]; then
  CUDA_VISIBLE_DEVICES=$gpu python \
    "$repo/feature_extract/tools/vfm/build_goal_maplet_canonical_field_from_contributors.py" \
    --contributors "$contributors" --physical_map "$physical" \
    --surface_mapper "$mapper" --feature_space retrieval_mapper \
    --exclude_trajectories "${excluded[@]}" \
    --output_field "$dir/canonical_field.npz" \
    --summary_json "$dir/canonical_field.json" --device cuda:0 "${force[@]}" \
    > "$dir/canonical_field.log" 2>&1
fi
if [[ ! -e $dir/field_feature_contract.json || ${G23_FORCE:-0} == 1 ]]; then
  python "$repo/feature_extract/tools/vfm/build_goal_maplet_feature_contract.py" \
    --canonical_field "$dir/canonical_field.npz" \
    --query_readout_type surface_maplet_mapper --query_readout "$mapper" \
    --render_protocol exact_clean_2dgs_identity_or_feature_token_grid_v1 \
    --output_json "$dir/field_feature_contract.json" "${force[@]}" \
    > "$dir/field_feature_contract.log" 2>&1
fi

if [[ ! -e $dir/physical_readout.pt || ! -e $dir/physical_readout.json || ${G23_FORCE:-0} == 1 ]]; then
  CUDA_VISIBLE_DEVICES=$gpu python \
    "$repo/feature_extract/tools/vfm/train_goal_maplet_physical_instance_readout.py" \
    --contributors "$contributors" --teacher_cache "$teachers" \
    --physical_map "$physical" --canonical_field "$dir/canonical_field.npz" \
    --surface_mapper "$mapper" --train_trajectories "${mapping_routes[@]}" \
    --selection_trajectories --validation_trajectories \
    --checkpoint_protocol fixed_step_no_selection \
    --steps 800 --batch_size 128 --learning_rate 0.0002 \
    --output_readout "$dir/physical_readout.pt" \
    --summary_json "$dir/physical_readout.json" --device cuda:0 "${force[@]}" \
    > "$dir/physical_readout.log" 2>&1
fi

if [[ ! -e $dir/typed_graph.npz || ! -e $dir/typed_graph.json || ${G23_FORCE:-0} == 1 ]]; then
  CUDA_VISIBLE_DEVICES=$gpu python \
    "$repo/feature_extract/tools/vfm/build_goal_maplet_typed_graph.py" \
    --contributors "$contributors" --physical_map "$physical" \
    --canonical_field "$dir/canonical_field.npz" \
    --physical_instance_readout "$dir/physical_readout.pt" \
    --exclude_trajectories "${excluded[@]}" \
    --output_graph "$dir/typed_graph.npz" --summary_json "$dir/typed_graph.json" \
    --device cuda:0 "${force[@]}" > "$dir/typed_graph.log" 2>&1
fi
if [[ ! -e $dir/mapping_view_graph.npz || ! -e $dir/mapping_view_graph.json || ${G23_FORCE:-0} == 1 ]]; then
  python "$repo/feature_extract/tools/vfm/build_goal_maplet_mapping_view_graph.py" \
    --contributors "$contributors" --physical_map "$physical" \
    --canonical_field "$dir/canonical_field.npz" \
    --exclude_trajectories "${excluded[@]}" \
    --output_graph "$dir/mapping_view_graph.npz" \
    --output_json "$dir/mapping_view_graph.json" "${force[@]}" \
    > "$dir/mapping_view_graph.log" 2>&1
fi
if [[ ! -e $dir/validity.npz || ! -e $dir/validity.json || ${G23_FORCE:-0} == 1 ]]; then
  CUDA_VISIBLE_DEVICES=$gpu python \
    "$repo/feature_extract/tools/vfm/calibrate_goal_maplet_validity.py" \
    --contributors "$contributors" --physical_map "$physical" \
    --canonical_field "$dir/canonical_field.npz" --surface_mapper "$mapper" \
    --physical_instance_readout "$dir/physical_readout.pt" \
    --pooling current_1x1_3x3_5x5_9x9 \
    --include_trajectories "${mapping_routes[@]}" \
    --output_calibration "$dir/validity.npz" \
    --summary_json "$dir/validity.json" --device cuda:0 "${force[@]}" \
    > "$dir/validity.log" 2>&1
fi
