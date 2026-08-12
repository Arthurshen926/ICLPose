#!/usr/bin/env bash
set -euo pipefail

repo=/root/ICLPose
base=$repo/output/vfm/2dgs_surface/StMarysChurch/full_train
goal=$base/goal_maplet
root=$goal/g23_official_train_oof_v1
contributors=$base/mainline_v6/contributors_alltrain_clean
teachers=$base/mainline_v6/v8_offline_radio_teacher_regions_alltrain
physical=$goal/physical_map_v4.npz
export PYTHONPATH=$repo

if [[ ! -e $base/mainline_v6/v8_offline_radio_teacher_regions_alltrain_audit.json ]]; then
  echo "missing audited all-train teacher cache; run run_goal_maplet_g23_alltrain_teachers.sh first" >&2
  exit 2
fi

run_fold() {
  local gpu=$1
  local fold=$2
  shift 2
  local held_count=$1
  shift
  local held=("${@:1:$held_count}")
  shift "$held_count"
  local mapping=("$@")
  local dir=$root/$fold
  local force=()
  if [[ ${G23_FORCE:-0} == 1 ]]; then
    force=(--force)
  fi

  if [[ ! -e $dir/canonical_field.npz || ! -e $dir/canonical_field.json || ${G23_FORCE:-0} == 1 ]]; then
    CUDA_VISIBLE_DEVICES=$gpu python \
      "$repo/feature_extract/tools/vfm/build_goal_maplet_canonical_field_from_contributors.py" \
      --contributors "$contributors" --physical_map "$physical" \
      --surface_mapper "$dir/surface_mapper.pt" --feature_space retrieval_mapper \
      --exclude_trajectories "${held[@]}" seq3 seq5 seq13 \
      --output_field "$dir/canonical_field.npz" \
      --summary_json "$dir/canonical_field.json" --device cuda:0 "${force[@]}" \
      > "$dir/canonical_field.log" 2>&1
  fi
  if [[ ! -e $dir/field_feature_contract.json || ${G23_FORCE:-0} == 1 ]]; then
    python "$repo/feature_extract/tools/vfm/build_goal_maplet_feature_contract.py" \
      --canonical_field "$dir/canonical_field.npz" \
      --query_readout_type surface_maplet_mapper \
      --query_readout "$dir/surface_mapper.pt" \
      --render_protocol exact_clean_2dgs_identity_or_feature_token_grid_v1 \
      --output_json "$dir/field_feature_contract.json" "${force[@]}" \
      > "$dir/field_feature_contract.log" 2>&1
  fi

  if [[ ! -e $dir/physical_readout.pt || ! -e $dir/physical_readout.json || ${G23_FORCE:-0} == 1 ]]; then
    CUDA_VISIBLE_DEVICES=$gpu python \
      "$repo/feature_extract/tools/vfm/train_goal_maplet_physical_instance_readout.py" \
      --contributors "$contributors" --teacher_cache "$teachers" \
      --physical_map "$physical" --canonical_field "$dir/canonical_field.npz" \
      --surface_mapper "$dir/surface_mapper.pt" \
      --train_trajectories "${mapping[@]}" \
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
      --exclude_trajectories "${held[@]}" seq3 seq5 seq13 \
      --output_graph "$dir/typed_graph.npz" --summary_json "$dir/typed_graph.json" \
      --device cuda:0 "${force[@]}" > "$dir/typed_graph.log" 2>&1
  fi

  if [[ ! -e $dir/mapping_view_graph.npz || ! -e $dir/mapping_view_graph.json || ${G23_FORCE:-0} == 1 ]]; then
    python "$repo/feature_extract/tools/vfm/build_goal_maplet_mapping_view_graph.py" \
      --contributors "$contributors" --physical_map "$physical" \
      --canonical_field "$dir/canonical_field.npz" \
      --exclude_trajectories "${held[@]}" seq3 seq5 seq13 \
      --output_graph "$dir/mapping_view_graph.npz" \
      --output_json "$dir/mapping_view_graph.json" "${force[@]}" \
      > "$dir/mapping_view_graph.log" 2>&1
  fi

  if [[ ! -e $dir/validity.npz || ! -e $dir/validity.json || ${G23_FORCE:-0} == 1 ]]; then
    CUDA_VISIBLE_DEVICES=$gpu python \
      "$repo/feature_extract/tools/vfm/calibrate_goal_maplet_validity.py" \
      --contributors "$contributors" --physical_map "$physical" \
      --canonical_field "$dir/canonical_field.npz" \
      --surface_mapper "$dir/surface_mapper.pt" \
      --physical_instance_readout "$dir/physical_readout.pt" \
      --pooling current_1x1_3x3_5x5_9x9 \
      --include_trajectories "${mapping[@]}" \
      --output_calibration "$dir/validity.npz" \
      --summary_json "$dir/validity.json" --device cuda:0 "${force[@]}" \
      > "$dir/validity.log" 2>&1
  fi
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

pids=()
if [[ ${G23_SKIP_GPU0:-0} != 1 ]]; then
  gpu0 &
  pids+=("$!")
fi
if [[ ${G23_SKIP_GPU1:-0} != 1 ]]; then
  gpu1 &
  pids+=("$!")
fi
for pid in "${pids[@]}"; do
  wait "$pid"
done

all_maps_ready=1
for fold in fold0 fold1 fold2 fold3 fold4; do
  dir=$root/$fold
  for artifact in \
    surface_mapper.json geometry_head/geometry_head_summary.json \
    canonical_field.json physical_readout.json typed_graph.json \
    mapping_view_graph.json validity.json; do
    if [[ ! -e $dir/$artifact ]]; then
      all_maps_ready=0
    fi
  done
done
audit=$root/oof_map_leakage_audit.json
if [[ $all_maps_ready == 1 && ( ! -e $audit || ${G23_FORCE:-0} == 1 ) ]]; then
  audit_force=()
  if [[ ${G23_FORCE:-0} == 1 ]]; then
    audit_force=(--force)
  fi
  python "$repo/feature_extract/tools/vfm/audit_goal_maplet_official_oof_maps.py" \
    --protocol "$repo/configs/vfm/goal_maplet_stmarys_official_train_oof_v1.json" \
    --fold_root "$root" --output_json "$audit" "${audit_force[@]}"
fi
