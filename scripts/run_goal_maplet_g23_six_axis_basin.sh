#!/usr/bin/env bash
set -euo pipefail

repo=/root/ICLPose
base=$repo/output/vfm/2dgs_surface/StMarysChurch/full_train
goal=$base/goal_maplet
root=${G23_OOF_ROOT:-$goal/g23_official_train_oof_v1}
default_contributors=$base/mainline_v6/contributors_alltrain_clean
default_physical=$goal/physical_map_v4.npz
per_route=${G23_BASIN_QUERIES_PER_ROUTE:-4}
protocol=$repo/configs/vfm/goal_maplet_stmarys_official_train_oof_v1.json
export PYTHONPATH=$repo

run_fold() {
  local gpu=$1
  local fold=$2
  shift 2
  local held=("$@")
  local dir=$root/$fold
  local contributors=$default_contributors
  local physical=$default_physical
  local tier=screening
  if [[ ${G23_STRICT_GEOMETRY:-0} == 1 ]]; then
    contributors=$dir/contributors_official_train
    physical=$dir/physical_map.npz
    tier=strict
    if [[ ! -e $dir/strict_map_audit.json \
          || ! -e $dir/contributors_official_train_audit.json ]]; then
      echo "strict basin inputs are incomplete for $fold" >&2
      exit 2
    fi
  fi
  local output=$dir/primitive_refinement_six_axis_basin_${tier}.json
  if [[ -e $output && ${G23_FORCE:-0} != 1 ]]; then
    return
  fi
  local force=()
  if [[ ${G23_FORCE:-0} == 1 ]]; then
    force=(--force)
  fi
  CUDA_VISIBLE_DEVICES=$gpu python \
    "$repo/feature_extract/tools/vfm/evaluate_goal_maplet_primitive_refinement_six_axis_basin.py" \
    --contributors "$contributors" --physical_map "$physical" \
    --canonical_field "$dir/canonical_field.npz" \
    --surface_mapper "$dir/surface_mapper.pt" \
    --include_trajectories "${held[@]}" \
    --translation_levels_m 0.1,0.25,0.5,1,2 \
    --rotation_levels_deg 2,5,10,20 \
    --translation_steps_m 0.60,0.40,0.25,0.12 \
    --rotation_steps_deg 5,3,2,1 --iterations_per_scale 2 \
    --minimum_score_improvement 1e-6 --maximum_splat_radius_tokens 0 \
    --maximum_queries_per_trajectory "$per_route" \
    --output_json "$output" --device cuda:0 "${force[@]}" \
    > "$dir/primitive_refinement_six_axis_basin_${tier}.log" 2>&1
}

gpu0() {
  run_fold 0 fold0 seq2
  run_fold 0 fold2 seq1 seq8 seq14
  run_fold 0 fold4 seq6 seq9 seq10 seq11
}
gpu1() {
  run_fold 1 fold1 seq4
  run_fold 1 fold3 seq7 seq12
}

gpu0 &
pid0=$!
gpu1 &
pid1=$!
wait "$pid0"
wait "$pid1"

if [[ ${G23_STRICT_GEOMETRY:-0} == 1 ]]; then
  aggregate=$root/primitive_refinement_six_axis_basin_strict_oof.json
  force=()
  if [[ ${G23_FORCE:-0} == 1 ]]; then
    force=(--force)
  fi
  if [[ ! -e $aggregate || ${G23_FORCE:-0} == 1 ]]; then
    python \
      "$repo/feature_extract/tools/vfm/aggregate_goal_maplet_six_axis_basin_oof.py" \
      --protocol "$protocol" --expected_queries_per_trajectory "$per_route" \
      --reports \
        "fold0=$root/fold0/primitive_refinement_six_axis_basin_strict.json" \
        "fold1=$root/fold1/primitive_refinement_six_axis_basin_strict.json" \
        "fold2=$root/fold2/primitive_refinement_six_axis_basin_strict.json" \
        "fold3=$root/fold3/primitive_refinement_six_axis_basin_strict.json" \
        "fold4=$root/fold4/primitive_refinement_six_axis_basin_strict.json" \
      --output_json "$aggregate" "${force[@]}"
  fi
fi
