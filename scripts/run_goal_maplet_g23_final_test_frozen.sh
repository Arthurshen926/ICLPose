#!/usr/bin/env bash
set -euo pipefail

repo=/root/ICLPose
base=$repo/output/vfm/2dgs_surface/StMarysChurch/full_train
root=$base/goal_maplet/g23_strict_oof_geometry_v1
final=$root/final_alltrain
frozen=$root/frozen_configuration.json
protocol=$repo/configs/vfm/goal_maplet_stmarys_official_train_oof_v1.json
contributors=$final/contributors_official_test
physical=$final/physical_map.npz
output=$root/official_test_frozen
export PYTHONPATH=$repo

if [[ ! -e $final/final_alltrain_fit_audit.json || ! -e $frozen ]]; then
  echo "refusing official test before strict final-fit audit and freeze" >&2
  exit 2
fi
python "$repo/feature_extract/tools/vfm/freeze_goal_maplet_g23_configuration.py" verify \
  --frozen "$frozen" --protocol "$protocol"
calibrator=$(python - "$frozen" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["success_calibration"]["path"])
PY
)
mapfile -t frozen_candidate_args < <(
  python "$repo/feature_extract/tools/vfm/render_goal_maplet_frozen_candidate_args.py" \
    --frozen "$frozen"
)
mapfile -t frozen_refinement_args < <(
  python "$repo/feature_extract/tools/vfm/render_goal_maplet_frozen_refinement_args.py" \
    --frozen "$frozen"
)
mkdir -p "$output"
force=()
if [[ ${G23_FORCE:-0} == 1 ]]; then
  force=(--force)
fi

run_candidate_shard() {
  local repeat=$1
  local gpu=$2
  local shard=$3
  local destination=$output/repeat${repeat}_candidates_shard${shard}.json
  if [[ -e $destination && ${G23_FORCE:-0} != 1 ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES=$gpu python \
    "$repo/feature_extract/tools/vfm/evaluate_goal_maplet_pose_modes.py" \
    --contributors "$contributors" --physical_map "$physical" \
    --canonical_field "$final/canonical_field.npz" \
    --surface_mapper "$final/surface_mapper.pt" \
    --field_feature_contract "$final/field_feature_contract.json" \
    --validity_calibration "$final/validity.npz" \
    --typed_graph "$final/typed_graph.npz" \
    --physical_instance_readout "$final/physical_readout.pt" \
    --geometry_head "$final/geometry_head/radio_highres_geometry_head.pt" \
    --mapping_view_graph "$final/mapping_view_graph.npz" \
    --output_json "$destination" "${frozen_candidate_args[@]}" \
    --include_trajectories seq3 seq5 seq13 \
    --shard_index "$shard" --shard_count 2 --checkpoint_every 10 \
    --device cuda:0 --quiet_rows "${force[@]}" \
    > "$output/repeat${repeat}_candidates_shard${shard}.log" 2>&1
}

run_refinement_shard() {
  local repeat=$1
  local gpu=$2
  local shard=$3
  local candidates=$output/repeat${repeat}_candidates_shard${shard}.json
  local destination=$output/repeat${repeat}_refinement_shard${shard}.json
  if [[ -e $destination && ${G23_FORCE:-0} != 1 ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES=$gpu python \
    "$repo/feature_extract/tools/vfm/refine_goal_maplet_pose_modes_with_primitive_vfm.py" \
    --contributors "$contributors" --candidate_report "$candidates" \
    --physical_map "$physical" --canonical_field "$final/canonical_field.npz" \
    --surface_mapper "$final/surface_mapper.pt" --output_json "$destination" \
    "${frozen_refinement_args[@]}" --device cuda:0 "${force[@]}" \
    > "$output/repeat${repeat}_refinement_shard${shard}.log" 2>&1
}

for repeat in 0 1 2; do
  run_candidate_shard "$repeat" 0 0 &
  pid0=$!
  run_candidate_shard "$repeat" 1 1 &
  pid1=$!
  wait "$pid0"
  wait "$pid1"
  candidates=(
    "$output/repeat${repeat}_candidates_shard0.json"
    "$output/repeat${repeat}_candidates_shard1.json"
  )
  python "$repo/feature_extract/tools/vfm/verify_goal_maplet_candidate_against_frozen.py" \
    --frozen "$frozen" --candidate_reports "${candidates[@]}"

  run_refinement_shard "$repeat" 0 0 &
  pid0=$!
  run_refinement_shard "$repeat" 1 1 &
  pid1=$!
  wait "$pid0"
  wait "$pid1"
  refinements=(
    "$output/repeat${repeat}_refinement_shard0.json"
    "$output/repeat${repeat}_refinement_shard1.json"
  )
  selection=$output/repeat${repeat}_selection.json
  if [[ ! -e $selection || ${G23_FORCE:-0} == 1 ]]; then
    python "$repo/feature_extract/tools/vfm/apply_goal_maplet_adaptive_refinement_policy.py" \
      --frozen "$frozen" --refinement_reports "${refinements[@]}" \
      --output_json "$selection" "${force[@]}"
  fi
  calibrated=$output/repeat${repeat}_calibrated.json
  if [[ ! -e $calibrated || ${G23_FORCE:-0} == 1 ]]; then
    python "$repo/feature_extract/tools/vfm/apply_goal_maplet_postselection_success.py" \
      --calibrator "$calibrator" --selection_report "$selection" \
      --output_json "$calibrated" "${force[@]}"
  fi
done

python "$repo/feature_extract/tools/vfm/evaluate_goal_maplet_final_test_repeats.py" \
  --protocol "$protocol" \
  --selection_reports "$output/repeat0_selection.json" \
    "$output/repeat1_selection.json" "$output/repeat2_selection.json" \
  --calibrated_reports "$output/repeat0_calibrated.json" \
    "$output/repeat1_calibrated.json" "$output/repeat2_calibrated.json" \
  --output_json "$output/final_test_three_repeat_evaluation.json" "${force[@]}"
