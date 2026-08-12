#!/usr/bin/env bash
set -euo pipefail

repo=/root/ICLPose
base=$repo/output/vfm/2dgs_surface/StMarysChurch/full_train
goal=$base/goal_maplet
root=${G23_OOF_ROOT:-$goal/g23_official_train_oof_v1}
screening_root=$goal/g23_official_train_oof_v1
default_contributors=$base/mainline_v6/contributors_alltrain_clean
default_physical=$goal/physical_map_v4.npz
tag=${G23_CANDIDATE_TAG:-exact128_anchor16x4}
protocol=$repo/configs/vfm/goal_maplet_stmarys_official_train_oof_v1.json
acquisition_strata=${G23_ACQUISITION_STRATA:-$screening_root/acquisition_extrapolation_strata.json}
export PYTHONPATH=$repo

run_fold() {
  local gpu=$1
  local fold=$2
  local suffix=$3
  local dir=$root/$fold
  local candidates=$dir/candidates_${tag}${suffix}.json
  local output=$dir/refinement_${tag}_score_topk32_radius01${suffix}.json
  local contributors=$default_contributors
  local physical=$default_physical
  if [[ ${G23_STRICT_GEOMETRY:-0} == 1 ]]; then
    contributors=$dir/contributors_official_train
    physical=$dir/physical_map.npz
  fi
  if [[ ! -e $candidates ]]; then
    echo "missing OOF candidates for $fold: $candidates" >&2
    exit 2
  fi
  if [[ -e $output && ${G23_FORCE:-0} != 1 ]]; then
    return
  fi
  local force=()
  if [[ ${G23_FORCE:-0} == 1 ]]; then
    force=(--force)
  fi
  CUDA_VISIBLE_DEVICES=$gpu python \
    "$repo/feature_extract/tools/vfm/refine_goal_maplet_pose_modes_with_primitive_vfm.py" \
    --contributors "$contributors" --candidate_report "$candidates" \
    --physical_map "$physical" --canonical_field "$dir/canonical_field.npz" \
    --surface_mapper "$dir/surface_mapper.pt" --output_json "$output" \
    --mode_name actual_parent_actual_child --refine_topk 32 \
    --refinement_candidate_policy score_topk --refinement_score_margin -1 \
    --minimum_refinement_candidates 1 \
    --translation_steps_m 0.60,0.40,0.25,0.12 \
    --rotation_steps_deg 5,3,2,1 --iterations_per_scale 2 \
    --minimum_score_improvement 1e-6 \
    --maximum_splat_radius_tokens 0 --validation_splat_radius_tokens 1 \
    --require_cross_splat_winner_consistency --device cuda:0 "${force[@]}" \
    > "$dir/refinement_${tag}_score_topk32_radius01${suffix}.log" 2>&1
}

gpu0() {
  run_fold 0 fold0 ""
  run_fold 0 fold3 ""
  run_fold 0 fold4 _shard0of2
}
gpu1() {
  run_fold 1 fold1 ""
  run_fold 1 fold2 ""
  run_fold 1 fold4 _shard1of2
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

six_axis_basin=$root/primitive_refinement_six_axis_basin_strict_oof.json
if [[ ${G23_STRICT_GEOMETRY:-0} == 1 && ! -e $six_axis_basin ]]; then
  "$repo/scripts/run_goal_maplet_g23_six_axis_basin.sh"
fi

reports=(
  "$root/fold0/refinement_${tag}_score_topk32_radius01.json"
  "$root/fold1/refinement_${tag}_score_topk32_radius01.json"
  "$root/fold2/refinement_${tag}_score_topk32_radius01.json"
  "$root/fold3/refinement_${tag}_score_topk32_radius01.json"
  "$root/fold4/refinement_${tag}_score_topk32_radius01_shard0of2.json"
  "$root/fold4/refinement_${tag}_score_topk32_radius01_shard1of2.json"
)
force=()
if [[ ${G23_FORCE:-0} == 1 ]]; then
  force=(--force)
fi
curve=$root/refinement_${tag}_score_topk32_radius01_k_curve.json
all_reports_ready=1
for report in "${reports[@]}"; do
  if [[ ! -e $report ]]; then
    all_reports_ready=0
  fi
done
if [[ $all_reports_ready == 1 && ( ! -e $curve || ${G23_FORCE:-0} == 1 ) ]]; then
  python "$repo/feature_extract/tools/vfm/evaluate_goal_maplet_refinement_k_curve.py" \
    --refinement_reports "${reports[@]}" --k_values 1,2,4,8,16,32 \
    --output_json "$curve" "${force[@]}"
fi
policy=$root/adaptive_refinement_${tag}_oof_selection.json
nested_selected=$root/nested_selected_refinement_${tag}_oof_report.json
final_policy_selected=$root/final_policy_selected_refinement_${tag}_training_report.json
if [[ $all_reports_ready == 1 \
      && ( ! -e $policy || ! -e $nested_selected \
           || ! -e $final_policy_selected || ${G23_FORCE:-0} == 1 ) ]]; then
  python "$repo/feature_extract/tools/vfm/evaluate_goal_maplet_adaptive_refinement_oof.py" \
    --refinement_reports "${reports[@]}" --protocol_json "$protocol" \
    --score_topk_budgets 1,2,4,8,16,32 \
    --basin_cover_budgets 8,16,32 \
    --basin_cover_score_margins 0.005,0.01,0.02,0.05 \
    --output_json "$policy" --selected_report_json "$nested_selected" \
    --final_policy_report_json "$final_policy_selected" "${force[@]}"
fi
calibration=$root/postselection_success_calibration_${tag}_oof.json
if [[ -e $nested_selected && -e $final_policy_selected \
      && ( ! -e $calibration || ${G23_FORCE:-0} == 1 ) ]]; then
  python "$repo/feature_extract/tools/vfm/calibrate_goal_maplet_postselection_success_oof.py" \
    --refinement_reports "$nested_selected" \
    --final_fit_refinement_report "$final_policy_selected" \
    --require_nested_policy_evaluation --protocol_json "$protocol" \
    --regularization_candidates 0.01,0.03,0.1,0.3,1.0 \
    --minimum_success_targets 0.8,0.9,0.95 \
    --output_json "$calibration" "${force[@]}"
fi

failure_taxonomy=$root/failure_taxonomy_${tag}_nested_oof.json
if [[ -e $nested_selected \
      && -e $root/fold0/candidates_${tag}.json \
      && -e $root/fold1/candidates_${tag}.json \
      && -e $root/fold2/candidates_${tag}.json \
      && -e $root/fold3/candidates_${tag}.json \
      && -e $root/fold4/candidates_${tag}.json \
      && ( ! -e $failure_taxonomy || ${G23_FORCE:-0} == 1 ) ]]; then
  python "$repo/feature_extract/tools/vfm/evaluate_goal_maplet_failure_taxonomy.py" \
    --protocol "$protocol" --candidate_reports \
      "fold0=$root/fold0/candidates_${tag}.json" \
      "fold1=$root/fold1/candidates_${tag}.json" \
      "fold2=$root/fold2/candidates_${tag}.json" \
      "fold3=$root/fold3/candidates_${tag}.json" \
      "fold4=$root/fold4/candidates_${tag}.json" \
    --final_postselection_report "$nested_selected" \
    --output_json "$failure_taxonomy" "${force[@]}"
fi

extrapolation_evaluation=$root/extrapolation_evaluation_${tag}_nested_oof.json
if [[ -e $nested_selected && -e $acquisition_strata \
      && ( ! -e $extrapolation_evaluation || ${G23_FORCE:-0} == 1 ) ]]; then
  python "$repo/feature_extract/tools/vfm/evaluate_goal_maplet_extrapolation_strata.py" \
    --strata "$acquisition_strata" \
    --postselection_report "$nested_selected" \
    --output_json "$extrapolation_evaluation" "${force[@]}"
fi

map_audit=$root/oof_map_leakage_audit.json
candidate_evaluation=$root/candidates_${tag}_oof_evaluation.json
frozen=$root/frozen_configuration.json
if [[ -e $map_audit && -e $candidate_evaluation && -e $policy \
      && -e $calibration && -e $failure_taxonomy \
      && -e $extrapolation_evaluation && -e $six_axis_basin \
      && ( ! -e $frozen || ${G23_FORCE:-0} == 1 ) ]]; then
  python "$repo/feature_extract/tools/vfm/freeze_goal_maplet_g23_configuration.py" build \
    --protocol "$protocol" --map_audit "$map_audit" \
    --candidate_evaluation "$candidate_evaluation" \
    --adaptive_policy "$policy" --calibrator "$calibration" \
    --failure_taxonomy "$failure_taxonomy" \
    --extrapolation_evaluation "$extrapolation_evaluation" \
    --six_axis_basin "$six_axis_basin" \
    --output_json "$frozen" "${force[@]}"
fi
