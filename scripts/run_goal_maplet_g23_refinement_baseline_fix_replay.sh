#!/usr/bin/env bash
set -euo pipefail

repo=/root/ICLPose
base=$repo/output/vfm/2dgs_surface/StMarysChurch/full_train
goal=$base/goal_maplet
contributors=$base/mainline_v6/contributors_setcover128_clean
physical=$goal/physical_map_v4.npz
crossfit=$goal/map_crossfit_g20_1
g21=$goal/g21_mapping_view
output=$goal/g23_refinement_baseline_fix_replay
selection=$goal/g20_6_joint/joint_loto_common_transfer.json
null_calibration=$goal/g20_8_sparse_vfm/calib_old_phase_primitive_refinement.json
export PYTHONPATH=$repo
mkdir -p "$output"

run_fold() {
  local gpu=$1
  local held=$2
  local fold=$3
  local baseline_report=$4
  local expert_report=$5
  local destination=$output/${held}.json
  if [[ -e $destination && ${G23_FORCE:-0} != 1 ]]; then
    return
  fi
  local force=()
  if [[ ${G23_FORCE:-0} == 1 ]]; then
    force=(--force)
  fi
  CUDA_VISIBLE_DEVICES=$gpu python \
    "$repo/feature_extract/tools/vfm/refine_goal_maplet_pose_modes_with_primitive_vfm.py" \
    --contributors "$contributors" \
    --candidate_report "$baseline_report" "$expert_report" \
    --baseline_selection_report "$selection" \
    --physical_map "$physical" --canonical_field "$fold/canonical_field.npz" \
    --surface_mapper "$fold/surface_mapper.pt" --output_json "$destination" \
    --refine_topk 3 --refinement_candidate_policy score_topk \
    --passthrough_without_additional_expert \
    --baseline_mode_count 1 --additional_expert_mode_count 2 \
    --maximum_splat_radius_tokens 0 --validation_splat_radius_tokens 1 \
    --baseline_null_calibration_report "$null_calibration" \
    --baseline_null_sigma 3 --require_cross_splat_winner_consistency \
    --device cuda:0 "${force[@]}"
}

run_fold 0 seq12 "$crossfit/hold_seq12" \
  "$crossfit/hold_seq12/directional_report_relative_geometry_dense_audit_g20_5_k.json" \
  "$g21/g21_gated_seq12_exact128.json" &
pid0=$!
run_fold 1 seq14 "$crossfit/hold_seq14" \
  "$crossfit/hold_seq14/directional_report_relative_geometry_dense_audit_g20_5_k.json" \
  "$g21/g21_gated_seq14_exact128.json" &
pid1=$!
wait "$pid0"
wait "$pid1"

evaluation=$output/selective_gate_evaluation.json
evaluation_force=()
if [[ ${G23_FORCE:-0} == 1 ]]; then
  evaluation_force=(--force)
fi
if [[ ! -e $evaluation || ${G23_FORCE:-0} == 1 ]]; then
  python -m feature_extract.tools.vfm.evaluate_goal_maplet_primitive_refinement_gate \
    --gate "$goal/g20_8_sparse_vfm/primitive_refinement_gate.json" \
    --joint_likelihood "$goal/g20_6_joint/joint_policy_common_transfer.json" \
    --baseline_evaluation "$selection" \
    --candidate_reports \
      "$crossfit/hold_seq12/directional_report_relative_geometry_dense_audit_g20_5_k.json" \
      "$crossfit/hold_seq14/directional_report_relative_geometry_dense_audit_g20_5_k.json" \
    --refinement_reports "$output/seq12.json" "$output/seq14.json" \
    --output_json "$evaluation" "${evaluation_force[@]}"
fi
