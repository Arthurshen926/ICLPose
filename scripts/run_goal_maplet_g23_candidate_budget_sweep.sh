#!/usr/bin/env bash
set -euo pipefail

repo=/root/ICLPose
root=$repo/output/vfm/2dgs_surface/StMarysChurch/full_train/goal_maplet/g23_official_train_oof_v1
runner=$repo/scripts/run_goal_maplet_g23_oof_candidates_exact128.sh

run() {
  local tag=$1
  shift
  env G23_CANDIDATE_TAG="$tag" "$@" "$runner"
}

# Coordinate sweep: each row changes one compute-allocation decision from the
# exact128/16x4 reference.  It is intentionally not a combinatorial grid.
run exact64_anchor16x4 G23_EXACT_VERIFY_COUNT=64
run exact128_anchor16x4
run exact256_anchor16x4 G23_EXACT_VERIFY_COUNT=256
run exact128_quota2 G23_EXACT_KEEP_PER_ANCHOR=2
run exact128_quota8 G23_EXACT_KEEP_PER_ANCHOR=8
run exact128_protected8 G23_EXACT_PROTECTED_ANCHORS=8
run exact128_nms05m5deg G23_TRANSLATION_NMS_M=0.5 G23_ROTATION_NMS_DEG=5
run exact128_nms1m10deg G23_TRANSLATION_NMS_M=1.0 G23_ROTATION_NMS_DEG=10

python "$repo/feature_extract/tools/vfm/select_goal_maplet_candidate_sweep.py" \
  --protocol "$repo/configs/vfm/goal_maplet_stmarys_official_train_oof_v1.json" \
  --reports \
    exact64_anchor16x4="$root/candidates_exact64_anchor16x4_oof_evaluation.json" \
    exact128_anchor16x4="$root/candidates_exact128_anchor16x4_oof_evaluation.json" \
    exact256_anchor16x4="$root/candidates_exact256_anchor16x4_oof_evaluation.json" \
    exact128_quota2="$root/candidates_exact128_quota2_oof_evaluation.json" \
    exact128_quota8="$root/candidates_exact128_quota8_oof_evaluation.json" \
    exact128_protected8="$root/candidates_exact128_protected8_oof_evaluation.json" \
    exact128_nms05m5deg="$root/candidates_exact128_nms05m5deg_oof_evaluation.json" \
    exact128_nms1m10deg="$root/candidates_exact128_nms1m10deg_oof_evaluation.json" \
  --output_json "$root/candidate_budget_sweep_selection.json"
