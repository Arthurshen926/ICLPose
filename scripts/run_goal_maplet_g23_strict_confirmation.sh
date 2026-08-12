#!/usr/bin/env bash
set -euo pipefail

repo=/root/ICLPose
root=$repo/output/vfm/2dgs_surface/StMarysChurch/full_train/goal_maplet/g23_strict_oof_geometry_v1
screening_root=$repo/output/vfm/2dgs_surface/StMarysChurch/full_train/goal_maplet/g23_official_train_oof_v1
protocol=$repo/configs/vfm/goal_maplet_stmarys_official_train_oof_v1.json
selection=${G23_SCREENING_SELECTION:-$screening_root/candidate_budget_sweep_selection.json}
export PYTHONPATH=$repo

if [[ ! -e $selection ]]; then
  echo "missing train-OOF candidate selection: $selection" >&2
  exit 2
fi
mapfile -t selected_candidate_env < <(
  python "$repo/feature_extract/tools/vfm/render_goal_maplet_selected_candidate_env.py" \
    --selection "$selection"
)

# Geometry/map construction keeps one independent map pipeline on each GPU.
"$repo/scripts/run_goal_maplet_g23_strict_maps.sh"

# Contributor rasterization for one geometry occupies both GPUs.  Run folds
# sequentially so a cache never mixes primitive identity from two PLYs.
for fold in fold0 fold1 fold2 fold3 fold4 final_alltrain; do
  "$repo/scripts/run_goal_maplet_g23_strict_contributors_fold.sh" "$fold"
done

# Geometry heads and map feature heads return to one fold pipeline per GPU.
"$repo/scripts/run_goal_maplet_g23_strict_features.sh"

python "$repo/feature_extract/tools/vfm/audit_goal_maplet_official_oof_maps.py" \
  --protocol "$protocol" --fold_root "$root" \
  --output_json "$root/oof_map_leakage_audit.json"

env G23_OOF_ROOT=$root G23_STRICT_GEOMETRY=1 \
  G23_CANDIDATE_SELECTION=$selection "${selected_candidate_env[@]}" \
  "$repo/scripts/run_goal_maplet_g23_oof_candidates_exact128.sh"
env G23_OOF_ROOT=$root G23_STRICT_GEOMETRY=1 "${selected_candidate_env[@]}" \
  "$repo/scripts/run_goal_maplet_g23_oof_refinement_and_calibration.sh"

"$repo/scripts/run_goal_maplet_g23_final_alltrain_fit.sh"
