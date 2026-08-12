#!/usr/bin/env bash
set -euo pipefail

# Final fitting is deliberately a verification-only boundary.  The strict
# all-train map is built by the same route-clean pipeline as the OOF maps; this
# entry point refuses every historical full-train geometry artifact.
repo=/root/ICLPose
base=$repo/output/vfm/2dgs_surface/StMarysChurch/full_train
strict_root=$base/goal_maplet/g23_strict_oof_geometry_v1
frozen=$strict_root/frozen_configuration.json
protocol=$repo/configs/vfm/goal_maplet_stmarys_official_train_oof_v1.json
final=$strict_root/final_alltrain
export PYTHONPATH=$repo

if [[ ! -e $frozen ]]; then
  echo "refusing final fit before train-only OOF configuration freeze: $frozen" >&2
  exit 2
fi
python "$repo/feature_extract/tools/vfm/freeze_goal_maplet_g23_configuration.py" verify \
  --frozen "$frozen" --protocol "$protocol"

# The strict pipeline does not even materialize official-test evaluation
# packages before the train-only configuration is frozen.  This step contains
# no metric computation and cannot modify the frozen method.
env G23_INCLUDE_OFFICIAL_TEST_INPUTS=1 \
  "$repo/scripts/run_goal_maplet_g23_strict_contributors_fold.sh" final_alltrain

required=(
  "$final/strict_map_audit.json"
  "$final/contributors_official_train_audit.json"
  "$final/contributors_official_test_audit.json"
  "$final/geometry_head/radio_highres_geometry_head.pt"
  "$final/geometry_head/geometry_head_summary.json"
  "$final/surface_mapper.pt"
  "$final/surface_mapper.json"
  "$final/physical_map.npz"
  "$final/physical_map_audit.json"
  "$final/canonical_field.npz"
  "$final/canonical_field.json"
  "$final/field_feature_contract.json"
  "$final/physical_readout.pt"
  "$final/physical_readout.json"
  "$final/typed_graph.npz"
  "$final/typed_graph.json"
  "$final/mapping_view_graph.npz"
  "$final/mapping_view_graph.json"
  "$final/validity.npz"
  "$final/validity.json"
)
for path in "${required[@]}"; do
  if [[ ! -e $path ]]; then
    echo "strict final all-train artifact is missing: $path" >&2
    exit 2
  fi
done

force=()
if [[ ${G23_FORCE:-0} == 1 ]]; then
  force=(--force)
fi
python "$repo/feature_extract/tools/vfm/audit_goal_maplet_final_alltrain_fit.py" \
  --protocol "$protocol" --frozen "$frozen" --final_dir "$final" \
  --output_json "$final/final_alltrain_fit_audit.json" \
  "${force[@]}"
