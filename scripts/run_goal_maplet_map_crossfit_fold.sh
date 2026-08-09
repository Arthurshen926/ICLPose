#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 FOLD_DIR HELD_TRAJECTORY VALIDATION_TRAJECTORY" >&2
  exit 2
fi

fold_dir=$1
held=$2
validation=$3
repo=/root/ICLPose
base=$repo/output/vfm/2dgs_surface/StMarysChurch/full_train
goal=$base/goal_maplet
contributors=$base/mainline_v6/contributors_setcover128_clean
teachers=$base/mainline_v6/v8_offline_radio_teacher_regions_setcover128
physical=$goal/physical_map_v4.npz
mapper=$fold_dir/surface_mapper.pt
device=cuda:0

export PYTHONPATH=$repo

python "$repo/feature_extract/tools/vfm/build_goal_maplet_canonical_field_from_contributors.py" \
  --contributors "$contributors" \
  --physical_map "$physical" \
  --surface_mapper "$mapper" \
  --feature_space retrieval_mapper \
  --exclude_trajectories "$held" seq3 seq5 seq13 \
  --output_field "$fold_dir/canonical_field.npz" \
  --summary_json "$fold_dir/canonical_field.json" \
  --device "$device" --force

python "$repo/feature_extract/tools/vfm/build_goal_maplet_feature_contract.py" \
  --canonical_field "$fold_dir/canonical_field.npz" \
  --query_readout_type surface_maplet_mapper \
  --query_readout "$mapper" \
  --render_protocol exact_clean_2dgs_identity_or_feature_token_grid_v1 \
  --output_json "$fold_dir/field_feature_contract.json" --force

python "$repo/feature_extract/tools/vfm/train_goal_maplet_physical_instance_readout.py" \
  --contributors "$contributors" \
  --teacher_cache "$teachers" \
  --physical_map "$physical" \
  --canonical_field "$fold_dir/canonical_field.npz" \
  --surface_mapper "$mapper" \
  --train_trajectories seq1 seq2 seq4 seq6 seq7 seq8 seq11 \
  --selection_trajectories seq9 seq10 \
  --validation_trajectories "$validation" \
  --steps 800 --batch_size 128 --learning_rate 0.0002 \
  --output_readout "$fold_dir/physical_readout.pt" \
  --summary_json "$fold_dir/physical_readout.json" \
  --device "$device" --force

python "$repo/feature_extract/tools/vfm/build_goal_maplet_typed_graph.py" \
  --contributors "$contributors" \
  --physical_map "$physical" \
  --canonical_field "$fold_dir/canonical_field.npz" \
  --physical_instance_readout "$fold_dir/physical_readout.pt" \
  --exclude_trajectories "$held" seq3 seq5 seq13 \
  --output_graph "$fold_dir/typed_graph.npz" \
  --summary_json "$fold_dir/typed_graph.json" \
  --device "$device" --force

python "$repo/feature_extract/tools/vfm/calibrate_goal_maplet_validity.py" \
  --contributors "$contributors" \
  --physical_map "$physical" \
  --canonical_field "$fold_dir/canonical_field.npz" \
  --surface_mapper "$mapper" \
  --physical_instance_readout "$fold_dir/physical_readout.pt" \
  --pooling current_1x1_3x3_5x5_9x9 \
  --include_trajectories seq10 \
  --output_calibration "$fold_dir/validity.npz" \
  --summary_json "$fold_dir/validity.json" \
  --device "$device" --force

python "$repo/feature_extract/tools/vfm/evaluate_goal_maplet_pose_modes.py" \
  --contributors "$contributors" \
  --physical_map "$physical" \
  --canonical_field "$fold_dir/canonical_field.npz" \
  --surface_mapper "$mapper" \
  --field_feature_contract "$fold_dir/field_feature_contract.json" \
  --validity_calibration "$fold_dir/validity.npz" \
  --typed_graph "$fold_dir/typed_graph.npz" \
  --physical_instance_readout "$fold_dir/physical_readout.pt" \
  --parent_mode actual --child_mode actual \
  --maximum_modes 32 --proposal_method graph \
  --render_identity_rerank --identity_render_mode child_splat \
  --include_trajectories "$held" \
  --output_json "$fold_dir/candidate_pool.json" \
  --device "$device" --force

for operator in directional jacobian; do
  python "$repo/feature_extract/tools/vfm/create_goal_maplet_map_crossfit_phase_policy.py" \
    --operator "$operator" \
    --physical_map "$physical" \
    --canonical_field "$fold_dir/canonical_field.npz" \
    --physical_instance_readout "$fold_dir/physical_readout.pt" \
    --output_json "$fold_dir/${operator}_policy.json" --force
done

python "$repo/feature_extract/tools/vfm/verify_goal_maplet_pose_modes_with_surface_field.py" \
  --contributors "$contributors" \
  --candidate_pool "$fold_dir/candidate_pool.json" \
  --physical_map "$physical" \
  --canonical_field "$fold_dir/canonical_field.npz" \
  --surface_mapper "$mapper" \
  --field_feature_contract "$fold_dir/field_feature_contract.json" \
  --physical_instance_readout "$fold_dir/physical_readout.pt" \
  --phase_preserving_dual_band \
  --phase_readout_model "$fold_dir/directional_policy.json" \
  --render_supersample_factor 2 --maximum_modes 16 \
  --role context --spatial_role_readout \
  --output_json "$fold_dir/directional_report.json" \
  --device "$device" --force

python "$repo/feature_extract/tools/vfm/verify_goal_maplet_pose_modes_with_surface_field.py" \
  --contributors "$contributors" \
  --candidate_pool "$fold_dir/candidate_pool.json" \
  --physical_map "$physical" \
  --canonical_field "$fold_dir/canonical_field.npz" \
  --surface_mapper "$mapper" \
  --field_feature_contract "$fold_dir/field_feature_contract.json" \
  --physical_instance_readout "$fold_dir/physical_readout.pt" \
  --phase_preserving_dual_band \
  --phase_readout_model "$fold_dir/jacobian_policy.json" \
  --render_supersample_factor 2 --maximum_modes 16 \
  --output_json "$fold_dir/jacobian_report.json" \
  --device "$device" --force

python "$repo/feature_extract/tools/vfm/evaluate_goal_maplet_phase_operator_crossfit.py" \
  --directional_report "$fold_dir/directional_report.json" \
  --jacobian_report "$fold_dir/jacobian_report.json" \
  --directional_policy "$fold_dir/directional_policy.json" \
  --jacobian_policy "$fold_dir/jacobian_policy.json" \
  --canonical_field_summary "$fold_dir/canonical_field.json" \
  --mapping_contributors "$contributors" \
  --surface_mapper "$mapper" \
  --physical_instance_readout "$fold_dir/physical_readout.pt" \
  --field_feature_contract "$fold_dir/field_feature_contract.json" \
  --validity_summary "$fold_dir/validity.json" \
  --typed_graph_summary "$fold_dir/typed_graph.json" \
  --heldout_trajectories "$held" \
  --geometry_protocol fixed_train_2dgs \
  --output_json "$fold_dir/phase_operator_evaluation.json" --force

for operator in directional jacobian; do
  python "$repo/feature_extract/tools/vfm/audit_goal_maplet_phase_basin.py" \
    --contributors "$contributors" \
    --physical_map "$physical" \
    --canonical_field "$fold_dir/canonical_field.npz" \
    --surface_mapper "$mapper" \
    --phase_readout_model "$fold_dir/${operator}_policy.json" \
    --trajectory_ids "$held" \
    --render_supersample_factor 2 \
    --translation_magnitudes 0.25,0.5 \
    --rotation_magnitudes_deg 3 \
    --output_json "$fold_dir/${operator}_basin.json" \
    --device "$device" --force
done
