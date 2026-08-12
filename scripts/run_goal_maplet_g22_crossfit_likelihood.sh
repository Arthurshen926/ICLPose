#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
base=$repo/output/vfm/2dgs_surface/StMarysChurch/full_train
goal=$base/goal_maplet
contributors=$base/mainline_v6/contributors_setcover128_clean
teachers=$base/mainline_v6/v8_offline_radio_teacher_regions_setcover128
physical=$goal/physical_map_v4.npz
surface_maplets=$base/bootstrap_surface/surface_maplets.npz
radio_manifest=$repo/output/vfm_tokens/StMarysChurch/full_1024x576/train_manifest.json
geometry_head=$goal/g20_6_joint/head_common_seed2020/radio_highres_geometry_head.pt
fold=${1:-$goal/map_crossfit_g22_1/hold_seq12_seq14}
gpu0=${G22_GPU0:-cuda:0}
gpu1=${G22_GPU1:-cuda:1}
force_flag=()
if [[ ${G22_FORCE:-0} == 1 ]]; then
  force_flag=(--force)
fi

mkdir -p "$fold/logs"
export PYTHONPATH=$repo

run_if_missing() {
  local sentinel=$1
  shift
  if [[ ! -e $sentinel || ${G22_FORCE:-0} == 1 ]]; then
    "$@"
  fi
}

# The mapper checkpoint is selected on seq9. Both target acquisitions are
# strict holdouts, and seq9 is subsequently excluded from the deployed field.
run_if_missing "$fold/surface_mapper.pt" \
  python "$repo/feature_extract/tools/vfm/train_surface_maplet_mapper.py" \
    --surface_maplets "$surface_maplets" \
    --radio_final_manifest "$radio_manifest" \
    --output_checkpoint "$fold/surface_mapper.pt" \
    --summary_json "$fold/surface_mapper.json" \
    --training_trajectory_ids seq1 seq2 seq4 seq6 seq7 seq8 seq10 seq11 \
    --validation_trajectory_ids seq9 \
    --prototype_trajectory_ids seq1 seq2 seq4 seq6 seq7 seq8 seq10 seq11 \
    --strict_holdout_trajectory_ids seq3 seq5 seq12 seq13 seq14 \
    --device "$gpu0"

run_if_missing "$fold/canonical_field.npz" \
  python "$repo/feature_extract/tools/vfm/build_goal_maplet_canonical_field_from_contributors.py" \
    --contributors "$contributors" \
    --physical_map "$physical" \
    --surface_mapper "$fold/surface_mapper.pt" \
    --feature_space retrieval_mapper \
    --exclude_trajectories seq3 seq5 seq9 seq12 seq13 seq14 \
    --output_field "$fold/canonical_field.npz" \
    --summary_json "$fold/canonical_field.json" \
    --device "$gpu0" "${force_flag[@]}"

run_if_missing "$fold/field_feature_contract.json" \
  python "$repo/feature_extract/tools/vfm/build_goal_maplet_feature_contract.py" \
    --canonical_field "$fold/canonical_field.npz" \
    --query_readout_type surface_maplet_mapper \
    --query_readout "$fold/surface_mapper.pt" \
    --render_protocol exact_clean_2dgs_identity_or_feature_token_grid_v1 \
    --output_json "$fold/field_feature_contract.json" "${force_flag[@]}"

run_if_missing "$fold/physical_readout.pt" \
  python "$repo/feature_extract/tools/vfm/train_goal_maplet_physical_instance_readout.py" \
    --contributors "$contributors" \
    --teacher_cache "$teachers" \
    --physical_map "$physical" \
    --canonical_field "$fold/canonical_field.npz" \
    --surface_mapper "$fold/surface_mapper.pt" \
    --train_trajectories seq1 seq2 seq4 seq6 seq7 seq8 seq11 \
    --selection_trajectories seq10 \
    --validation_trajectories seq9 \
    --steps 800 --batch_size 128 --learning_rate 0.0002 \
    --output_readout "$fold/physical_readout.pt" \
    --summary_json "$fold/physical_readout.json" \
    --device "$gpu0" "${force_flag[@]}"

run_if_missing "$fold/typed_graph.npz" \
  python "$repo/feature_extract/tools/vfm/build_goal_maplet_typed_graph.py" \
    --contributors "$contributors" \
    --physical_map "$physical" \
    --canonical_field "$fold/canonical_field.npz" \
    --physical_instance_readout "$fold/physical_readout.pt" \
    --exclude_trajectories seq3 seq5 seq9 seq12 seq13 seq14 \
    --output_graph "$fold/typed_graph.npz" \
    --summary_json "$fold/typed_graph.json" \
    --device "$gpu0" "${force_flag[@]}"

run_if_missing "$fold/validity.npz" \
  python "$repo/feature_extract/tools/vfm/calibrate_goal_maplet_validity.py" \
    --contributors "$contributors" \
    --physical_map "$physical" \
    --canonical_field "$fold/canonical_field.npz" \
    --surface_mapper "$fold/surface_mapper.pt" \
    --physical_instance_readout "$fold/physical_readout.pt" \
    --pooling current_1x1_3x3_5x5_9x9 \
    --include_trajectories seq10 \
    --output_calibration "$fold/validity.npz" \
    --summary_json "$fold/validity.json" \
    --device "$gpu0" "${force_flag[@]}"

run_if_missing "$fold/mapping_view_graph_g22.npz" \
  python "$repo/feature_extract/tools/vfm/build_goal_maplet_mapping_view_graph.py" \
    --contributors "$contributors" \
    --physical_map "$physical" \
    --canonical_field "$fold/canonical_field.npz" \
    --exclude_trajectories seq3 seq5 seq9 seq12 seq13 seq14 \
    --output_graph "$fold/mapping_view_graph_g22.npz" \
    --output_json "$fold/mapping_view_graph_g22.json" "${force_flag[@]}"

run_candidates() {
  local held=$1
  local device=$2
  shift 2
  run_if_missing "$fold/${held}_exact128.json" \
    python "$repo/feature_extract/tools/vfm/evaluate_goal_maplet_pose_modes.py" \
      --contributors "$contributors" \
      --physical_map "$physical" \
      --canonical_field "$fold/canonical_field.npz" \
      --surface_mapper "$fold/surface_mapper.pt" \
      --field_feature_contract "$fold/field_feature_contract.json" \
      --validity_calibration "$fold/validity.npz" \
      --typed_graph "$fold/typed_graph.npz" \
      --physical_instance_readout "$fold/physical_readout.pt" \
      --geometry_head "$geometry_head" \
      --mapping_view_graph "$fold/mapping_view_graph_g22.npz" \
      --output_json "$fold/${held}_exact128.json" \
      --parent_mode actual --child_mode actual \
      --maximum_modes 128 --proposal_method view_geometry \
      --mapping_view_candidates 128 --mapping_view_anchors 16 \
      --mapping_view_support_pairs 64 --mapping_view_hypotheses 4 \
      --view_geometry_prescore_per_anchor 96 \
      --view_geometry_exact_verify_count 128 \
      --view_geometry_exact_keep_per_anchor 4 \
      --view_geometry_exact_protected_anchors 16 \
      --view_geometry_exact_pool_semantics anchor_quota \
      --view_geometry_final_ranking_semantics exact_vfm \
      --geometry_proposal_confidence 0.05 --geometry_pair_supports 64 \
      --geometry_support_pairs 512 --geometry_pair_candidates 8 \
      --geometry_extension_candidates 16 --geometry_pair_hypotheses 2 \
      --geometry_preliminary_poses 768 \
      --sparse_vfm_primitives_per_child 8 \
      --sparse_vfm_batch_size 16 \
      --sparse_vfm_maximum_splat_radius_tokens 2 \
      --sparse_primitive_score_semantics visible_sample_mean \
      --image_ids "$@" --device "$device" --quiet_rows "${force_flag[@]}"
}

run_candidates seq12 "$gpu0" \
  seq12/frame00063.png seq12/frame00064.png seq12/frame00065.png \
  seq12/frame00066.png seq12/frame00067.png seq12/frame00093.png \
  seq12/frame00097.png seq12/frame00139.png seq12/frame00144.png \
  seq12/frame00155.png seq12/frame00159.png \
  > "$fold/logs/seq12_candidates.log" 2>&1 &
pid0=$!

run_candidates seq14 "$gpu1" \
  seq14/frame00001.png seq14/frame00004.png seq14/frame00005.png \
  seq14/frame00020.png seq14/frame00021.png seq14/frame00026.png \
  > "$fold/logs/seq14_candidates.log" 2>&1 &
pid1=$!

status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
if [[ $status -ne 0 ]]; then
  exit "$status"
fi

build_samples() {
  local held=$1
  local device=$2
  run_if_missing "$fold/${held}_surface_samples.npz" \
    python "$repo/feature_extract/tools/vfm/build_goal_maplet_surface_likelihood_samples.py" \
      --contributors "$contributors" \
      --candidate_pool "$fold/${held}_exact128.json" \
      --physical_map "$physical" \
      --canonical_field "$fold/canonical_field.npz" \
      --surface_mapper "$fold/surface_mapper.pt" \
      --physical_instance_readout "$fold/physical_readout.pt" \
      --teacher_cache "$teachers" \
      --maximum_modes 128 \
      --output_npz "$fold/${held}_surface_samples.npz" \
      --device "$device" "${force_flag[@]}"
}

build_samples seq12 "$gpu0" > "$fold/logs/seq12_samples.log" 2>&1 &
pid0=$!
build_samples seq14 "$gpu1" > "$fold/logs/seq14_samples.log" 2>&1 &
pid1=$!
status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
if [[ $status -ne 0 ]]; then
  exit "$status"
fi

# Direction-specific models never see their target trajectory. Epoch 20 and
# all hyperparameters are fixed before the two outer-fold evaluations.
train_likelihood() {
  local source=$1
  local device=$2
  run_if_missing "$fold/surface_likelihood_train_${source}.pt" \
    python "$repo/feature_extract/tools/vfm/train_goal_maplet_surface_pose_likelihood.py" \
      --train_samples "$fold/${source}_surface_samples.npz" \
      --checkpoint_protocol fixed_epoch_no_selection \
      --epochs 20 --learning_rate 0.0002 --weight_decay 0.0001 \
      --typed_weight 0.05 --hidden_dim 64 --seed 1801 \
      --output_model "$fold/surface_likelihood_train_${source}.pt" \
      --summary_json "$fold/surface_likelihood_train_${source}.json" \
      --device "$device" "${force_flag[@]}"
}

train_likelihood seq12 "$gpu0" > "$fold/logs/train_seq12.log" 2>&1 &
pid0=$!
train_likelihood seq14 "$gpu1" > "$fold/logs/train_seq14.log" 2>&1 &
pid1=$!
status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
if [[ $status -ne 0 ]]; then
  exit "$status"
fi

run_if_missing "$fold/surface_likelihood_outer_crossfit.json" \
  python "$repo/feature_extract/tools/vfm/evaluate_goal_maplet_surface_likelihood_crossfit.py" \
    --canonical_field_summary "$fold/canonical_field.json" \
    --fold seq12 "$fold/surface_likelihood_train_seq14.pt" "$fold/seq12_surface_samples.npz" \
    --fold seq14 "$fold/surface_likelihood_train_seq12.pt" "$fold/seq14_surface_samples.npz" \
    --output_json "$fold/surface_likelihood_outer_crossfit.json" \
    --device "$gpu0" "${force_flag[@]}"

# One untouched seq12 candidate set is re-rendered through the actual runtime
# path, guarding against a sample-only evaluation/inference mismatch.
run_if_missing "$fold/seq12_runtime_smoke_frame00139.json" \
  python "$repo/feature_extract/tools/vfm/verify_goal_maplet_pose_modes_with_surface_field.py" \
    --contributors "$contributors" \
    --candidate_pool "$fold/seq12_exact128.json" \
    --physical_map "$physical" \
    --canonical_field "$fold/canonical_field.npz" \
    --surface_mapper "$fold/surface_mapper.pt" \
    --field_feature_contract "$fold/field_feature_contract.json" \
    --physical_instance_readout "$fold/physical_readout.pt" \
    --surface_pose_likelihood "$fold/surface_likelihood_train_seq14.pt" \
    --spatial_role_readout --role context --maximum_modes 128 \
    --shard_index 7 --shard_count 11 \
    --output_json "$fold/seq12_runtime_smoke_frame00139.json" \
    --device "$gpu0" "${force_flag[@]}"
