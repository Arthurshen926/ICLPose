#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
base=$repo/output/vfm/2dgs_surface/StMarysChurch/full_train
goal=$base/goal_maplet
contributors=$base/mainline_v6/contributors_setcover128_clean
physical=$goal/physical_map_v4.npz
output_dir=${1:-$goal/g21_p0_correctness_replay}
gpu0=${G21_GPU0:-cuda:0}
gpu1=${G21_GPU1:-cuda:1}
force_flag=()
if [[ ${G21_FORCE:-0} == 1 ]]; then
  force_flag=(--force)
fi

mkdir -p "$output_dir"
export PYTHONPATH=$repo

run_fold() {
  local fold_name=$1
  local held=$2
  local device=$3
  local geometry_head=$4
  shift 4
  local fold=$goal/map_crossfit_g20_1/$fold_name
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
    --mapping_view_graph "$fold/mapping_view_graph_g21.npz" \
    --output_json "$output_dir/${held}_exact128.json" \
    --parent_mode actual --child_mode actual \
    --maximum_modes 32 --proposal_method view_geometry \
    --mapping_view_candidates 128 --mapping_view_anchors 16 \
    --mapping_view_support_pairs 64 --mapping_view_hypotheses 4 \
    --view_geometry_prescore_per_anchor 96 \
    --view_geometry_exact_verify_count 128 \
    --view_geometry_exact_keep_per_anchor 4 \
    --geometry_proposal_confidence 0.05 --geometry_pair_supports 64 \
    --geometry_support_pairs 512 --geometry_pair_candidates 8 \
    --geometry_extension_candidates 16 --geometry_pair_hypotheses 2 \
    --geometry_preliminary_poses 768 \
    --sparse_vfm_primitives_per_child 8 \
    --sparse_vfm_batch_size 16 \
    --sparse_vfm_maximum_splat_radius_tokens 2 \
    --sparse_primitive_score_semantics visible_sample_mean \
    --image_ids "$@" --device "$device" --quiet_rows "${force_flag[@]}" \
    > "$output_dir/${held}.log" 2>&1
}

run_fold \
  hold_seq12 seq12 "$gpu0" \
  "$goal/g20_5_geometry/head_hold_seq12/radio_highres_geometry_head.pt" \
  seq12/frame00065.png seq12/frame00066.png seq12/frame00093.png \
  seq12/frame00144.png seq12/frame00155.png &
pid0=$!

run_fold \
  hold_seq14 seq14 "$gpu1" \
  "$goal/g20_5_geometry/head_hold_seq14/radio_highres_geometry_head.pt" \
  seq14/frame00004.png seq14/frame00026.png &
pid1=$!

status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
exit "$status"
