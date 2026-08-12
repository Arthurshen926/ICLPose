#!/usr/bin/env bash
set -euo pipefail

repo=/root/ICLPose
base=$repo/output/vfm/2dgs_surface/StMarysChurch/full_train
goal=$base/goal_maplet
root=${G23_OOF_ROOT:-$goal/g23_official_train_oof_v1}
default_contributors=$base/mainline_v6/contributors_alltrain_clean
default_physical=$goal/physical_map_v4.npz
tag=${G23_CANDIDATE_TAG:-exact128_anchor16x4}
mapping_candidates=${G23_MAPPING_VIEW_CANDIDATES:-128}
mapping_anchors=${G23_MAPPING_VIEW_ANCHORS:-16}
mapping_support_pairs=${G23_MAPPING_VIEW_SUPPORT_PAIRS:-64}
mapping_hypotheses=${G23_MAPPING_VIEW_HYPOTHESES:-4}
prescore_per_anchor=${G23_PRESCORE_PER_ANCHOR:-96}
exact_verify_count=${G23_EXACT_VERIFY_COUNT:-128}
exact_keep_per_anchor=${G23_EXACT_KEEP_PER_ANCHOR:-4}
exact_protected_anchors=${G23_EXACT_PROTECTED_ANCHORS:-16}
sparse_primitives_per_child=${G23_SPARSE_PRIMITIVES_PER_CHILD:-8}
sparse_splat_radius=${G23_SPARSE_SPLAT_RADIUS:-2}
geometry_confidence=${G23_GEOMETRY_CONFIDENCE:-0.05}
translation_nms=${G23_TRANSLATION_NMS_M:-0.2}
rotation_nms=${G23_ROTATION_NMS_DEG:-3}
export PYTHONPATH=$repo
selected_candidate_args=()
selected_candidate_lineage_args=()
if [[ -n ${G23_CANDIDATE_SELECTION:-} ]]; then
  if [[ ! -e $G23_CANDIDATE_SELECTION ]]; then
    echo "missing selected candidate report: $G23_CANDIDATE_SELECTION" >&2
    exit 2
  fi
  mapfile -t selected_candidate_args < <(
    python "$repo/feature_extract/tools/vfm/render_goal_maplet_selected_candidate_env.py" \
      --selection "$G23_CANDIDATE_SELECTION" --format args
  )
  selected_candidate_lineage_args=(
    --candidate_selection_report "$G23_CANDIDATE_SELECTION"
  )
fi

run_fold() {
  local gpu=$1
  local fold=$2
  local shard_index=$3
  local shard_count=$4
  shift 4
  local held=("$@")
  local dir=$root/$fold
  local suffix=
  if [[ $shard_count -gt 1 ]]; then
    suffix=_shard${shard_index}of${shard_count}
  fi
  local contributors=$default_contributors
  local physical=$default_physical
  if [[ ${G23_STRICT_GEOMETRY:-0} == 1 ]]; then
    contributors=$dir/contributors_official_train
    physical=$dir/physical_map.npz
    if [[ ! -e $dir/strict_map_audit.json \
          || ! -e $dir/contributors_official_train_audit.json ]]; then
      echo "strict fold inputs are incomplete for $fold" >&2
      exit 2
    fi
  fi
  local output=$dir/candidates_${tag}${suffix}.json
  if [[ -e $output && ${G23_FORCE:-0} != 1 ]]; then
    return
  fi
  local force=()
  if [[ ${G23_FORCE:-0} == 1 ]]; then
    force=(--force)
  fi
  CUDA_VISIBLE_DEVICES=$gpu python \
    "$repo/feature_extract/tools/vfm/evaluate_goal_maplet_pose_modes.py" \
    --contributors "$contributors" --physical_map "$physical" \
    --canonical_field "$dir/canonical_field.npz" \
    --surface_mapper "$dir/surface_mapper.pt" \
    --field_feature_contract "$dir/field_feature_contract.json" \
    --validity_calibration "$dir/validity.npz" \
    --typed_graph "$dir/typed_graph.npz" \
    --physical_instance_readout "$dir/physical_readout.pt" \
    --geometry_head "$dir/geometry_head/radio_highres_geometry_head.pt" \
    --mapping_view_graph "$dir/mapping_view_graph.npz" \
    --output_json "$output" --parent_mode actual --child_mode actual \
    --maximum_modes 32 --proposal_method view_geometry \
    --mapping_view_candidates "$mapping_candidates" --mapping_view_anchors "$mapping_anchors" \
    --mapping_view_support_pairs "$mapping_support_pairs" --mapping_view_hypotheses "$mapping_hypotheses" \
    --view_geometry_prescore_per_anchor "$prescore_per_anchor" \
    --view_geometry_exact_verify_count "$exact_verify_count" \
    --view_geometry_exact_keep_per_anchor "$exact_keep_per_anchor" \
    --view_geometry_exact_protected_anchors "$exact_protected_anchors" \
    --view_geometry_exact_pool_semantics anchor_quota \
    --geometry_proposal_confidence "$geometry_confidence" --geometry_pair_supports 64 \
    --geometry_support_pairs 512 --geometry_pair_candidates 8 \
    --geometry_extension_candidates 16 --geometry_pair_hypotheses 2 \
    --geometry_preliminary_poses 768 \
    --sparse_vfm_primitives_per_child "$sparse_primitives_per_child" --sparse_vfm_batch_size 16 \
    --sparse_vfm_maximum_splat_radius_tokens "$sparse_splat_radius" \
    --sparse_primitive_score_semantics visible_sample_mean \
    --translation_nms_m "$translation_nms" --rotation_nms_deg "$rotation_nms" \
    --include_trajectories "${held[@]}" \
    --shard_index "$shard_index" --shard_count "$shard_count" \
    --device cuda:0 --quiet_rows "${selected_candidate_args[@]}" \
    "${selected_candidate_lineage_args[@]}" \
    "${force[@]}" > "$dir/candidates_${tag}${suffix}.log" 2>&1
}

gpu0() {
  run_fold 0 fold0 0 1 seq2
  run_fold 0 fold3 0 1 seq7 seq12
  run_fold 0 fold4 0 2 seq6 seq9 seq10 seq11
}
gpu1() {
  run_fold 1 fold1 0 1 seq4
  run_fold 1 fold2 0 1 seq1 seq8 seq14
  run_fold 1 fold4 1 2 seq6 seq9 seq10 seq11
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

fold4=$root/fold4
fold4_merged=$fold4/candidates_${tag}.json
if [[ -e $fold4/candidates_${tag}_shard0of2.json \
      && -e $fold4/candidates_${tag}_shard1of2.json \
      && ( ! -e $fold4_merged || ${G23_FORCE:-0} == 1 ) ]]; then
  merge_force=()
  if [[ ${G23_FORCE:-0} == 1 ]]; then
    merge_force=(--force)
  fi
  python "$repo/feature_extract/tools/vfm/merge_goal_maplet_pose_modes.py" \
    --inputs \
      "$fold4/candidates_${tag}_shard0of2.json" \
      "$fold4/candidates_${tag}_shard1of2.json" \
    --output_json "$fold4_merged" "${merge_force[@]}"
fi

aggregate=$root/candidates_${tag}_oof_evaluation.json
aggregate_force=()
if [[ ${G23_FORCE:-0} == 1 ]]; then
  aggregate_force=(--force)
fi
if [[ -e $root/fold0/candidates_${tag}.json \
      && -e $root/fold1/candidates_${tag}.json \
      && -e $root/fold2/candidates_${tag}.json \
      && -e $root/fold3/candidates_${tag}.json \
      && -e $root/fold4/candidates_${tag}.json \
      && ( ! -e $aggregate || ${G23_FORCE:-0} == 1 ) ]]; then
  python "$repo/feature_extract/tools/vfm/evaluate_goal_maplet_official_oof_candidates.py" \
    --protocol "$repo/configs/vfm/goal_maplet_stmarys_official_train_oof_v1.json" \
    --fold_reports \
      "fold0=$root/fold0/candidates_${tag}.json" \
      "fold1=$root/fold1/candidates_${tag}.json" \
      "fold2=$root/fold2/candidates_${tag}.json" \
      "fold3=$root/fold3/candidates_${tag}.json" \
      "fold4=$root/fold4/candidates_${tag}.json" \
    --output_json "$aggregate" "${aggregate_force[@]}"
fi
