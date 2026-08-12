#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <fold0..fold4|final_alltrain>" >&2
  exit 2
fi

repo=/root/ICLPose
fold=$1
base=$repo/output/vfm/2dgs_surface/StMarysChurch/full_train
root=$base/goal_maplet/g23_strict_oof_geometry_v1
dir=$root/$fold
protocol=$repo/configs/vfm/goal_maplet_stmarys_official_train_oof_v1.json
iterations=${G23_MATCHA_GAUSSIAN_ITERATIONS:-30000}
ply=$dir/strict_geometry/free_gaussians/point_cloud/iteration_${iterations}/point_cloud.ply
surface=$dir/bootstrap_map/surface_elements.npz
camera_train=$base/mainline_v3_feature_aligned/mapping_camera_manifest.json
export PYTHONPATH=$repo

if [[ ! -e $dir/strict_map_audit.json || ! -e $ply || ! -e $surface ]]; then
  echo "strict physical map is not complete for $fold" >&2
  exit 2
fi

build_split() {
  local split=$1
  local manifest=$2
  local poses=$3
  local cameras=$4
  shift 4
  local routes=("$@")
  local output=$dir/contributors_${split}
  local summary0=$dir/contributors_${split}_shard0.json
  local summary1=$dir/contributors_${split}_shard1.json
  local audit=$dir/contributors_${split}_audit.json
  local force=()
  if [[ ${G23_FORCE:-0} == 1 ]]; then
    force=(--force)
  fi
  common=(
    "$repo/feature_extract/tools/vfm/build_v6_contributor_cache.py"
    --gaussian_ply "$ply" --clean_surface_elements "$surface"
    --mapping_manifest "$manifest" --mapping_pose_file "$poses"
    --mapping_camera_manifest "$cameras" --output_dir "$output"
    --trajectory_ids "${routes[@]}" --views_per_trajectory 10000
    --width 256 --height 144 --top_k 4 --shard_count 2 --device cuda:0
  )
  pids=()
  if [[ ! -e $summary0 || ${G23_FORCE:-0} == 1 ]]; then
    CUDA_VISIBLE_DEVICES=0 python "${common[@]}" --shard_index 0 \
      --summary_json "$summary0" "${force[@]}" \
      > "$dir/contributors_${split}_shard0.log" 2>&1 &
    pids+=("$!")
  fi
  if [[ ! -e $summary1 || ${G23_FORCE:-0} == 1 ]]; then
    CUDA_VISIBLE_DEVICES=1 python "${common[@]}" --shard_index 1 \
      --summary_json "$summary1" "${force[@]}" \
      > "$dir/contributors_${split}_shard1.log" 2>&1 &
    pids+=("$!")
  fi
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  if [[ ! -e $audit || ${G23_FORCE:-0} == 1 ]]; then
    python "$repo/feature_extract/tools/vfm/audit_goal_maplet_contributor_cache.py" \
      --contributors "$output" --protocol_json "$protocol" --split "$split" \
      --shard_summaries "$summary0" "$summary1" \
      --output_json "$audit" "${force[@]}"
  fi
}

build_split official_train \
  "$repo/output/vfm_tokens/StMarysChurch/full_1024x576/train_manifest.json" \
  /hy-tmp/Cambridge_stdloc/StMarysChurch/dataset_train.txt \
  "$camera_train" \
  seq1 seq2 seq4 seq6 seq7 seq8 seq9 seq10 seq11 seq12 seq14

if [[ $fold == final_alltrain && ${G23_INCLUDE_OFFICIAL_TEST_INPUTS:-0} == 1 ]]; then
  build_split official_test \
    "$repo/output/vfm_tokens/StMarysChurch/full_1024x576/test_manifest.json" \
    /hy-tmp/Cambridge_stdloc/StMarysChurch/dataset_test.txt \
    "$base/mainline_v3_feature_aligned/query_camera_manifest.json" \
    seq3 seq5 seq13
fi
