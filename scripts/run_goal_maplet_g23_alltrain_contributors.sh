#!/usr/bin/env bash
set -euo pipefail

repo=/root/ICLPose
base=$repo/output/vfm/2dgs_surface/StMarysChurch/full_train
output=$base/mainline_v6/contributors_alltrain_clean
protocol=$repo/configs/vfm/goal_maplet_stmarys_official_train_oof_v1.json
summary0=$base/mainline_v6/contributors_alltrain_clean_shard0.json
summary1=$base/mainline_v6/contributors_alltrain_clean_shard1.json
audit=$base/mainline_v6/contributors_alltrain_clean_audit.json
force=()
if [[ ${G23_FORCE:-0} == 1 ]]; then
  force=(--force)
fi

export PYTHONPATH=$repo

python "$repo/feature_extract/tools/vfm/build_goal_maplet_official_oof_protocol.py" \
  --train_pose_file /hy-tmp/Cambridge_stdloc/StMarysChurch/dataset_train.txt \
  --test_pose_file /hy-tmp/Cambridge_stdloc/StMarysChurch/dataset_test.txt \
  --train_token_manifest "$repo/output/vfm_tokens/StMarysChurch/full_1024x576/train_manifest.json" \
  --test_token_manifest "$repo/output/vfm_tokens/StMarysChurch/full_1024x576/test_manifest.json" \
  --mapping_camera_manifest "$base/mainline_v3_feature_aligned/mapping_camera_manifest.json" \
  --mapping_depth_bank "$base/final_surface/mapping_depth_bank.json" \
  --output_json "$protocol" --force

common=(
  "$repo/feature_extract/tools/vfm/build_v6_contributor_cache.py"
  --gaussian_ply /root/StMaryChurch2dgs.ply
  --clean_gaussian_ply /root/StMaryChurch2dgs_clean.ply
  --mapping_manifest "$repo/output/vfm_tokens/StMarysChurch/full_1024x576/train_manifest.json"
  --mapping_pose_file /hy-tmp/Cambridge_stdloc/StMarysChurch/dataset_train.txt
  --mapping_camera_manifest "$base/mainline_v3_feature_aligned/mapping_camera_manifest.json"
  --output_dir "$output"
  --trajectory_ids seq1 seq2 seq4 seq6 seq7 seq8 seq9 seq10 seq11 seq12 seq14
  --views_per_trajectory 10000
  --width 256 --height 144 --top_k 4
  --shard_count 2 --device cuda:0
)

pids=()
if [[ ! -e $summary0 || ${G23_FORCE:-0} == 1 ]]; then
  CUDA_VISIBLE_DEVICES=0 python "${common[@]}" --shard_index 0 \
    --summary_json "$summary0" "${force[@]}" &
  pids+=("$!")
fi
if [[ ! -e $summary1 || ${G23_FORCE:-0} == 1 ]]; then
  CUDA_VISIBLE_DEVICES=1 python "${common[@]}" --shard_index 1 \
    --summary_json "$summary1" "${force[@]}" &
  pids+=("$!")
fi
for pid in "${pids[@]}"; do
  wait "$pid"
done

python "$repo/feature_extract/tools/vfm/audit_goal_maplet_contributor_cache.py" \
  --contributors "$output" --protocol_json "$protocol" \
  --shard_summaries "$summary0" "$summary1" \
  --output_json "$audit" --force
