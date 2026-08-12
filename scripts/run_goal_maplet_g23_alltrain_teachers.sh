#!/usr/bin/env bash
set -euo pipefail

repo=/root/ICLPose
base=$repo/output/vfm/2dgs_surface/StMarysChurch/full_train
contributors=$base/mainline_v6/contributors_alltrain_clean
output=$base/mainline_v6/v8_offline_radio_teacher_regions_alltrain
protocol=$repo/configs/vfm/goal_maplet_stmarys_official_train_oof_v1.json
summary0=$base/mainline_v6/v8_offline_radio_teacher_regions_alltrain_shard0.json
summary1=$base/mainline_v6/v8_offline_radio_teacher_regions_alltrain_shard1.json
audit=$base/mainline_v6/v8_offline_radio_teacher_regions_alltrain_audit.json
force=()
if [[ ${G23_FORCE:-0} == 1 ]]; then
  force=(--force)
fi
export PYTHONPATH=$repo

pids=()
if [[ ! -e $summary0 || ${G23_FORCE:-0} == 1 ]]; then
  CUDA_VISIBLE_DEVICES=0 python \
    "$repo/feature_extract/tools/vfm/extract_v8_radio_teacher_regions.py" \
    --contributors "$contributors" \
    --image_root /hy-tmp/Cambridge_stdloc/StMarysChurch \
    --output_dir "$output" --summary_json "$summary0" \
    --shard_index 0 --shard_count 2 --device cuda:0 "${force[@]}" &
  pids+=("$!")
fi
if [[ ! -e $summary1 || ${G23_FORCE:-0} == 1 ]]; then
  CUDA_VISIBLE_DEVICES=1 python \
    "$repo/feature_extract/tools/vfm/extract_v8_radio_teacher_regions.py" \
    --contributors "$contributors" \
    --image_root /hy-tmp/Cambridge_stdloc/StMarysChurch \
    --output_dir "$output" --summary_json "$summary1" \
    --shard_index 1 --shard_count 2 --device cuda:0 "${force[@]}" &
  pids+=("$!")
fi
for pid in "${pids[@]}"; do
  wait "$pid"
done

if [[ ! -e $audit || ${G23_FORCE:-0} == 1 ]]; then
  python "$repo/feature_extract/tools/vfm/audit_goal_maplet_teacher_cache.py" \
    --teacher_cache "$output" --protocol_json "$protocol" \
    --shard_summaries "$summary0" "$summary1" \
    --output_json "$audit" "${force[@]}"
fi
