#!/usr/bin/env bash
set -euo pipefail

repo=/root/ICLPose
protocol=$repo/configs/vfm/goal_maplet_stmarys_official_train_oof_v1.json
image_root=/hy-tmp/Cambridge_stdloc/StMarysChurch
camera_manifest=$repo/output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v3_feature_aligned/mapping_camera_manifest.json
root=$repo/output/vfm/2dgs_surface/StMarysChurch/full_train/goal_maplet/g23_strict_oof_geometry_v1
maximum_charts=${G23_MATCHA_MAXIMUM_CHARTS:-64}
export PYTHONPATH=$repo

force=()
if [[ ${G23_FORCE:-0} == 1 ]]; then
  force=(--force)
fi

for fold_index in 0 1 2 3 4; do
  fold=fold$fold_index
  output=$root/$fold/posed_colmap
  summary=$output/fold_colmap_dataset.json
  if [[ -e $summary && ${G23_FORCE:-0} != 1 ]]; then
    continue
  fi
  mkdir -p "$root/$fold"
  python "$repo/feature_extract/tools/vfm/build_goal_maplet_fold_colmap_dataset.py" \
    --protocol_json "$protocol" --fold_id "$fold" \
    --image_root "$image_root" --camera_manifest "$camera_manifest" \
    --maximum_images "$maximum_charts" --output_dir "$output" "${force[@]}" \
    > "$root/$fold/posed_colmap_build.log" 2>&1
done

fold=final_alltrain
output=$root/$fold/posed_colmap
summary=$output/fold_colmap_dataset.json
if [[ ! -e $summary || ${G23_FORCE:-0} == 1 ]]; then
  mkdir -p "$root/$fold"
  python "$repo/feature_extract/tools/vfm/build_goal_maplet_fold_colmap_dataset.py" \
    --protocol_json "$protocol" --fold_id "$fold" \
    --image_root "$image_root" --camera_manifest "$camera_manifest" \
    --maximum_images "$maximum_charts" --output_dir "$output" "${force[@]}" \
    > "$root/$fold/posed_colmap_build.log" 2>&1
fi
