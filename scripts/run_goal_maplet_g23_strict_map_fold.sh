#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <fold0..fold4|final_alltrain> <physical_gpu_index>" >&2
  exit 2
fi

repo=/root/ICLPose
fold=$1
gpu=$2
protocol=$repo/configs/vfm/goal_maplet_stmarys_official_train_oof_v1.json
root=$repo/output/vfm/2dgs_surface/StMarysChurch/full_train/goal_maplet/g23_strict_oof_geometry_v1
dir=$root/$fold
posed=$dir/posed_colmap
inputs=$dir/strict_inputs
geometry=$dir/strict_geometry
iterations=${G23_MATCHA_GAUSSIAN_ITERATIONS:-30000}
export PYTHONPATH=$repo

if [[ $fold != final_alltrain && ! $fold =~ ^fold[0-4]$ ]]; then
  echo "unsupported fold: $fold" >&2
  exit 2
fi
if [[ ! -e $posed/fold_colmap_dataset.json ]]; then
  echo "missing posed fold dataset: $posed" >&2
  exit 2
fi
mkdir -p "$inputs" "$geometry"

if [[ ! -e $inputs/strict_mapping_inputs.json ]]; then
  python "$repo/feature_extract/tools/vfm/build_goal_maplet_strict_mapping_inputs.py" \
    --protocol "$protocol" --fold_id "$fold" \
    --output_manifest "$inputs/train_manifest.json" \
    --output_pose_file "$inputs/dataset_train.txt" \
    --output_json "$inputs/strict_mapping_inputs.json" \
    > "$inputs/build.log" 2>&1
fi

python "$repo/feature_extract/tools/vfm/run_goal_maplet_strict_matcha_fold.py" \
  --fold_dataset "$posed" --output_dir "$geometry" --gpu "$gpu" \
  --gaussian_iterations "$iterations" \
  --manifest "$geometry/strict_matcha_run_manifest.json" \
  > "$geometry/run.log" 2>&1

ply=$geometry/free_gaussians/point_cloud/iteration_${iterations}/point_cloud.ply
bootstrap_map=$dir/bootstrap_map
bootstrap_surface=$dir/bootstrap_surface
final_map=$dir/final_map
final_surface=$dir/final_surface
mapper=$dir/surface_mapper.pt
camera_dir=$posed/sparse/0

if [[ ! -e $bootstrap_map/region_summary.json ]]; then
  mkdir -p "$bootstrap_map"
  CUDA_VISIBLE_DEVICES=$gpu python \
    "$repo/feature_extract/tools/vfm/build_vfm_2dgs_anchor_map.py" \
    --gaussian_ply "$ply" \
    --reference_manifest "$inputs/train_manifest.json" \
    --reference_pose_file "$inputs/dataset_train.txt" \
    --camera_model_dir "$camera_dir" \
    --require_camera_for_every_view \
    --canonical_vfm_2dgs --canonical_token_supply grid_top \
    --disable_virtual_surface_cells \
    --surface_adjacency_element_radius_cap 0.1 \
    --output_npz "$bootstrap_map/region_map.npz" \
    --summary_json "$bootstrap_map/region_summary.json" \
    --surface_npz "$bootstrap_map/surface_elements.npz" \
    --descriptor_index_npz "$bootstrap_map/region_descriptor_index.npz" \
    --contribution_dir "$bootstrap_map/contributions" \
    --observation_bank_npz "$bootstrap_map/observation_bank.npz" \
    --contribution_device cuda:0 --mapper_device cuda:0 \
    > "$bootstrap_map/build.log" 2>&1
fi

if [[ ! -e $bootstrap_surface/surface_map_summary.json ]]; then
  mkdir -p "$bootstrap_surface"
  python "$repo/feature_extract/tools/vfm/build_2dgs_surface_map.py" \
    --surface_elements "$bootstrap_map/surface_elements.npz" \
    --region_map "$bootstrap_map/region_map.npz" \
    --observation_bank "$bootstrap_map/observation_bank.npz" \
    --radio_final_manifest "$inputs/train_manifest.json" \
    --reference_pose_file "$inputs/dataset_train.txt" \
    --camera_model_dir "$camera_dir" \
    --require_camera_for_every_view \
    --output_maplets "$bootstrap_surface/surface_maplets.npz" \
    --output_anchors "$bootstrap_surface/stable_surface_anchors.npz" \
    --summary_json "$bootstrap_surface/surface_map_summary.json" \
    > "$bootstrap_surface/build.log" 2>&1
fi

mapfile -t mapping_routes < <(python - "$inputs/strict_mapping_inputs.json" <<'PY'
import json,sys
sys.stdout.write("\n".join(json.load(open(sys.argv[1]))["mapping_trajectories"]))
PY
)
mapfile -t held_routes < <(python - "$inputs/strict_mapping_inputs.json" <<'PY'
import json,sys
sys.stdout.write("\n".join(json.load(open(sys.argv[1]))["held_query_trajectories"]))
PY
)
strict_holdout=("${held_routes[@]}" seq3 seq5 seq13)
if [[ $fold == final_alltrain ]]; then
  fold_seed=2350
else
  fold_seed=$((2350 + ${fold#fold}))
fi
if [[ ! -e $mapper || ! -e $dir/surface_mapper.json ]]; then
  CUDA_VISIBLE_DEVICES=$gpu python \
    "$repo/feature_extract/tools/vfm/train_surface_maplet_mapper.py" \
    --surface_maplets "$bootstrap_surface/surface_maplets.npz" \
    --radio_final_manifest "$inputs/train_manifest.json" \
    --output_checkpoint "$mapper" --summary_json "$dir/surface_mapper.json" \
    --checkpoint_protocol fixed_epoch_no_selection \
    --training_trajectory_ids "${mapping_routes[@]}" \
    --strict_holdout_trajectory_ids "${strict_holdout[@]}" \
    --epochs 120 --steps_per_epoch 8 --batch_maplets 64 \
    --hidden_dim 256 --output_dim 128 --dropout 0 \
    --learning_rate 0.0002 --weight_decay 0.0001 \
    --temperature 0.07 --hard_negative_radius 0.75 \
    --hard_negative_margin 0.25 --hard_negative_weight 0.20 \
    --seed "$fold_seed" --device cuda:0 \
    > "$dir/surface_mapper.log" 2>&1
fi

if [[ ! -e $final_map/region_summary.json ]]; then
  mkdir -p "$final_map"
  CUDA_VISIBLE_DEVICES=$gpu python \
    "$repo/feature_extract/tools/vfm/build_vfm_2dgs_anchor_map.py" \
    --gaussian_ply "$ply" \
    --reference_manifest "$inputs/train_manifest.json" \
    --reference_pose_file "$inputs/dataset_train.txt" \
    --camera_model_dir "$camera_dir" \
    --require_camera_for_every_view \
    --surface_maplet_mapper_checkpoint "$mapper" \
    --canonical_vfm_2dgs --canonical_token_supply grid_top \
    --disable_virtual_surface_cells \
    --reuse_surface_npz "$bootstrap_map/surface_elements.npz" \
    --reuse_contribution_dir "$bootstrap_map/contributions" \
    --contribution_dir "$bootstrap_map/contributions" \
    --output_npz "$final_map/region_map.npz" \
    --summary_json "$final_map/region_summary.json" \
    --descriptor_index_npz "$final_map/region_descriptor_index.npz" \
    --observation_bank_npz "$final_map/observation_bank.npz" \
    --contribution_device cuda:0 --mapper_device cuda:0 \
    > "$final_map/build.log" 2>&1
fi

if [[ ! -e $final_surface/surface_map_summary.json ]]; then
  mkdir -p "$final_surface"
  CUDA_VISIBLE_DEVICES=$gpu python \
    "$repo/feature_extract/tools/vfm/build_2dgs_surface_map.py" \
    --surface_elements "$bootstrap_map/surface_elements.npz" \
    --region_map "$final_map/region_map.npz" \
    --observation_bank "$final_map/observation_bank.npz" \
    --radio_final_manifest "$inputs/train_manifest.json" \
    --reference_pose_file "$inputs/dataset_train.txt" \
    --camera_model_dir "$camera_dir" \
    --require_camera_for_every_view \
    --surface_maplet_mapper_checkpoint "$mapper" --mapper_device cuda:0 \
    --output_maplets "$final_surface/surface_maplets.npz" \
    --output_anchors "$final_surface/stable_surface_anchors.npz" \
    --summary_json "$final_surface/surface_map_summary.json" \
    > "$final_surface/build.log" 2>&1
fi

if [[ ! -e $dir/physical_map.npz || ! -e $dir/physical_map_audit.json ]]; then
  python "$repo/feature_extract/tools/vfm/build_goal_maplet_physical_map.py" \
    --surface_elements "$bootstrap_map/surface_elements.npz" \
    --clean_surface_elements \
    --legacy_maplets "$final_surface/surface_maplets.npz" \
    --region_map "$final_map/region_map.npz" \
    --mapping_pose_file "$inputs/dataset_train.txt" \
    --output_map "$dir/physical_map.npz" \
    --audit_json "$dir/physical_map_audit.json" \
    > "$dir/physical_map.log" 2>&1
fi

if [[ ! -e $dir/strict_map_audit.json ]]; then
  python "$repo/feature_extract/tools/vfm/audit_goal_maplet_strict_map_fold.py" \
    --fold_dir "$dir" --output_json "$dir/strict_map_audit.json" \
    > "$dir/strict_map_audit.log" 2>&1
fi
