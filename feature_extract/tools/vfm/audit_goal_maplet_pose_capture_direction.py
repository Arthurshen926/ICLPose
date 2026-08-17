"""Audit full-map score capture from fixed coupled SE(3) perturbations.

This is an explicit GT-seeded oracle diagnostic, not localization.  For two
frozen coupled directions and four error scales it compares the score at the
perturbed start, the halfway pose, and GT.  A monotone path is a necessary but
not sufficient condition for derivative-free basin capture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.localize_2dgs_surface_queries import _load_raw_final
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.pose_proposal import _rotation_distance_degrees
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import PureRadioPhysicalRetrieval
from feature_extract.vfm.localization_goal_maplet.resident_surface_renderer import FrozenSoftSurfaceSceneGPU
from feature_extract.vfm.localization_goal_maplet.se3_local_quadratic import left_retract_pose_w2c
from feature_extract.vfm.localization_goal_maplet.soft_surface_pose_energy import (
    score_hierarchical_spatial_soft_surface_pose_energy,
)


SCALES = ((0.5, 5.0), (1.0, 10.0), (2.0, 20.0), (2.0, 45.0))
DIRECTIONS = (
    np.asarray([1.0, 0.4, -0.2, 0.3, -0.5, 1.0], dtype=np.float64),
    np.asarray([-0.3, 0.8, 0.5, 1.0, 0.2, -0.4], dtype=np.float64),
)


def _normalized_directions() -> tuple[np.ndarray, ...]:
    result = []
    for value in DIRECTIONS:
        direction = value.copy()
        direction[:3] /= np.linalg.norm(direction[:3])
        direction[3:] /= np.linalg.norm(direction[3:])
        result.append(direction)
    return tuple(result)


def _camera_and_token(contributor: Path) -> tuple[ColmapCamera, Path]:
    with np.load(contributor, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        camera = ColmapCamera(
            0, int(data["camera_model_id"]), int(data["camera_width"]),
            int(data["camera_height"]),
            tuple(np.asarray(data["camera_params"], dtype=np.float64).tolist()),
        )
    return camera, Path(str(metadata["token_path"]))


def _errors(pose: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    center = -pose[:3, :3].T @ pose[:3, 3]
    truth = -target[:3, :3].T @ target[:3, 3]
    return float(np.linalg.norm(center - truth)), float(_rotation_distance_degrees(pose, target))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--contributor", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--image_id", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite capture-direction audit")
    physical_path, field_path = Path(args.physical_map), Path(args.canonical_field)
    mapper_path, contributor_path = Path(args.surface_mapper), Path(args.contributor)
    retrieval_path, pose_path = Path(args.retrieval), Path(args.query_pose_file)
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    field = CanonicalSurfaceField.load_npz(field_path)
    retrieval = PureRadioPhysicalRetrieval.load_npz(retrieval_path)
    if retrieval.image_id != str(args.image_id):
        raise ValueError("retrieval image differs from requested query")
    gt_by_id = {row.image_id: row.pose_w2c for row in parse_cambridge_pose_file(pose_path)}
    if str(args.image_id) not in gt_by_id:
        raise KeyError("query pose file lacks requested image")
    gt = gt_by_id[str(args.image_id)]
    camera, token_path = _camera_and_token(contributor_path)
    mapper, _ = load_surface_maplet_mapper(mapper_path, device=str(args.device))
    query = np.asarray(mapper.project(_load_raw_final(token_path, "radio_final")).measurement_context)
    directions = _normalized_directions()
    labels = ["gt"]
    poses = [gt]
    trajectory_keys = []
    for scale_index, (translation, rotation) in enumerate(SCALES):
        for direction_index, direction in enumerate(directions):
            key = f"s{scale_index}_d{direction_index}"
            trajectory_keys.append(key)
            for fraction, suffix in ((1.0, "start"), (0.5, "half")):
                labels.append(f"{key}_{suffix}")
                poses.append(left_retract_pose_w2c(
                    gt, direction * fraction,
                    translation_step_m=float(translation),
                    rotation_step_degrees=float(rotation),
                ))
    poses_array = np.asarray(poses, dtype=np.float64)
    scene = FrozenSoftSurfaceSceneGPU(physical, field, device=str(args.device))
    rendered = []
    audits = []
    for begin in range(0, poses_array.shape[0], int(args.batch_size)):
        batch = scene.render_exact_batch(
            poses_array[begin : begin + int(args.batch_size)], camera,
            width=query.shape[2], height=query.shape[1],
            selected_child_rows=retrieval.scene_child_rows, top_l=4,
            coordinate_supersample_factor=4,
        )
        rendered.extend(batch.rendered)
        audits.append(batch.audit.__dict__)
    scores = []
    for label, pose, observation in zip(labels, poses_array, rendered):
        energy = score_hierarchical_spatial_soft_surface_pose_energy(
            query, retrieval, observation,
            child_to_parent_ids=physical.maplet_ids[physical.child_parent_rows],
            radio_weight=0.5, spatial_kernel_radius=1,
        )
        translation, rotation = _errors(pose, gt)
        scores.append({
            "label": label, "translation_error_m": translation,
            "rotation_error_deg": rotation, **energy.__dict__,
        })
    by_label = {row["label"]: row for row in scores}
    gt_score = float(by_label["gt"]["combined_score"])
    trajectories = []
    for key in trajectory_keys:
        start, half = by_label[key + "_start"], by_label[key + "_half"]
        monotone = (
            float(start["combined_score"]) < float(half["combined_score"])
            and float(half["combined_score"]) < gt_score
        )
        trajectories.append({
            "key": key, "start": start, "half": half,
            "gt_score": gt_score, "strictly_monotone_toward_gt": bool(monotone),
            "start_to_half_gain": float(half["combined_score"] - start["combined_score"]),
            "half_to_gt_gain": float(gt_score - half["combined_score"]),
        })
    report = {
        "artifact_type": "goal_maplet_full_map_pose_capture_direction_audit_v1",
        "image_id": str(args.image_id),
        "input_file_sha256": {
            "physical_map": file_sha256(physical_path),
            "canonical_field": file_sha256(field_path),
            "surface_mapper": file_sha256(mapper_path),
            "contributor": file_sha256(contributor_path),
            "retrieval": file_sha256(retrieval_path),
            "query_pose_file": file_sha256(pose_path),
            "radio_token": file_sha256(token_path),
        },
        "scales": [{"translation_m": t, "rotation_deg": r} for t, r in SCALES],
        "directions": [row.tolist() for row in directions],
        "trajectory_count": len(trajectories),
        "strictly_monotone_count": int(sum(row["strictly_monotone_toward_gt"] for row in trajectories)),
        "all_trajectories_strictly_monotone": bool(all(row["strictly_monotone_toward_gt"] for row in trajectories)),
        "minimum_start_to_half_gain": float(min(row["start_to_half_gain"] for row in trajectories)),
        "minimum_half_to_gt_gain": float(min(row["half_to_gt_gain"] for row in trajectories)),
        "renderer_batches": audits,
        "trajectories": trajectories,
        "claims": {
            "uses_gt_to_construct_oracle_trajectory": True,
            "is_localization_result": False,
            "is_pattern_search_capture_rate": False,
            "is_necessary_not_sufficient_capture_condition": True,
            "uses_alike": False, "uses_pnp": False, "uses_hard_correspondence": False,
        },
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "trajectories"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
