"""Post-label oracle decomposition for hard chart points and query sub-token pixels.

This tool is a diagnostic only.  The query pose is used to select the correct
member of the already retrieved candidate set, to expose the exact image
location of that map point, and to intersect the query ray with a bounded local
tangent patch around that point.  None of its outputs may be used for training,
selection, or deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import (
    _load as _load_correspondences,
    _solve,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _token_pixels
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import (
    GeometryNativePlanarMap,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.retrieval_surface_metrics import (
    inverse_simple_radial,
)


THRESHOLDS = (
    ("0.1m_1deg", 0.1, 1.0),
    ("0.25m_2deg", 0.25, 2.0),
    ("0.5m_5deg", 0.5, 5.0),
    ("1m_10deg", 1.0, 10.0),
    ("2m_45deg", 2.0, 45.0),
)


def _load_pose_inventory(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
    if (
        metadata.get("artifact_type") not in (
            "goal_maplet_dual_surface_geometry_consensus_selected_v1",
            "goal_maplet_uncertainty_normalized_plane_pose_selected_v1",
        )
        or metadata.get("query_pose_or_ground_truth_read") is not False
        or metadata.get("source_rgb_stored_or_consumed_at_runtime") is not False
        or not {"names", "pose_w2c", "usable"}.issubset(arrays)
        or arrays_sha256(arrays) != metadata.get("arrays_sha256")
    ):
        raise ValueError("continuous oracle pose inventory differs")
    return arrays, metadata


def _pose_error(pose_w2c: np.ndarray | None, target_w2c: np.ndarray) -> tuple[float, float]:
    if pose_w2c is None or not np.all(np.isfinite(pose_w2c)):
        return float("inf"), float("inf")
    pose = np.asarray(pose_w2c, np.float64)
    target = np.asarray(target_w2c, np.float64)
    center = -pose[:3, :3].T @ pose[:3, 3]
    target_center = -target[:3, :3].T @ target[:3, 3]
    translation = float(np.linalg.norm(center - target_center))
    rotation = float(Rotation.from_matrix(
        pose[:3, :3] @ target[:3, :3].T,
    ).magnitude() * 180.0 / np.pi)
    return translation, rotation


def _oracle_candidate_rows(
    target_w2c: np.ndarray,
    world_points: np.ndarray,
    query_tokens: np.ndarray,
    query_pixels: np.ndarray,
    camera_matrix: np.ndarray,
    radial_k1: float,
    *,
    maximum_error_px: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the target pose to pick one existing 3D hypothesis per token."""

    world = np.asarray(world_points, np.float64).reshape(-1, 3)
    token = np.asarray(query_tokens, np.int64).reshape(-1)
    pixel = np.asarray(query_pixels, np.float64).reshape(-1, 2)
    pose = np.asarray(target_w2c, np.float64)
    camera = world @ pose[:3, :3].T + pose[:3, 3]
    projected, _ = cv2.projectPoints(
        world, cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3], camera_matrix,
        np.asarray([radial_k1, 0.0, 0.0, 0.0, 0.0], np.float64),
    )
    projected = projected.reshape(-1, 2)
    error = np.linalg.norm(projected - pixel, axis=1)
    valid = (camera[:, 2] > 0.0) & np.isfinite(error) & (error <= float(maximum_error_px))
    rows: list[int] = []
    for token_id in np.unique(token):
        candidate = np.flatnonzero((token == token_id) & valid)
        if len(candidate):
            rows.append(int(candidate[np.argmin(error[candidate])]))
    return np.asarray(rows, np.int64), projected, error


def _inside_token(projected_xy: np.ndarray, query_tokens: np.ndarray) -> np.ndarray:
    token = np.asarray(query_tokens, np.int64).reshape(-1)
    xy = np.asarray(projected_xy, np.float64).reshape(-1, 2)
    origin = np.c_[(token % 64) * 4, (token // 64) * 4]
    return np.all((xy >= origin) & (xy <= origin + 3.0), axis=1)


def _local_continuous_surface_points(
    target_w2c: np.ndarray,
    query_pixels: np.ndarray,
    hard_world_points: np.ndarray,
    physical_plane_ids: np.ndarray,
    camera_matrix: np.ndarray,
    radial_k1: float,
    planar_map: GeometryNativePlanarMap,
    *,
    maximum_tangent_shift_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect target rays with bounded tangent patches through hard points."""

    pose = np.asarray(target_w2c, np.float64)
    pixel = np.asarray(query_pixels, np.float64).reshape(-1, 2)
    hard = np.asarray(hard_world_points, np.float64).reshape(-1, 3)
    plane = np.asarray(physical_plane_ids, np.int64).reshape(-1)
    distorted = np.c_[
        (pixel[:, 0] - camera_matrix[0, 2]) / camera_matrix[0, 0],
        (pixel[:, 1] - camera_matrix[1, 2]) / camera_matrix[1, 1],
    ]
    ideal = inverse_simple_radial(distorted, float(radial_k1))
    ray_camera = np.c_[ideal, np.ones(len(ideal))]
    ray_world = ray_camera @ pose[:3, :3]
    center = -pose[:3, :3].T @ pose[:3, 3]
    normal = planar_map.normals_world[plane]
    denominator = np.einsum("ij,ij->i", normal, ray_world)
    distance = np.einsum("ij,ij->i", normal, hard - center) / denominator
    continuous = center + distance[:, None] * ray_world
    delta = continuous - hard
    tangent = np.stack((
        np.einsum("ij,ij->i", delta, planar_map.frames_world[plane, 0]),
        np.einsum("ij,ij->i", delta, planar_map.frames_world[plane, 1]),
    ), axis=1)
    valid = (
        np.isfinite(continuous).all(axis=1) & np.isfinite(tangent).all(axis=1)
        & (distance > 0.0)
        & (np.abs(denominator) > 1e-8)
        & np.all(np.abs(tangent) <= float(maximum_tangent_shift_m), axis=1)
    )
    return continuous, valid, tangent


def _solve_rows(
    world: np.ndarray,
    token: np.ndarray,
    pixel: np.ndarray,
    camera_matrix: np.ndarray,
    radial_k1: float,
) -> np.ndarray | None:
    return _solve(
        np.asarray(world, np.float64), np.asarray(token, np.int64), camera_matrix,
        radial_k1, np.arange(len(world), dtype=np.int64), np.asarray(pixel, np.float64),
    )


def _summary(rows: list[dict[str, object]], stage: str) -> dict[str, object]:
    translation = np.asarray([row[stage]["translation_error_m"] for row in rows], np.float64)
    rotation = np.asarray([row[stage]["rotation_error_deg"] for row in rows], np.float64)
    finite = np.isfinite(translation) & np.isfinite(rotation)
    return {
        "usable_count": int(np.sum(finite)),
        "median_translation_m": None if not np.any(finite) else float(np.median(translation[finite])),
        "median_rotation_deg": None if not np.any(finite) else float(np.median(rotation[finite])),
        "threshold_hit_counts": {
            key: int(np.sum(finite & (translation <= t) & (rotation <= r)))
            for key, t, r in THRESHOLDS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen_correspondences", type=Path, required=True)
    parser.add_argument("--selected_pose_inventory", type=Path, required=True)
    parser.add_argument("--planar_map", type=Path, required=True)
    parser.add_argument("--query_contributors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite continuous-coordinate oracle")
    corr, corr_meta = _load_correspondences(args.frozen_correspondences)
    selected, selected_meta = _load_pose_inventory(args.selected_pose_inventory)
    planes = GeometryNativePlanarMap.load_npz(args.planar_map)
    names = corr["names"].astype(str)
    if not np.array_equal(names, selected["names"].astype(str)):
        raise ValueError("oracle pose and correspondence inventories differ")

    output_rows: list[dict[str, object]] = []
    for query_index, name in enumerate(names.tolist()):
        with np.load(args.query_contributors / name, allow_pickle=False) as data:
            target = np.asarray(data["pose_w2c"], np.float64)
        lo, hi = map(int, corr["correspondence_offsets"][query_index:query_index + 2])
        world = np.asarray(corr["world_points"][lo:hi], np.float64)
        token = np.asarray(corr["query_tokens"][lo:hi], np.int64)
        provenance = np.asarray(corr["provenance_region_plane_atlas_row"][lo:hi], np.int64)
        K = np.asarray(corr["camera_matrices"][query_index], np.float64)
        k1 = float(corr["radial_k1"][query_index])
        center_pixel = _token_pixels(token, tuple(corr_meta.get("token_grid", (36, 64))))
        oracle_rows, projected, _ = _oracle_candidate_rows(
            target, world, token, center_pixel, K, k1,
        )
        hard_pose = _solve_rows(
            world[oracle_rows], token[oracle_rows], center_pixel[oracle_rows], K, k1,
        )
        hard_t, hard_r = _pose_error(hard_pose, target)

        inside = _inside_token(projected[oracle_rows], token[oracle_rows])
        sub_rows = oracle_rows[inside]
        subtoken_pose = _solve_rows(
            world[sub_rows], token[sub_rows], projected[sub_rows], K, k1,
        )
        sub_t, sub_r = _pose_error(subtoken_pose, target)

        continuous_result: dict[str, dict[str, float | int]] = {}
        for radius in (0.25, 0.50):
            continuous, valid, tangent = _local_continuous_surface_points(
                target, center_pixel[oracle_rows], world[oracle_rows],
                provenance[oracle_rows, 1], K, k1, planes,
                maximum_tangent_shift_m=radius,
            )
            pose = _solve_rows(
                continuous[valid], token[oracle_rows][valid], center_pixel[oracle_rows][valid], K, k1,
            )
            translation, rotation = _pose_error(pose, target)
            continuous_result[f"continuous_uv_radius_{str(radius).replace('.', 'p')}m"] = {
                "translation_error_m": translation, "rotation_error_deg": rotation,
                "correspondence_count": int(np.sum(valid)),
                "tangent_shift_median_m": (
                    float(np.median(np.linalg.norm(tangent[valid], axis=1)))
                    if np.any(valid) else float("inf")
                ),
            }
        deployed_t, deployed_r = _pose_error(selected["pose_w2c"][query_index], target)
        output_rows.append({
            "name": name,
            "deployed": {"translation_error_m": deployed_t, "rotation_error_deg": deployed_r},
            "candidate_oracle_token_center": {
                "translation_error_m": hard_t, "rotation_error_deg": hard_r,
                "correspondence_count": int(len(oracle_rows)),
            },
            "candidate_oracle_subtoken": {
                "translation_error_m": sub_t, "rotation_error_deg": sub_r,
                "correspondence_count": int(len(sub_rows)),
            },
            **continuous_result,
        })

    stages = (
        "deployed", "candidate_oracle_token_center", "candidate_oracle_subtoken",
        "continuous_uv_radius_0p25m", "continuous_uv_radius_0p5m",
    )
    report = {
        "artifact_type": "goal_maplet_continuous_coordinate_pose_postlabel_oracle_v1",
        "evaluation_role": "POSTLABEL_DIAGNOSTIC_ONLY_NOT_DEPLOYABLE",
        "query_pose_or_ground_truth_read": True,
        "selection_or_training_eligible": False,
        "candidate_oracle": "minimum_GT_reprojection_existing_3D_hypothesis_per_query_token_within_4px",
        "subtoken_oracle": "exact_GT_projection_of_selected_existing_3D_hypothesis_if_inside_original_4x4_token",
        "continuous_uv_oracle": "GT_ray_intersection_with_local_tangent_plane_through_selected_hard_point",
        "continuous_uv_bounds_m": [0.25, 0.5],
        "frozen_correspondence_file_sha256": file_sha256(args.frozen_correspondences),
        "frozen_correspondence_content_sha256": corr_meta.get("content_sha256"),
        "selected_pose_inventory_file_sha256": file_sha256(args.selected_pose_inventory),
        "selected_pose_inventory_content_sha256": selected_meta.get("content_sha256"),
        "planar_map_file_sha256": file_sha256(args.planar_map),
        "query_count": int(len(names)),
        "stage_summary": {stage: _summary(output_rows, stage) for stage in stages},
        "rows": output_rows,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
