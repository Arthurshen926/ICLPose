"""Evaluate Goal-Maplet geometry, support, parent, child, and local pose oracles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField, readout_canonical_field
from feature_extract.vfm.localization_goal_maplet.oracle_pose import (
    grouped_oracle_correspondences,
    intersect_rays_with_primitive_planes,
    token_oracle_evidence,
)
from feature_extract.vfm.localization_goal_maplet.pfir import ContributorLabels, _primitive_to_maplet_links
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.query_support import all_token_coordinates, group_tokens_after_retrieval
from feature_extract.vfm.localization_goal_maplet.retrieval import ValidityCalibration, retrieve_maplet_posterior
from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion, pnp_pose_error
from feature_extract.vfm.surface_maplet_bank import RadioFinalRegionConfig, encode_radio_final_regions


def _camera(path: Path) -> ColmapCamera:
    with np.load(path, allow_pickle=False) as data:
        return ColmapCamera(
            camera_id=0,
            model_id=int(data["camera_model_id"]),
            width=int(data["camera_width"]),
            height=int(data["camera_height"]),
            params=tuple(np.asarray(data["camera_params"], dtype=np.float64).tolist()),
        )


def _solve(xy: np.ndarray, xyz: np.ndarray, camera: ColmapCamera, gt_pose: np.ndarray) -> dict[str, object]:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    unique_count = int(np.unique(np.round(xyz, decimals=6), axis=0).shape[0]) if xyz.size else 0
    if len(xyz) < 6 or unique_count < 6:
        return {"success": False, "correspondence_count": len(xyz), "unique_xyz_count": unique_count,
                "inlier_count": 0, "translation_m": None, "rotation_deg": None,
                "gt_reprojection_median_px": None}
    matrix, distortion = camera_matrix_and_distortion(camera)
    gt_rotation, _ = cv2.Rodrigues(np.asarray(gt_pose[:3, :3], dtype=np.float64))
    projected, _ = cv2.projectPoints(xyz, gt_rotation, gt_pose[:3, 3], matrix, distortion)
    gt_reprojection = np.linalg.norm(projected.reshape(-1, 2) - xy, axis=1)
    cv2.setRNGSeed(194917)
    success, rotation, translation, inliers = cv2.solvePnPRansac(
        xyz,
        xy,
        matrix,
        distortion,
        iterationsCount=4000,
        reprojectionError=12.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success:
        return {"success": False, "correspondence_count": len(xyz), "unique_xyz_count": unique_count,
                "inlier_count": 0, "translation_m": None, "rotation_deg": None,
                "gt_reprojection_median_px": float(np.median(gt_reprojection))}
    if inliers is not None and len(inliers) >= 6:
        rows = np.asarray(inliers, dtype=np.int64).reshape(-1)
        rotation, translation = cv2.solvePnPRefineLM(xyz[rows], xy[rows], matrix, distortion, rotation, translation)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = cv2.Rodrigues(rotation)[0]
    pose[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    error = pnp_pose_error(pose, gt_pose)
    return {
        "success": True,
        "correspondence_count": len(xyz),
        "unique_xyz_count": unique_count,
        "inlier_count": int(len(inliers)) if inliers is not None else 0,
        "translation_m": float(error.translation_m),
        "rotation_deg": float(error.rotation_deg),
        "gt_reprojection_median_px": float(np.median(gt_reprojection)),
    }


def _pixel_oracle(
    labels: ContributorLabels,
    physical: GoalMapletPhysicalMap,
    camera: ColmapCamera,
    *,
    owned_only: bool,
    surface_intersection: bool,
    stride: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    primitive_row = {int(value): row for row, value in enumerate(physical.primitive_ids.tolist())}
    link_offsets, _, _ = _primitive_to_maplet_links(physical)
    xy, xyz, selected_rows = [], [], []
    height, width, topk = labels.topk_primitive_ids.shape
    for y in range(stride // 2, height, stride):
        for x in range(stride // 2, width, stride):
            order = np.argsort(-labels.topk_weights[y, x], kind="stable")
            selected = None
            for slot in order.tolist():
                row = primitive_row.get(int(labels.topk_primitive_ids[y, x, slot]))
                if row is None or float(labels.topk_weights[y, x, slot]) <= 0.0:
                    continue
                if owned_only and int(link_offsets[row + 1]) <= int(link_offsets[row]):
                    continue
                selected = row
                break
            if selected is not None:
                xy.append([(x + 0.5) * camera.width / width, (y + 0.5) * camera.height / height])
                xyz.append(physical.primitive_centers[selected])
                selected_rows.append(int(selected))
    output_xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    output_xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if surface_intersection and output_xyz.size:
        rows = np.asarray(selected_rows, dtype=np.int64)
        point, valid = intersect_rays_with_primitive_planes(output_xy, rows, physical, labels.pose_w2c, camera)
        output_xy, output_xyz = output_xy[valid], point[valid]
    return output_xy, output_xyz


def _summarize(rows: list[dict[str, object]], names: list[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in names:
        values = [row["oracles"][name] for row in rows]
        translation = [float(value["translation_m"]) for value in values if value.get("translation_m") is not None]
        rotation = [float(value["rotation_deg"]) for value in values if value.get("rotation_deg") is not None]
        result[name] = {
            "success_fraction": float(np.mean([bool(value["success"]) for value in values])) if values else 0.0,
            "translation_median_m": float(np.median(translation)) if translation else None,
            "translation_p90_m": float(np.percentile(translation, 90.0)) if translation else None,
            "rotation_median_deg": float(np.median(rotation)) if rotation else None,
            "rotation_p90_deg": float(np.percentile(rotation, 90.0)) if rotation else None,
            "within_0.25m_5deg": float(np.mean([
                value.get("translation_m") is not None and float(value["translation_m"]) <= 0.25
                and float(value["rotation_deg"]) <= 5.0 for value in values
            ])) if values else 0.0,
            "within_0.5m_10deg": float(np.mean([
                value.get("translation_m") is not None and float(value["translation_m"]) <= 0.5
                and float(value["rotation_deg"]) <= 10.0 for value in values
            ])) if values else 0.0,
            "gt_reprojection_median_px": float(np.median([
                float(value["gt_reprojection_median_px"]) for value in values
                if value.get("gt_reprojection_median_px") is not None
            ])) if any(value.get("gt_reprojection_median_px") is not None for value in values) else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--validity_calibration", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--grouping_cosine", type=float, default=0.90)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--maximum_queries", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite Goal-Maplet oracle report")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    readout = readout_canonical_field(field, physical)
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(args.device))
    calibration = ValidityCalibration.load_json(Path(args.validity_calibration))
    paths = sorted(Path(args.contributors).glob("*.npz"))[int(args.shard_index) :: int(args.shard_count)]
    if int(args.maximum_queries) > 0:
        paths = paths[: int(args.maximum_queries)]
    config = RadioFinalRegionConfig()
    rows = []
    oracle_names: list[str] | None = None
    for path in paths:
        labels = ContributorLabels.load_npz(path)
        camera = _camera(path)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        with np.load(Path(str(metadata["token_path"])), allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        mapped = mapper.project(raw).measurement_context
        _, token_xy = all_token_coordinates(int(raw.shape[1]), int(raw.shape[2]))
        descriptor = encode_radio_final_regions(mapped, token_xy, config)
        candidate_ids, _, _, _ = retrieve_maplet_posterior(
            descriptor,
            readout.parent_descriptors,
            physical.maplet_ids,
            readout.parent_coverage > 0.0,
            maximum_candidates=64,
            temperature=0.07,
            null_similarity_center=float(calibration.center),
            null_similarity_scale=float(calibration.scale),
        )
        evidence = token_oracle_evidence(
            labels,
            physical,
            token_xy,
            token_height=int(raw.shape[1]),
            token_width=int(raw.shape[2]),
            image_height=camera.height,
            image_width=camera.width,
            camera=camera,
        )
        identity_by_token = np.full((token_xy.shape[0],), -1, dtype=np.int64)
        valid_parent = evidence.parent_rows >= 0
        identity_by_token[valid_parent] = physical.maplet_ids[evidence.parent_rows[valid_parent]]
        grouping = {}
        for name, identity in (("oracle_group", identity_by_token), ("current_group", candidate_ids[:, 0])):
            grouped = group_tokens_after_retrieval(
                token_xy,
                descriptor,
                identity,
                token_height=int(raw.shape[1]),
                token_width=int(raw.shape[2]),
                image_width=camera.width,
                image_height=camera.height,
                descriptor_half_size_tokens=2.0,
                minimum_descriptor_cosine=float(args.grouping_cosine),
            )
            grouping[name] = (grouped, grouped_oracle_correspondences(
                evidence, physical, grouped.member_offsets, grouped.member_token_indices
            ))
        oracles: dict[str, object] = {}
        clean_center_xy, clean_center_xyz = _pixel_oracle(
            labels, physical, camera, owned_only=False, surface_intersection=False
        )
        owned_center_xy, owned_center_xyz = _pixel_oracle(
            labels, physical, camera, owned_only=True, surface_intersection=False
        )
        clean_surface_xy, clean_surface_xyz = _pixel_oracle(
            labels, physical, camera, owned_only=False, surface_intersection=True
        )
        owned_surface_xy, owned_surface_xyz = _pixel_oracle(
            labels, physical, camera, owned_only=True, surface_intersection=True
        )
        oracles["o0_clean_pixel_center_geometry"] = _solve(
            clean_center_xy, clean_center_xyz, camera, labels.pose_w2c
        )
        oracles["o0_clean_pixel_surface_geometry"] = _solve(
            clean_surface_xy, clean_surface_xyz, camera, labels.pose_w2c
        )
        oracles["o1_owned_pixel_center_geometry"] = _solve(
            owned_center_xy, owned_center_xyz, camera, labels.pose_w2c
        )
        oracles["o1_owned_pixel_surface_geometry"] = _solve(
            owned_surface_xy, owned_surface_xyz, camera, labels.pose_w2c
        )
        for group_name, (_, correspondence) in grouping.items():
            for layer_name, (xy, xyz) in correspondence.items():
                oracles[f"{group_name}_{layer_name}"] = _solve(xy, xyz, camera, labels.pose_w2c)
        oracle_names = list(oracles)
        row = {
            "image_id": str(metadata["image_id"]),
            "oracle_group_count": int(grouping["oracle_group"][0].member_offsets.size - 1),
            "current_group_count": int(grouping["current_group"][0].member_offsets.size - 1),
            "oracles": oracles,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "stage": "goal_maplet_oracle_ladder_o0_o3",
        "query_count": len(rows),
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "validity_calibration_sha256": calibration.content_sha256,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "summary": _summarize(rows, oracle_names or []),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
