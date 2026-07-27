"""Build a RADIO feature field at detector-repeatable 2DGS surface locations.

ALIKE contributes only sub-pixel detection coordinates and detector scores.
Its descriptors are never read, fused, matched, or persisted.  Mapping depth
and poses are offline construction inputs; the resulting map contains only one
RADIO-final distribution per retained clean 2DGS surfel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import cv2
from plyfile import PlyData
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from feature_extract.tools.vfm.build_2dgs_surface_feature_field import (
    _clean_source_indices,
    _maplet_owners,
)
from feature_extract.tools.vfm.localize_2dgs_surface_queries import (
    _load_query_camera_manifest,
    _load_raw_final,
    _sample_mapped_vfm_at_pixels,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.gaussian_vfm_field import load_gaussian_vfm_source_from_ply
from feature_extract.vfm.localization.surface_feature_field import SurfaceFeatureField
from feature_extract.vfm.localization.surface_metric_feature_mapper import (
    load_surface_metric_feature_mapper,
)
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.rendered_keypoint_matching import backproject_depth_to_world
from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion
from feature_extract.vfm.surface_maplet_bank import VfmSurfaceMapletBank
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.vfm_2dgs_mapping import _surface_tangent_axes_and_scales


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaussian_ply", required=True)
    parser.add_argument("--clean_gaussian_ply", required=True)
    parser.add_argument("--mapping_manifest", required=True)
    parser.add_argument("--mapping_pose_file", required=True)
    parser.add_argument("--mapping_camera_manifest", required=True)
    parser.add_argument("--mapping_depth_bank", required=True)
    parser.add_argument("--alike_detection_cache", required=True)
    parser.add_argument("--surface_mapper_checkpoint", required=True)
    parser.add_argument("--metric_mapper_checkpoint", default="")
    parser.add_argument("--maplets", required=True)
    parser.add_argument("--output_field", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument(
        "--training_samples",
        default="",
        help="Optional offline-only RADIO/source sample cache for a metric head.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detections_per_view", type=int, default=256)
    parser.add_argument("--minimum_observations", type=int, default=2)
    parser.add_argument("--minimum_coherence", type=float, default=0.55)
    parser.add_argument("--maximum_surface_snap_m", type=float, default=0.15)
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _sample_depth(depth: np.ndarray, xy: np.ndarray) -> np.ndarray:
    image = np.asarray(depth, dtype=np.float32)
    points = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    x = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, image.shape[1] - 1)
    y = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, image.shape[0] - 1)
    return image[y, x]


def _safe_cache_name(image_id: str) -> str:
    return str(image_id).replace("/", "__") + ".npz"


def _intersect_assigned_surfel_planes(
    xy: np.ndarray,
    source_ids: np.ndarray,
    *,
    camera,
    pose_w2c: np.ndarray,
    source,
) -> tuple[np.ndarray, np.ndarray]:
    matrix, distortion = camera_matrix_and_distortion(camera)
    normalized = cv2.undistortPoints(
        np.asarray(xy, dtype=np.float64).reshape(-1, 1, 2),
        matrix,
        distortion,
    ).reshape(-1, 2)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    origin = -rotation.T @ translation
    ray_camera = np.concatenate(
        [normalized, np.ones((normalized.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    rays = ray_camera @ rotation
    rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
    plane_centers = np.asarray(source.xyz, dtype=np.float64)[source_ids]
    plane_normals = np.asarray(source.normal, dtype=np.float64)[source_ids]
    denominator = np.sum(plane_normals * rays, axis=1)
    distance = np.sum(plane_normals * (plane_centers - origin), axis=1) / np.where(
        np.abs(denominator) >= 1e-6,
        denominator,
        np.where(denominator < 0.0, -1e-6, 1e-6),
    )
    xyz = origin[None] + distance[:, None] * rays
    valid = (
        np.isfinite(xyz).all(axis=1)
        & (distance > 0.0)
        & (np.abs(denominator) >= 1e-4)
    )
    return xyz, valid


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = Path(args.output_field)
    summary_path = Path(args.summary_json)
    if (output_path.exists() or summary_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite detector surface field outputs")

    clean_indices = _clean_source_indices(Path(args.clean_gaussian_ply))
    source = load_gaussian_vfm_source_from_ply(Path(args.gaussian_ply))
    clean_centers = np.asarray(source.xyz, dtype=np.float64)[clean_indices]
    clean_tree = cKDTree(clean_centers)
    clean_row_by_source = np.full((source.xyz.shape[0],), -1, dtype=np.int64)
    clean_row_by_source[clean_indices] = np.arange(clean_indices.size, dtype=np.int64)
    maplets = VfmSurfaceMapletBank.load_npz(Path(args.maplets))
    mapper, _mapper_metadata = load_surface_maplet_mapper(
        Path(args.surface_mapper_checkpoint), device=str(args.device)
    )
    manifest = TokenBankManifest.from_json(Path(args.mapping_manifest))
    manifest.validate()
    pose_by_image = {
        record.image_id: record.pose_w2c
        for record in parse_cambridge_pose_file(Path(args.mapping_pose_file))
    }
    camera_by_image, camera_audit = _load_query_camera_manifest(
        Path(args.mapping_camera_manifest)
    )
    depth_payload = json.loads(Path(args.mapping_depth_bank).read_text())
    depth_by_image = {
        str(record["image_id"]): Path(record["path"])
        for record in depth_payload["records"]
    }
    records = [
        record
        for record in manifest.records
        if record.image_id in pose_by_image
        and record.image_id in camera_by_image
        and record.image_id in depth_by_image
        and (
            Path(args.alike_detection_cache) / _safe_cache_name(record.image_id)
        ).is_file()
    ]
    if int(args.max_views) > 0 and len(records) > int(args.max_views):
        selection = np.linspace(
            0, len(records) - 1, int(args.max_views), dtype=np.int64
        )
        records = [records[int(row)] for row in selection.tolist()]

    source_rows: list[np.ndarray] = []
    descriptors: list[np.ndarray] = []
    detector_weights: list[np.ndarray] = []
    lifted_positions: list[np.ndarray] = []
    sample_view_rows: list[np.ndarray] = []
    accepted_per_view: list[int] = []
    for view_index, record in enumerate(records):
        cache_path = Path(args.alike_detection_cache) / _safe_cache_name(
            record.image_id
        )
        # Intentionally access only detection outputs, never `descriptors`.
        with np.load(cache_path, allow_pickle=False) as cache:
            xy = np.asarray(cache["xy"], dtype=np.float32)
            scores = np.asarray(cache["scores"], dtype=np.float32)
        order = np.argsort(-scores, kind="mergesort")[
            : int(args.detections_per_view)
        ]
        xy = xy[order]
        scores = scores[order]
        depth = np.load(depth_by_image[record.image_id], mmap_mode="r")
        depth_values = _sample_depth(depth, xy)
        world, valid = backproject_depth_to_world(
            xy,
            depth_values,
            camera_by_image[record.image_id],
            pose_by_image[record.image_id],
        )
        valid &= np.isfinite(scores)
        if not np.any(valid):
            accepted_per_view.append(0)
            continue
        xy = xy[valid]
        scores = scores[valid]
        world = world[valid]
        distances, clean_rows = clean_tree.query(world, k=1)
        valid_snap = distances <= float(args.maximum_surface_snap_m)
        if not np.any(valid_snap):
            accepted_per_view.append(0)
            continue
        xy = xy[valid_snap]
        scores = scores[valid_snap]
        clean_rows = np.asarray(clean_rows[valid_snap], dtype=np.int64)
        depth_world = world[valid_snap]
        source_ids = clean_indices[clean_rows]
        plane_world, plane_valid = _intersect_assigned_surfel_planes(
            xy,
            source_ids,
            camera=camera_by_image[record.image_id],
            pose_w2c=pose_by_image[record.image_id],
            source=source,
        )
        plane_valid &= (
            np.linalg.norm(plane_world - depth_world, axis=1)
            <= max(2.0 * float(args.maximum_surface_snap_m), 0.10)
        )
        if not np.any(plane_valid):
            accepted_per_view.append(0)
            continue
        xy = xy[plane_valid]
        scores = scores[plane_valid]
        clean_rows = clean_rows[plane_valid]
        plane_world = plane_world[plane_valid]
        raw = _load_raw_final(Path(record.token_path), "radio_final")
        mapped = mapper.project(raw).measurement_context
        feature = _sample_mapped_vfm_at_pixels(
            mapped,
            xy,
            image_width=int(camera_by_image[record.image_id].width),
            image_height=int(camera_by_image[record.image_id].height),
        )
        # At most one observation of a surfel from one image: support count is
        # therefore a conservative multi-view repeatability count.
        local_order = np.argsort(-scores, kind="mergesort")
        _unique, first = np.unique(clean_rows[local_order], return_index=True)
        keep = local_order[np.sort(first)]
        source_rows.append(clean_rows[keep])
        descriptors.append(feature[keep])
        detector_weights.append(np.maximum(scores[keep], 1e-6))
        lifted_positions.append(plane_world[keep])
        sample_view_rows.append(
            np.full((keep.size,), int(view_index), dtype=np.int32)
        )
        accepted_per_view.append(int(keep.size))
        if (view_index + 1) % 100 == 0:
            print(
                f"processed {view_index + 1}/{len(records)} mapping views; "
                f"accepted {sum(accepted_per_view)} detector surface samples"
            )

    if not source_rows:
        raise ValueError("no detector surface samples survived geometry lifting")
    row_values = np.concatenate(source_rows).astype(np.int64, copy=False)
    feature_values = np.concatenate(descriptors).astype(np.float32, copy=False)
    weight_values = np.concatenate(detector_weights).astype(np.float32, copy=False)
    position_values = np.concatenate(lifted_positions).astype(np.float64, copy=False)
    observation_ids = np.arange(row_values.size, dtype=np.int64)
    view_values = np.concatenate(sample_view_rows).astype(np.int32, copy=False)
    if str(args.training_samples):
        training_path = Path(args.training_samples)
        training_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            training_path,
            source_indices=clean_indices[row_values].astype(np.int64),
            radio_features=feature_values.astype(np.float16),
            detector_weights=weight_values.astype(np.float16),
            view_indices=view_values,
            metadata_json=np.asarray(
                json.dumps(
                    {
                        "artifact_type": "offline_detector_radio_metric_samples",
                        "vfm_layer": "radio_final",
                        "uses_alike_descriptors": False,
                        "uses_radio_intermediate": False,
                        "uses_sfm_points": False,
                        "uses_sfm_tracks": False,
                    },
                    sort_keys=True,
                )
            ),
        )
    if str(args.metric_mapper_checkpoint):
        metric_mapper = load_surface_metric_feature_mapper(
            Path(args.metric_mapper_checkpoint), device=str(args.device)
        )
        feature_values = metric_mapper.project_points(feature_values)
    matrix = coo_matrix(
        (weight_values, (row_values, observation_ids)),
        shape=(clean_indices.size, row_values.size),
        dtype=np.float32,
    ).tocsr()
    feature_sum = matrix @ feature_values
    position_sum = matrix @ position_values
    weight_sum = np.asarray(matrix.sum(axis=1), dtype=np.float32).reshape(-1)
    observation_count = np.asarray(matrix.getnnz(axis=1), dtype=np.int32)
    mean = feature_sum / np.maximum(weight_sum[:, None], 1e-8)
    coherence = np.linalg.norm(mean, axis=1).astype(np.float32)
    mean /= np.maximum(coherence[:, None], 1e-8)
    retained = (
        (observation_count >= int(args.minimum_observations))
        & (coherence >= float(args.minimum_coherence))
    )
    rows = clean_indices[retained]
    measured_centers = position_sum[retained] / np.maximum(
        weight_sum[retained, None], 1e-8
    )
    primitive_centers = np.asarray(source.xyz, dtype=np.float64)[rows]
    normals = np.asarray(source.normal, dtype=np.float32)[rows]
    # Preserve the detector's continuous tangential surface coordinate while
    # removing only normal-direction depth noise against the clean surfel plane.
    normal_offset = np.sum(
        (measured_centers - primitive_centers) * normals, axis=1
    )
    centers = measured_centers - normal_offset[:, None] * normals
    tangent1, tangent2, scale1, scale2 = _surface_tangent_axes_and_scales(
        source, rows, normals
    )
    owner = _maplet_owners(rows, centers, maplets)
    retained_count = observation_count[retained]
    count_scale = float(np.quantile(retained_count, 0.90)) if rows.size else 1.0
    confidence = (
        np.clip(retained_count / max(count_scale, 1.0), 0.0, 1.0)
        * coherence[retained]
        * np.asarray(source.opacity, dtype=np.float32)[rows]
    )
    metadata = {
        "artifact_type": "radio_final_2dgs_surface_feature_field",
        "vfm_layer": "radio_final",
        "representation": "one_distribution_per_detector_repeatable_clean_2dgs_surfel",
        "surface_coordinate": "depth_assigned_camera_ray_clean_surfel_plane_intersection_mean",
        "feature_dim": int(feature_values.shape[1]),
        "clean_primitive_count": int(clean_indices.size),
        "field_surfel_count": int(rows.size),
        "mapping_view_count": int(len(records)),
        "alike_role": "detection_coordinates_and_weights_only",
        "stores_mapping_rgb": False,
        "stores_mapping_image_paths": False,
        "uses_mapping_rgb_at_inference": False,
        "uses_pairwise_image_matching": False,
        "uses_alike_descriptors": False,
        "uses_radio_intermediate": False,
        "uses_sfm_points": False,
        "uses_sfm_tracks": False,
        "uses_stable_anchor_identity": False,
        "uses_surface_metric_mapper": bool(args.metric_mapper_checkpoint),
    }
    field = SurfaceFeatureField(
        source_indices=rows,
        centers=centers,
        normals=normals,
        tangent1=tangent1,
        tangent2=tangent2,
        scale1=scale1,
        scale2=scale2,
        opacity=np.asarray(source.opacity, dtype=np.float32)[rows],
        features=mean[retained],
        uncertainty=np.clip(1.0 - coherence[retained], 0.0, 1.0),
        confidence=confidence,
        support_weight=weight_sum[retained],
        support_count=retained_count,
        owner_maplet_ids=owner,
        metadata=metadata,
    )
    field.save_npz(output_path)
    summary = {
        "stage": "build_detector_weighted_anchor_free_2dgs_surface_field",
        "field_surfel_count": len(field),
        "mapping_view_count": len(records),
        "lifted_detection_count": int(row_values.size),
        "multi_view_clean_surfel_count": int(np.sum(observation_count >= 2)),
        "accepted_detections_per_view": {
            "median": float(np.median(accepted_per_view)),
            "p10": float(np.quantile(accepted_per_view, 0.10)),
            "p90": float(np.quantile(accepted_per_view, 0.90)),
        },
        "coherence": {
            "median": float(np.median(coherence[retained])),
            "p10": float(np.quantile(coherence[retained], 0.10)),
            "p90": float(np.quantile(coherence[retained], 0.90)),
        },
        "camera_audit": camera_audit,
        "production_contract": metadata,
        "output_field": str(output_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
