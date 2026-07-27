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
from feature_extract.vfm.localization.highres_surface_metric_decoder import (
    decode_highres_surface_metric,
    load_highres_surface_metric_decoder,
)
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
    parser.add_argument("--highres_metric_decoder_checkpoint", default="")
    parser.add_argument(
        "--mapping_image_root",
        default="",
        help="Offline RGB root, required only while baking high-resolution features.",
    )
    parser.add_argument(
        "--feature_branch",
        choices=(
            "raw_radio_final",
            "retrieval_mapped",
            "metric_mapped",
            "highres_metric_decoded",
        ),
        default="metric_mapped",
    )
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
    parser.add_argument(
        "--sampling_mode",
        choices=("detector", "dense_radio_grid", "hybrid"),
        default="dense_radio_grid",
    )
    parser.add_argument(
        "--dense_sample_weight",
        type=float,
        default=0.35,
        help="Observation weight for dense RADIO-grid samples in hybrid/dense mode.",
    )
    parser.add_argument("--minimum_observations", type=int, default=2)
    parser.add_argument("--minimum_coherence", type=float, default=0.55)
    parser.add_argument(
        "--maximum_surface_snap_m",
        type=float,
        default=0.30,
        help="Candidate-center search radius only; acceptance uses ray-disk geometry.",
    )
    parser.add_argument("--surface_candidate_count", type=int, default=16)
    parser.add_argument("--maximum_plane_depth_residual_m", type=float, default=0.03)
    parser.add_argument("--maximum_disk_sigma", type=float, default=3.0)
    parser.add_argument("--minimum_ray_normal_cosine", type=float, default=0.05)
    parser.add_argument("--tangent_grid_size", type=int, default=8)
    parser.add_argument("--tangent_extent_sigma", type=float, default=3.0)
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


def _camera_rays(
    xy: np.ndarray,
    *,
    camera,
    pose_w2c: np.ndarray,
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
    return origin, rays


def _assign_ray_disk_candidates(
    xy: np.ndarray,
    depth_world: np.ndarray,
    *,
    clean_tree: cKDTree,
    clean_indices: np.ndarray,
    clean_centers: np.ndarray,
    clean_normals: np.ndarray,
    clean_tangent1: np.ndarray,
    clean_tangent2: np.ndarray,
    clean_scale1: np.ndarray,
    clean_scale2: np.ndarray,
    clean_opacity: np.ndarray,
    camera,
    pose_w2c: np.ndarray,
    candidate_count: int,
    maximum_candidate_center_m: float,
    maximum_plane_depth_residual_m: float,
    maximum_disk_sigma: float,
    minimum_ray_normal_cosine: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Assign depth samples by exact camera-ray/2DGS-disk compatibility."""

    del clean_indices  # Rows are already clean-source indexed.
    points = np.asarray(depth_world, dtype=np.float64).reshape(-1, 3)
    count = min(max(int(candidate_count), 1), int(clean_centers.shape[0]))
    distances, candidates = clean_tree.query(
        points,
        k=count,
        distance_upper_bound=float(maximum_candidate_center_m),
    )
    distances = np.asarray(distances, dtype=np.float64).reshape(points.shape[0], count)
    candidates = np.asarray(candidates, dtype=np.int64).reshape(points.shape[0], count)
    candidate_valid = np.isfinite(distances) & (candidates < clean_centers.shape[0])
    safe_rows = np.where(candidate_valid, candidates, 0)
    origin, rays = _camera_rays(xy, camera=camera, pose_w2c=pose_w2c)
    centers = clean_centers[safe_rows]
    normals = clean_normals[safe_rows]
    denominator = np.sum(normals * rays[:, None, :], axis=2)
    ray_distance = np.sum(normals * (centers - origin[None, None, :]), axis=2)
    ray_distance /= np.where(
        np.abs(denominator) >= 1e-8,
        denominator,
        np.where(denominator < 0.0, -1e-8, 1e-8),
    )
    intersections = origin[None, None, :] + ray_distance[..., None] * rays[:, None, :]
    relative = intersections - centers
    local_u = np.sum(relative * clean_tangent1[safe_rows], axis=2) / np.maximum(
        clean_scale1[safe_rows], 1e-6
    )
    local_v = np.sum(relative * clean_tangent2[safe_rows], axis=2) / np.maximum(
        clean_scale2[safe_rows], 1e-6
    )
    disk_radius2 = local_u * local_u + local_v * local_v
    depth_residual = np.linalg.norm(intersections - points[:, None, :], axis=2)
    incidence = np.abs(denominator)
    valid = (
        candidate_valid
        & np.isfinite(intersections).all(axis=2)
        & (ray_distance > 0.0)
        & (depth_residual <= float(maximum_plane_depth_residual_m))
        & (disk_radius2 <= float(maximum_disk_sigma) ** 2)
        & (incidence >= float(minimum_ray_normal_cosine))
    )
    score = (
        depth_residual / max(float(maximum_plane_depth_residual_m), 1e-6)
        + 0.10 * disk_radius2
        - 0.05 * np.log(np.clip(clean_opacity[safe_rows], 1e-4, 1.0))
    )
    score[~valid] = np.inf
    best_column = np.argmin(score, axis=1)
    rows = candidates[np.arange(points.shape[0]), best_column]
    accepted = np.isfinite(score[np.arange(points.shape[0]), best_column])
    rows = np.where(accepted, rows, -1)
    xyz = intersections[np.arange(points.shape[0]), best_column]
    stats = {
        "candidate_search_rejected": int(np.sum(~np.any(candidate_valid, axis=1))),
        "ray_disk_rejected": int(np.sum(np.any(candidate_valid, axis=1) & ~accepted)),
        "accepted": int(np.sum(accepted)),
    }
    return rows.astype(np.int64), xyz, accepted, stats


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
    clean_normals = np.asarray(source.normal, dtype=np.float32)[clean_indices]
    (
        clean_tangent1,
        clean_tangent2,
        clean_scale1,
        clean_scale2,
    ) = _surface_tangent_axes_and_scales(source, clean_indices, clean_normals)
    clean_opacity = np.asarray(source.opacity, dtype=np.float32)[clean_indices]
    maplets = VfmSurfaceMapletBank.load_npz(Path(args.maplets))
    mapper, _mapper_metadata = load_surface_maplet_mapper(
        Path(args.surface_mapper_checkpoint), device=str(args.device)
    )
    metric_mapper = (
        load_surface_metric_feature_mapper(
            Path(args.metric_mapper_checkpoint), device=str(args.device)
        )
        if str(args.metric_mapper_checkpoint)
        else None
    )
    if str(args.feature_branch) == "metric_mapped" and metric_mapper is None:
        raise ValueError("metric_mapped feature branch requires a metric checkpoint")
    highres_decoder = (
        load_highres_surface_metric_decoder(
            Path(args.highres_metric_decoder_checkpoint), device=str(args.device)
        )[0]
        if str(args.highres_metric_decoder_checkpoint)
        else None
    )
    if str(args.feature_branch) == "highres_metric_decoded":
        if highres_decoder is None or not str(args.mapping_image_root):
            raise ValueError(
                "highres branch requires its decoder checkpoint and mapping image root"
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
            str(args.sampling_mode) == "dense_radio_grid"
            or (
                Path(args.alike_detection_cache) / _safe_cache_name(record.image_id)
            ).is_file()
        )
    ]
    if int(args.max_views) > 0 and len(records) > int(args.max_views):
        selection = np.linspace(
            0, len(records) - 1, int(args.max_views), dtype=np.int64
        )
        records = [records[int(row)] for row in selection.tolist()]

    source_rows: list[np.ndarray] = []
    surface_keys: list[np.ndarray] = []
    tangent_coordinates: list[np.ndarray] = []
    observation_pixels: list[np.ndarray] = []
    descriptors: list[np.ndarray] = []
    coarse_descriptors: list[np.ndarray] = []
    detector_weights: list[np.ndarray] = []
    lifted_positions: list[np.ndarray] = []
    sample_view_rows: list[np.ndarray] = []
    accepted_per_view: list[int] = []
    geometry_rejections = {
        "candidate_search_rejected": 0,
        "ray_disk_rejected": 0,
        "accepted": 0,
    }
    for view_index, record in enumerate(records):
        raw = _load_raw_final(Path(record.token_path), "radio_final")
        mapped = mapper.project(raw).measurement_context
        if str(args.feature_branch) == "raw_radio_final":
            decoded_map = raw
        elif str(args.feature_branch) == "retrieval_mapped":
            decoded_map = mapped
        elif str(args.feature_branch) == "metric_mapped":
            assert metric_mapper is not None
            decoded_map = metric_mapper.project_map(mapped)
        else:
            assert highres_decoder is not None
            bgr = cv2.imread(
                str(Path(args.mapping_image_root) / record.image_id),
                cv2.IMREAD_COLOR,
            )
            if bgr is None:
                raise ValueError(f"failed to decode mapping RGB for {record.image_id}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            decoded_map = decode_highres_surface_metric(
                highres_decoder,
                raw,
                rgb,
                device=str(args.device),
            )["fine"]
        xy_parts: list[np.ndarray] = []
        score_parts: list[np.ndarray] = []
        if str(args.sampling_mode) in {"detector", "hybrid"}:
            cache_path = Path(args.alike_detection_cache) / _safe_cache_name(
                record.image_id
            )
            # Intentionally access only detection outputs, never `descriptors`.
            with np.load(cache_path, allow_pickle=False) as cache:
                detector_xy = np.asarray(cache["xy"], dtype=np.float32)
                detector_scores = np.asarray(cache["scores"], dtype=np.float32)
            order = np.argsort(-detector_scores, kind="mergesort")[
                : int(args.detections_per_view)
            ]
            xy_parts.append(detector_xy[order])
            score_parts.append(detector_scores[order])
        if str(args.sampling_mode) in {"dense_radio_grid", "hybrid"}:
            camera = camera_by_image[record.image_id]
            if str(args.feature_branch) == "highres_metric_decoded":
                grid_x = (
                    (np.arange(int(decoded_map.shape[2]), dtype=np.float32) + 0.5)
                    * float(camera.width)
                    / int(decoded_map.shape[2])
                    - 0.5
                )
                grid_y = (
                    (np.arange(int(decoded_map.shape[1]), dtype=np.float32) + 0.5)
                    * float(camera.height)
                    / int(decoded_map.shape[1])
                    - 0.5
                )
            else:
                grid_x = np.linspace(
                    0.0,
                    float(camera.width - 1),
                    int(decoded_map.shape[2]),
                    dtype=np.float32,
                )
                grid_y = np.linspace(
                    0.0,
                    float(camera.height - 1),
                    int(decoded_map.shape[1]),
                    dtype=np.float32,
                )
            yy, xx = np.meshgrid(grid_y, grid_x, indexing="ij")
            dense_xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
            xy_parts.append(dense_xy)
            score_parts.append(
                np.full(
                    (dense_xy.shape[0],),
                    float(args.dense_sample_weight),
                    dtype=np.float32,
                )
            )
        xy = np.concatenate(xy_parts, axis=0)
        scores = np.concatenate(score_parts, axis=0)
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
        clean_rows, plane_world, plane_valid, geometry_stats = (
            _assign_ray_disk_candidates(
                xy,
                world,
                clean_tree=clean_tree,
                clean_indices=clean_indices,
                clean_centers=clean_centers,
                clean_normals=clean_normals,
                clean_tangent1=clean_tangent1,
                clean_tangent2=clean_tangent2,
                clean_scale1=clean_scale1,
                clean_scale2=clean_scale2,
                clean_opacity=clean_opacity,
                camera=camera_by_image[record.image_id],
                pose_w2c=pose_by_image[record.image_id],
                candidate_count=int(args.surface_candidate_count),
                maximum_candidate_center_m=float(args.maximum_surface_snap_m),
                maximum_plane_depth_residual_m=float(
                    args.maximum_plane_depth_residual_m
                ),
                maximum_disk_sigma=float(args.maximum_disk_sigma),
                minimum_ray_normal_cosine=float(args.minimum_ray_normal_cosine),
            )
        )
        for key in geometry_rejections:
            geometry_rejections[key] += int(geometry_stats[key])
        if not np.any(plane_valid):
            accepted_per_view.append(0)
            continue
        xy = xy[plane_valid]
        scores = scores[plane_valid]
        clean_rows = clean_rows[plane_valid]
        plane_world = plane_world[plane_valid]
        primitive_delta = plane_world - clean_centers[clean_rows]
        tangent_uv = np.stack(
            [
                np.sum(primitive_delta * clean_tangent1[clean_rows], axis=1)
                / np.maximum(clean_scale1[clean_rows], 1e-6),
                np.sum(primitive_delta * clean_tangent2[clean_rows], axis=1)
                / np.maximum(clean_scale2[clean_rows], 1e-6),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        grid_size = int(args.tangent_grid_size)
        extent = float(args.tangent_extent_sigma)
        if grid_size <= 0 or extent <= 0.0:
            raise ValueError("tangent grid size and extent must be positive")
        cell_xy = np.floor(
            (np.clip(tangent_uv, -extent, extent) + extent)
            / (2.0 * extent)
            * grid_size
        ).astype(np.int64)
        cell_xy = np.clip(cell_xy, 0, grid_size - 1)
        subcell = cell_xy[:, 1] * grid_size + cell_xy[:, 0]
        cell_keys = clean_rows * (grid_size * grid_size) + subcell
        coarse_feature = _sample_mapped_vfm_at_pixels(
            mapped,
            xy,
            image_width=int(camera_by_image[record.image_id].width),
            image_height=int(camera_by_image[record.image_id].height),
        )
        feature = _sample_mapped_vfm_at_pixels(
            decoded_map,
            xy,
            image_width=int(camera_by_image[record.image_id].width),
            image_height=int(camera_by_image[record.image_id].height),
            align_corners=str(args.feature_branch) != "highres_metric_decoded",
        )
        # At most one observation of a surfel from one image: support count is
        # therefore a conservative multi-view repeatability count.
        local_order = np.argsort(-scores, kind="mergesort")
        _unique, first = np.unique(cell_keys[local_order], return_index=True)
        keep = local_order[np.sort(first)]
        source_rows.append(clean_rows[keep])
        surface_keys.append(cell_keys[keep])
        tangent_coordinates.append(tangent_uv[keep])
        observation_pixels.append(xy[keep])
        descriptors.append(feature[keep])
        coarse_descriptors.append(coarse_feature[keep])
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
    clean_row_values = np.concatenate(source_rows).astype(np.int64, copy=False)
    cell_key_values = np.concatenate(surface_keys).astype(np.int64, copy=False)
    unique_cell_keys, row_values = np.unique(cell_key_values, return_inverse=True)
    cell_clean_rows = unique_cell_keys // (
        int(args.tangent_grid_size) * int(args.tangent_grid_size)
    )
    tangent_values = np.concatenate(tangent_coordinates).astype(
        np.float32, copy=False
    )
    pixel_values = np.concatenate(observation_pixels).astype(np.float32, copy=False)
    feature_values = np.concatenate(descriptors).astype(np.float32, copy=False)
    coarse_feature_values = np.concatenate(coarse_descriptors).astype(
        np.float32, copy=False
    )
    weight_values = np.concatenate(detector_weights).astype(np.float32, copy=False)
    position_values = np.concatenate(lifted_positions).astype(np.float64, copy=False)
    observation_ids = np.arange(row_values.size, dtype=np.int64)
    view_values = np.concatenate(sample_view_rows).astype(np.int32, copy=False)
    if str(args.training_samples):
        training_path = Path(args.training_samples)
        training_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            training_path,
            source_indices=clean_indices[clean_row_values].astype(np.int64),
            surface_cell_ids=cell_key_values.astype(np.int64),
            tangent_uv=tangent_values.astype(np.float16),
            observation_xy=pixel_values.astype(np.float16),
            radio_features=coarse_feature_values.astype(np.float16),
            detector_weights=weight_values.astype(np.float16),
            view_indices=view_values,
            view_image_ids_json=np.asarray(
                json.dumps([record.image_id for record in records])
            ),
            view_token_paths_json=np.asarray(
                json.dumps([str(record.token_path) for record in records])
            ),
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
    matrix = coo_matrix(
        (weight_values, (row_values, observation_ids)),
        shape=(unique_cell_keys.size, row_values.size),
        dtype=np.float32,
    ).tocsr()
    feature_sum = matrix @ feature_values
    position_sum = matrix @ position_values
    tangent_sum = matrix @ tangent_values
    weight_sum = np.asarray(matrix.sum(axis=1), dtype=np.float32).reshape(-1)
    observation_count = np.asarray(matrix.getnnz(axis=1), dtype=np.int32)
    mean = feature_sum / np.maximum(weight_sum[:, None], 1e-8)
    coherence = np.linalg.norm(mean, axis=1).astype(np.float32)
    mean /= np.maximum(coherence[:, None], 1e-8)
    retained = (
        (observation_count >= int(args.minimum_observations))
        & (coherence >= float(args.minimum_coherence))
    )
    rows = clean_indices[cell_clean_rows[retained]]
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
    texel_half_width = float(args.tangent_extent_sigma) / max(
        int(args.tangent_grid_size), 1
    )
    scale1 = scale1 * texel_half_width
    scale2 = scale2 * texel_half_width
    retained_tangent_uv = tangent_sum[retained] / np.maximum(
        weight_sum[retained, None], 1e-8
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
        "representation": "tangent_subcell_distribution_per_clean_2dgs_disk",
        "surface_coordinate": "depth_assigned_camera_ray_clean_surfel_plane_intersection_mean",
        "geometry_assignment": "multi_candidate_exact_ray_disk_footprint",
        "maximum_candidate_center_m": float(args.maximum_surface_snap_m),
        "maximum_plane_depth_residual_m": float(
            args.maximum_plane_depth_residual_m
        ),
        "maximum_disk_sigma": float(args.maximum_disk_sigma),
        "minimum_ray_normal_cosine": float(args.minimum_ray_normal_cosine),
        "metric_protocol": "decode_full_feature_map_then_bilinear_sample",
        "pixel_grid_convention": (
            "half_pixel_centers_align_corners_false"
            if str(args.feature_branch) == "highres_metric_decoded"
            else "legacy_radio_endpoint_align_corners_true"
        ),
        "feature_branch": str(args.feature_branch),
        "tangent_grid_size": int(args.tangent_grid_size),
        "tangent_extent_sigma": float(args.tangent_extent_sigma),
        "feature_dim": int(feature_values.shape[1]),
        "clean_primitive_count": int(clean_indices.size),
        "field_surfel_count": int(rows.size),
        "mapping_view_count": int(len(records)),
        "alike_role": "detection_coordinates_and_weights_only",
        "surface_sampling_mode": str(args.sampling_mode),
        "dense_sample_weight": float(args.dense_sample_weight),
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
        "uses_highres_metric_decoder": bool(
            str(args.highres_metric_decoder_checkpoint)
        ),
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
        surface_cell_ids=unique_cell_keys[retained],
        tangent_uv=retained_tangent_uv,
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
        "geometry_assignment": geometry_rejections,
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
