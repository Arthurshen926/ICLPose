"""Build detector-repeatable local anchors on an existing 2DGS surface map.

Mapping RGB is consumed only by this offline builder.  The output map contains
stable 3D surface points, ALIKE descriptors, support-mode coordinates, and no
RGB pixels or image paths.  A detected ALIKE point is lifted by intersecting
its calibrated camera ray with the local 2DGS surfel plane; multi-view
geometry and descriptor agreement then define anchor identity.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from scipy.spatial import cKDTree

from feature_extract.tools.vfm.build_stage_h2_raw_gaussian_anchor_map import (
    _load_camera_by_image,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.localization.real_image_observation_features import (
    AlikeDenseObservationExtractor,
)
from feature_extract.vfm.localization.surface_localization import (
    AnchorLocalDescriptorBank,
)
from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion
from feature_extract.vfm.surface_maplet_bank import (
    StableSurfaceAnchorMap,
    VfmSurfaceMapletBank,
    load_2dgs_primitive_quality,
)


@dataclass(frozen=True)
class _LiftedDetection:
    image_id: str
    xy: np.ndarray
    xyz: np.ndarray
    descriptor: np.ndarray
    detector_score: float
    weight: float
    depth: float


@dataclass(frozen=True)
class _FeatureAnchor:
    source_anchor_row: int
    source_anchor_id: int
    owner_maplet_id: int
    xyz: np.ndarray
    support_radius: float
    observations: tuple[_LiftedDetection, ...]
    prototype: np.ndarray
    repeatability: float
    consistency: float
    spread_m: float
    base_quality: float


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maplets", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--reference_pose_file", required=True)
    parser.add_argument("--camera_model_dir", required=True)
    parser.add_argument(
        "--camera_manifest",
        default="",
        help="Per-image calibration-only manifest for the actual mapping RGB.",
    )
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_maplets", required=True)
    parser.add_argument("--output_anchors", required=True)
    parser.add_argument("--output_local_descriptor_bank", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--clean_gaussian_ply", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--matcha_repo", default="/root/matcha")
    parser.add_argument("--alike_model_name", default="alike-t")
    parser.add_argument("--detector_top_k", type=int, default=2048)
    parser.add_argument("--detector_candidate_top_k", type=int, default=8192)
    parser.add_argument("--detector_nms_radius_px", type=float, default=2.0)
    parser.add_argument("--minimum_search_radius_px", type=float, default=3.0)
    parser.add_argument("--maximum_search_radius_px", type=float, default=20.0)
    parser.add_argument("--footprint_radius_scale", type=float, default=0.75)
    parser.add_argument("--cluster_radius_m", type=float, default=0.04)
    parser.add_argument("--maximum_cluster_radius_m", type=float, default=0.10)
    parser.add_argument("--minimum_descriptor_cosine", type=float, default=0.65)
    parser.add_argument("--minimum_cluster_views", type=int, default=3)
    parser.add_argument("--maximum_modes_per_source_anchor", type=int, default=2)
    parser.add_argument("--maximum_prototypes", type=int, default=8)
    parser.add_argument("--maximum_anchors_per_maplet", type=int, default=64)
    parser.add_argument("--minimum_anchors_per_maplet", type=int, default=4)
    parser.add_argument("--minimum_anchor_separation_m", type=float, default=0.015)
    parser.add_argument("--cache_shard_count", type=int, default=1)
    parser.add_argument("--cache_shard_index", type=int, default=0)
    parser.add_argument("--cache_only", action="store_true")
    parser.add_argument("--force_cache", action="store_true")
    return parser.parse_args(argv)


def _safe_cache_name(image_id: str) -> str:
    return str(image_id).replace("/", "__") + ".npz"


def _cache_detections(
    *,
    image_ids: Sequence[str],
    image_root: Path,
    camera_by_image,
    cache_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    shard_count = int(args.cache_shard_count)
    shard_index = int(args.cache_shard_index)
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("cache shard index/count are invalid")
    selected = [
        image_id
        for row, image_id in enumerate(sorted(image_ids))
        if row % shard_count == shard_index
    ]
    missing = [
        image_id
        for image_id in selected
        if bool(args.force_cache)
        or not (cache_dir / _safe_cache_name(image_id)).is_file()
    ]
    extractor = None
    if missing:
        extractor = AlikeDenseObservationExtractor(
            device=str(args.device),
            matcha_repo=Path(args.matcha_repo),
            model_name=str(args.alike_model_name),
        )
    written = 0
    for image_id in missing:
        camera = camera_by_image[image_id]
        detected = extractor.detect(
            image_root / image_id,
            image_width=int(camera.width),
            image_height=int(camera.height),
            top_k=int(args.detector_top_k),
            candidate_top_k=int(args.detector_candidate_top_k),
            nms_radius_px=float(args.detector_nms_radius_px),
            grid_rows=8,
            grid_cols=8,
        )
        np.savez_compressed(
            cache_dir / _safe_cache_name(image_id),
            image_id=np.asarray(image_id),
            xy=detected.xy.astype(np.float32),
            descriptors=detected.descriptors.astype(np.float32),
            scores=detected.scores.astype(np.float32),
            dispersions=detected.dispersions.astype(np.float32),
            image_sha256=np.asarray(detected.image_sha256),
        )
        written += 1
    return {
        "shard_count": shard_count,
        "shard_index": shard_index,
        "selected_image_count": len(selected),
        "cache_written_count": int(written),
        "cache_reused_count": int(len(selected) - written),
    }


def _load_detection(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return (
            np.asarray(data["xy"], dtype=np.float32),
            np.asarray(data["descriptors"], dtype=np.float32),
            np.asarray(data["scores"], dtype=np.float32),
        )


def _load_camera_manifest(path: Path):
    from feature_extract.vfm.colmap_tracks import ColmapCamera

    payload = json.loads(Path(path).read_text())
    if payload.get("format") != "per_query_colmap_calibration_only_v1":
        raise ValueError("unsupported camera calibration manifest")
    contract = dict(payload.get("production_contract") or {})
    if any(
        bool(contract.get(key, True))
        for key in (
            "contains_camera_pose",
            "contains_sfm_points",
            "contains_sfm_tracks",
        )
    ):
        raise ValueError("camera manifest contains forbidden pose/geometry")
    return {
        str(image_id): ColmapCamera(
            camera_id=int(row),
            model_id=int(record["model_id"]),
            width=int(record["width"]),
            height=int(record["height"]),
            params=tuple(float(value) for value in record["params"]),
        )
        for row, (image_id, record) in enumerate(
            sorted(dict(payload["cameras"]).items())
        )
    }


def _project_surface_anchor(
    xyz: np.ndarray,
    pose_w2c: np.ndarray,
    camera,
) -> tuple[np.ndarray, float] | None:
    matrix, distortion = camera_matrix_and_distortion(camera)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    camera_xyz = pose[:3, :3] @ np.asarray(xyz, dtype=np.float64) + pose[:3, 3]
    if float(camera_xyz[2]) <= 1e-6:
        return None
    rvec, _jacobian = cv2.Rodrigues(pose[:3, :3])
    projected, _jacobian = cv2.projectPoints(
        np.asarray(xyz, dtype=np.float64).reshape(1, 3),
        rvec,
        pose[:3, 3],
        matrix,
        distortion,
    )
    xy = projected.reshape(2)
    if not (
        0.0 <= float(xy[0]) <= float(camera.width - 1)
        and 0.0 <= float(xy[1]) <= float(camera.height - 1)
    ):
        return None
    return xy.astype(np.float32), float(camera_xyz[2])


def _lift_to_plane(
    *,
    xy: np.ndarray,
    anchor_xyz: np.ndarray,
    anchor_normal: np.ndarray,
    pose_w2c: np.ndarray,
    camera,
) -> tuple[np.ndarray, float] | None:
    matrix, distortion = camera_matrix_and_distortion(camera)
    normalized = cv2.undistortPoints(
        np.asarray(xy, dtype=np.float64).reshape(1, 1, 2),
        matrix,
        distortion,
    ).reshape(2)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    origin = -rotation.T @ translation
    ray = rotation.T @ np.asarray(
        [normalized[0], normalized[1], 1.0], dtype=np.float64
    )
    ray /= max(float(np.linalg.norm(ray)), 1e-12)
    normal = np.asarray(anchor_normal, dtype=np.float64).reshape(3)
    denominator = float(np.dot(normal, ray))
    if abs(denominator) < 1e-5:
        return None
    distance = float(
        np.dot(normal, np.asarray(anchor_xyz, dtype=np.float64) - origin)
        / denominator
    )
    if distance <= 0.0:
        return None
    xyz = origin + distance * ray
    depth = float((rotation @ xyz + translation)[2])
    if depth <= 1e-6 or not np.isfinite(xyz).all():
        return None
    return xyz, depth


def _collect_lifted_detections(
    *,
    anchors: StableSurfaceAnchorMap,
    image_ids: Sequence[str],
    cache_dir: Path,
    pose_by_image: dict[str, np.ndarray],
    camera_by_image,
    retained_anchor_rows: np.ndarray,
    args: argparse.Namespace,
) -> list[list[_LiftedDetection]]:
    retained = np.zeros((len(anchors),), dtype=bool)
    retained[np.asarray(retained_anchor_rows, dtype=np.int64)] = True
    observations_by_image: dict[str, list[int]] = {}
    for observation_row, image_id in enumerate(anchors.observation_image_ids):
        observations_by_image.setdefault(str(image_id), []).append(
            int(observation_row)
        )
    anchor_row_by_observation = np.repeat(
        np.arange(len(anchors), dtype=np.int64),
        np.diff(anchors.observation_offsets),
    )
    output: list[list[_LiftedDetection]] = [[] for _ in range(len(anchors))]
    for image_id in sorted(image_ids):
        cache_path = cache_dir / _safe_cache_name(image_id)
        if not cache_path.is_file():
            raise FileNotFoundError(f"ALIKE deployment cache is missing: {cache_path}")
        xy, descriptors, scores = _load_detection(cache_path)
        if len(xy) == 0:
            continue
        tree = cKDTree(xy)
        camera = camera_by_image[image_id]
        matrix, _distortion = camera_matrix_and_distortion(camera)
        focal = float(max(matrix[0, 0], matrix[1, 1]))
        for observation_row in observations_by_image.get(image_id, ()):
            anchor_row = int(anchor_row_by_observation[observation_row])
            if not retained[anchor_row]:
                continue
            projected = _project_surface_anchor(
                anchors.xyz[anchor_row],
                pose_by_image[image_id],
                camera,
            )
            if projected is None:
                continue
            projected_xy, projected_depth = projected
            depth = max(float(projected_depth), 1e-6)
            projected_radius = (
                focal
                * float(anchors.support_radii[anchor_row])
                / depth
                * float(args.footprint_radius_scale)
            )
            search_radius = float(
                np.clip(
                    projected_radius,
                    float(args.minimum_search_radius_px),
                    float(args.maximum_search_radius_px),
                )
            )
            distance, detection_row = tree.query(
                projected_xy, k=1
            )
            if not np.isfinite(distance) or float(distance) > search_radius:
                continue
            detection_row = int(detection_row)
            lifted = _lift_to_plane(
                xy=xy[detection_row],
                anchor_xyz=anchors.xyz[anchor_row],
                anchor_normal=anchors.normals[anchor_row],
                pose_w2c=pose_by_image[image_id],
                camera=camera,
            )
            if lifted is None:
                continue
            xyz, lifted_depth = lifted
            maximum_surface_displacement = max(
                float(args.cluster_radius_m) * 2.0,
                min(
                    float(anchors.support_radii[anchor_row]) * 1.25,
                    float(args.maximum_cluster_radius_m) * 2.0,
                ),
            )
            if (
                float(np.linalg.norm(xyz - anchors.xyz[anchor_row]))
                > maximum_surface_displacement
            ):
                continue
            spatial_weight = float(
                np.exp(-0.5 * (float(distance) / max(search_radius, 1e-6)) ** 2)
            )
            output[anchor_row].append(
                _LiftedDetection(
                    image_id=str(image_id),
                    xy=xy[detection_row].astype(np.float32),
                    xyz=xyz.astype(np.float64),
                    descriptor=descriptors[detection_row].astype(np.float32),
                    detector_score=float(max(scores[detection_row], 0.0)),
                    weight=float(
                        max(anchors.observation_weights[observation_row], 1e-8)
                        * max(scores[detection_row], 1e-8)
                        * spatial_weight
                    ),
                    depth=float(lifted_depth),
                )
            )
    return output


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    return value / max(float(np.linalg.norm(value)), 1e-8)


def _cluster_source_anchor(
    *,
    anchor_row: int,
    anchors: StableSurfaceAnchorMap,
    observations: Sequence[_LiftedDetection],
    args: argparse.Namespace,
) -> list[_FeatureAnchor]:
    if len({item.image_id for item in observations}) < int(
        args.minimum_cluster_views
    ):
        return []
    remaining = set(range(len(observations)))
    output: list[_FeatureAnchor] = []
    radius = min(
        float(args.maximum_cluster_radius_m),
        max(
            float(args.cluster_radius_m),
            0.25 * float(anchors.support_radii[anchor_row]),
        ),
    )
    ranked_seeds = sorted(
        remaining,
        key=lambda row: (
            -float(observations[row].detector_score),
            observations[row].image_id,
        ),
    )
    for seed_row in ranked_seeds:
        if seed_row not in remaining:
            continue
        seed = observations[seed_row]
        compatible = [
            row
            for row in remaining
            if float(np.linalg.norm(observations[row].xyz - seed.xyz)) <= radius
            and float(
                np.dot(observations[row].descriptor, seed.descriptor)
            )
            >= float(args.minimum_descriptor_cosine)
        ]
        best_by_view: dict[str, int] = {}
        for row in compatible:
            image_id = observations[row].image_id
            previous = best_by_view.get(image_id)
            if previous is None or observations[row].weight > observations[previous].weight:
                best_by_view[image_id] = int(row)
        cluster_rows = list(best_by_view.values())
        if len(cluster_rows) < int(args.minimum_cluster_views):
            continue
        weights = np.asarray(
            [max(observations[row].weight, 1e-12) for row in cluster_rows],
            dtype=np.float64,
        )
        weights /= max(float(np.sum(weights)), 1e-12)
        xyz = np.sum(
            np.stack([observations[row].xyz for row in cluster_rows], axis=0)
            * weights[:, None],
            axis=0,
        )
        prototype = _normalize(
            np.sum(
                np.stack(
                    [observations[row].descriptor for row in cluster_rows],
                    axis=0,
                )
                * weights[:, None],
                axis=0,
            )
        )
        refined_rows = [
            row
            for row in compatible
            if float(np.linalg.norm(observations[row].xyz - xyz)) <= radius
            and float(np.dot(observations[row].descriptor, prototype))
            >= float(args.minimum_descriptor_cosine)
        ]
        best_by_view = {}
        for row in refined_rows:
            image_id = observations[row].image_id
            previous = best_by_view.get(image_id)
            if previous is None or observations[row].weight > observations[previous].weight:
                best_by_view[image_id] = int(row)
        cluster_rows = list(best_by_view.values())
        if len(cluster_rows) < int(args.minimum_cluster_views):
            continue
        cluster = tuple(observations[row] for row in cluster_rows)
        weights = np.asarray(
            [max(item.weight, 1e-12) for item in cluster], dtype=np.float64
        )
        weights /= max(float(np.sum(weights)), 1e-12)
        xyz = np.sum(
            np.stack([item.xyz for item in cluster], axis=0) * weights[:, None],
            axis=0,
        )
        prototype = _normalize(
            np.sum(
                np.stack([item.descriptor for item in cluster], axis=0)
                * weights[:, None],
                axis=0,
            )
        )
        similarities = np.asarray(
            [float(np.dot(item.descriptor, prototype)) for item in cluster]
        )
        distances = np.asarray(
            [float(np.linalg.norm(item.xyz - xyz)) for item in cluster]
        )
        available_views = max(len({item.image_id for item in observations}), 1)
        repeatability = float(len(cluster) / available_views)
        consistency = float(np.median(similarities))
        spread = float(np.quantile(distances, 0.9))
        detector_quality = float(
            np.median([max(item.detector_score, 0.0) for item in cluster])
        )
        base_quality = (
            max(float(anchors.quality_scores[anchor_row]), 1e-8)
            * max(repeatability, 1e-3)
            * max((consistency - 0.5) / 0.5, 1e-3)
            * max(np.sqrt(detector_quality), 1e-4)
            * np.log1p(len(cluster))
        )
        output.append(
            _FeatureAnchor(
                source_anchor_row=int(anchor_row),
                source_anchor_id=int(anchors.anchor_ids[anchor_row]),
                owner_maplet_id=int(anchors.owner_maplet_ids[anchor_row]),
                xyz=xyz.astype(np.float64),
                support_radius=float(max(0.01, min(radius, spread + 0.01))),
                observations=cluster,
                prototype=prototype,
                repeatability=repeatability,
                consistency=consistency,
                spread_m=spread,
                base_quality=float(base_quality),
            )
        )
        remaining.difference_update(cluster_rows)
        if len(output) >= int(args.maximum_modes_per_source_anchor):
            break
    return output


def _select_maplet_anchors(
    candidates: Sequence[_FeatureAnchor],
    *,
    maximum_count: int,
    minimum_separation: float,
) -> list[tuple[_FeatureAnchor, float, float]]:
    if not candidates:
        return []
    prototypes = np.stack([item.prototype for item in candidates], axis=0)
    similarity = prototypes @ prototypes.T
    np.fill_diagonal(similarity, -np.inf)
    nearest_negative = np.max(similarity, axis=1)
    distinctiveness = np.clip(
        (1.0 - nearest_negative) / 0.30,
        0.02,
        1.0,
    )
    base = np.asarray([item.base_quality for item in candidates], dtype=np.float64)
    positive = base[base > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    localization_quality = np.sqrt(np.maximum(base / max(scale, 1e-12), 1e-8))
    localization_quality *= distinctiveness
    order = np.argsort(-localization_quality, kind="mergesort")
    selected: list[int] = []
    for row in order.tolist():
        if len(selected) >= int(maximum_count):
            break
        if selected:
            distance = np.linalg.norm(
                np.stack([candidates[value].xyz for value in selected], axis=0)
                - candidates[row].xyz,
                axis=1,
            )
            if float(np.min(distance)) < float(minimum_separation):
                continue
        selected.append(int(row))
    return [
        (
            candidates[row],
            float(localization_quality[row]),
            float(distinctiveness[row]),
        )
        for row in selected
    ]


def _rebuild_artifacts(
    *,
    maplets: VfmSurfaceMapletBank,
    anchors: StableSurfaceAnchorMap,
    selected_by_maplet: dict[int, list[tuple[_FeatureAnchor, float, float]]],
    pose_by_image: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[VfmSurfaceMapletBank, StableSurfaceAnchorMap, AnchorLocalDescriptorBank]:
    kept_rows = [
        row
        for row, maplet_id in enumerate(maplets.maplet_ids.tolist())
        if len(selected_by_maplet.get(int(maplet_id), ()))
        >= int(args.minimum_anchors_per_maplet)
    ]
    new_anchor_ids: list[int] = []
    owner_ids: list[int] = []
    surface_ids: list[int] = []
    parent_ids: list[int] = []
    xyz: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    radii: list[float] = []
    qualities: list[float] = []
    geometry: list[float] = []
    opacity: list[float] = []
    observation_offsets = [0]
    observation_image_ids: list[str] = []
    observation_xy: list[np.ndarray] = []
    observation_depth: list[float] = []
    observation_weights: list[float] = []
    descriptor_offsets = [0]
    descriptors: list[np.ndarray] = []
    descriptor_image_ids: list[str] = []
    descriptor_quality: list[float] = []
    descriptor_view_directions: list[np.ndarray] = []
    anchor_ids_by_maplet: dict[int, list[int]] = {}
    next_anchor_id = 1_000_000_000
    for maplet_row in kept_rows:
        maplet_id = int(maplets.maplet_ids[maplet_row])
        anchor_ids_by_maplet[maplet_id] = []
        for candidate, quality, _distinctiveness in selected_by_maplet[maplet_id]:
            source_row = int(candidate.source_anchor_row)
            anchor_id = int(next_anchor_id)
            next_anchor_id += 1
            anchor_ids_by_maplet[maplet_id].append(anchor_id)
            new_anchor_ids.append(anchor_id)
            owner_ids.append(maplet_id)
            surface_ids.append(int(anchors.surface_element_ids[source_row]))
            parent_ids.append(int(anchors.parent_primitive_indices[source_row]))
            xyz.append(candidate.xyz)
            normal = anchors.normals[source_row].astype(
                np.float64
            ).copy()
            weighted_view = np.zeros((3,), dtype=np.float64)
            weight_sum = 0.0
            for observation in candidate.observations:
                pose = pose_by_image.get(observation.image_id)
                if pose is None:
                    continue
                camera_center = -pose[:3, :3].T @ pose[:3, 3]
                direction = camera_center - candidate.xyz
                direction_norm = float(np.linalg.norm(direction))
                if direction_norm <= 1e-12:
                    continue
                weight = max(float(observation.weight), 1e-8)
                weighted_view += weight * direction / direction_norm
                weight_sum += weight
            if (
                weight_sum > 0.0
                and float(np.dot(normal, weighted_view)) < 0.0
            ):
                normal *= -1.0
            normals.append(normal)
            covariances.append(anchors.tangent_covariances[source_row])
            radii.append(float(candidate.support_radius))
            qualities.append(float(quality))
            geometry.append(float(anchors.geometry_confidence[source_row]))
            opacity.append(float(anchors.opacity[source_row]))
            ordered = sorted(
                candidate.observations,
                key=lambda item: (-item.weight, item.image_id),
            )
            for item in ordered:
                observation_image_ids.append(item.image_id)
                observation_xy.append(item.xy)
                observation_depth.append(float(item.depth))
                observation_weights.append(float(item.weight))
            observation_offsets.append(len(observation_image_ids))
            chosen = ordered[: int(args.maximum_prototypes)]
            for item in chosen:
                descriptors.append(item.descriptor)
                descriptor_image_ids.append(item.image_id)
                pose = pose_by_image[item.image_id]
                camera_center = -pose[:3, :3].T @ pose[:3, 3]
                view_direction = camera_center - candidate.xyz
                view_direction /= max(
                    float(np.linalg.norm(view_direction)),
                    1e-12,
                )
                descriptor_view_directions.append(view_direction)
                descriptor_quality.append(
                    float(
                        max(item.detector_score, 1e-8)
                        * max(candidate.consistency, 1e-3)
                        * max(candidate.repeatability, 1e-3)
                    )
                )
            descriptor_offsets.append(len(descriptors))

    stable = StableSurfaceAnchorMap(
        anchor_ids=np.asarray(new_anchor_ids, dtype=np.int64),
        owner_maplet_ids=np.asarray(owner_ids, dtype=np.int64),
        surface_element_ids=np.asarray(surface_ids, dtype=np.int64),
        parent_primitive_indices=np.asarray(parent_ids, dtype=np.int64),
        xyz=np.asarray(xyz, dtype=np.float64).reshape(-1, 3),
        normals=np.asarray(normals, dtype=np.float32).reshape(-1, 3),
        tangent_covariances=np.asarray(covariances, dtype=np.float32).reshape(-1, 3, 3),
        support_radii=np.asarray(radii, dtype=np.float32),
        quality_scores=np.asarray(qualities, dtype=np.float32),
        geometry_confidence=np.asarray(geometry, dtype=np.float32),
        opacity=np.asarray(opacity, dtype=np.float32),
        observation_offsets=np.asarray(observation_offsets, dtype=np.int64),
        observation_image_ids=tuple(observation_image_ids),
        observation_xy=np.asarray(observation_xy, dtype=np.float32).reshape(-1, 2),
        observation_depth=np.asarray(observation_depth, dtype=np.float32),
        observation_weights=np.asarray(observation_weights, dtype=np.float32),
        metadata={
            "representation": "feature_aligned_2dgs_surface_anchor",
            "identity": "multiview_alike_detection_cluster_on_2dgs_surface",
            "normal_orientation": (
                "signed_toward_weighted_mapping_observation_camera_centers"
            ),
            "normal_orientation_uses_mapping_pose_at_runtime": False,
            "uses_mapping_rgb_at_inference": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_radio_intermediate": False,
        },
    )
    descriptor_dim = int(descriptors[0].shape[0]) if descriptors else 0
    local_bank = AnchorLocalDescriptorBank(
        anchor_ids=np.asarray(new_anchor_ids, dtype=np.int64),
        descriptor_offsets=np.asarray(descriptor_offsets, dtype=np.int64),
        descriptors=np.asarray(descriptors, dtype=np.float32).reshape(
            -1, descriptor_dim
        ),
        support_image_ids=tuple(descriptor_image_ids),
        descriptor_quality=np.asarray(descriptor_quality, dtype=np.float32),
        support_view_directions=np.asarray(
            descriptor_view_directions, dtype=np.float32
        ).reshape(-1, 3),
        metadata={
            "representation": "feature_aligned_2dgs_surface_anchor_local_descriptors",
            "local_feature": "alike_detected_points",
            "view_conditioning": (
                "per_descriptor_anchor_to_support_camera_unit_direction"
            ),
            "stores_mapping_pose": False,
            "uses_mapping_rgb_at_inference": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_radio_intermediate": False,
        },
    )

    support_offsets = [0]
    support_ids: list[int] = []
    view_offsets = [0]
    view_ids: list[str] = []
    view_xy: list[np.ndarray] = []
    view_sizes: list[np.ndarray] = []
    view_descriptors: list[np.ndarray] = []
    view_quality: list[float] = []
    anchor_offsets = [0]
    flattened_anchor_ids: list[int] = []
    for row in kept_rows:
        maplet_id = int(maplets.maplet_ids[row])
        flattened_anchor_ids.extend(anchor_ids_by_maplet[maplet_id])
        anchor_offsets.append(len(flattened_anchor_ids))
        start, end = int(maplets.support_offsets[row]), int(maplets.support_offsets[row + 1])
        support_ids.extend(maplets.support_element_ids[start:end].tolist())
        support_offsets.append(len(support_ids))
        start, end = int(maplets.view_offsets[row]), int(maplets.view_offsets[row + 1])
        view_ids.extend(maplets.view_image_ids[start:end])
        view_xy.extend(maplets.view_token_xy[start:end])
        view_sizes.extend(maplets.view_grid_sizes[start:end])
        view_descriptors.extend(maplets.view_descriptors[start:end])
        view_quality.extend(maplets.view_quality_scores[start:end].tolist())
        view_offsets.append(len(view_ids))
    rows = np.asarray(kept_rows, dtype=np.int64)
    rebuilt_maplets = VfmSurfaceMapletBank(
        maplet_ids=maplets.maplet_ids[rows],
        centers=maplets.centers[rows],
        normals=maplets.normals[rows],
        tangent_frames=maplets.tangent_frames[rows],
        extents=maplets.extents[rows],
        descriptors=maplets.descriptors[rows],
        quality_scores=maplets.quality_scores[rows],
        descriptor_variances=maplets.descriptor_variances[rows],
        anchor_offsets=np.asarray(anchor_offsets, dtype=np.int64),
        anchor_ids=np.asarray(flattened_anchor_ids, dtype=np.int64),
        support_offsets=np.asarray(support_offsets, dtype=np.int64),
        support_element_ids=np.asarray(support_ids, dtype=np.int64),
        view_offsets=np.asarray(view_offsets, dtype=np.int64),
        view_image_ids=tuple(view_ids),
        view_token_xy=np.asarray(view_xy, dtype=np.float32).reshape(-1, 2),
        view_grid_sizes=np.asarray(view_sizes, dtype=np.int32).reshape(-1, 2),
        view_descriptors=np.asarray(view_descriptors, dtype=np.float32).reshape(
            -1, maplets.feature_dim
        ),
        view_quality_scores=np.asarray(view_quality, dtype=np.float32),
        metadata={
            **dict(maplets.metadata or {}),
            "anchor_representation": "feature_aligned_2dgs_surface_anchor",
            "local_descriptor_dim": int(local_bank.feature_dim),
        },
    )
    return rebuilt_maplets, stable, local_bank


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    maplets = VfmSurfaceMapletBank.load_npz(Path(args.maplets))
    anchors = StableSurfaceAnchorMap.load_npz(Path(args.anchors))
    pose_by_image = {
        record.image_id: record.pose_w2c
        for record in parse_cambridge_pose_file(Path(args.reference_pose_file))
    }
    camera_by_image = (
        _load_camera_manifest(Path(args.camera_manifest))
        if str(args.camera_manifest)
        else _load_camera_by_image(str(args.camera_model_dir))
    )
    image_ids = sorted(set(anchors.observation_image_ids))
    missing = sorted(
        (set(image_ids) - set(pose_by_image))
        | (set(image_ids) - set(camera_by_image))
    )
    if missing:
        raise ValueError(f"anchor support view lacks pose/intrinsics: {missing[0]}")
    cache_summary = _cache_detections(
        image_ids=image_ids,
        image_root=Path(args.image_root),
        camera_by_image=camera_by_image,
        cache_dir=Path(args.cache_dir),
        args=args,
    )
    if bool(args.cache_only):
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(
            json.dumps(
                {
                    "stage": "cache_feature_aligned_surface_anchor_detections",
                    **cache_summary,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return
    retained_rows = np.arange(len(anchors), dtype=np.int64)
    clean_summary = None
    if str(args.clean_gaussian_ply):
        quality = load_2dgs_primitive_quality(Path(args.clean_gaussian_ply))
        parents = anchors.parent_primitive_indices
        retained_rows = np.flatnonzero(
            (parents >= 0)
            & (parents < len(quality))
            & (quality.geometry_confidence[np.clip(parents, 0, len(quality) - 1)] > 0.0)
        )
        clean_summary = {
            **dict(quality.metadata or {}),
            "input_anchor_count": int(len(anchors)),
            "retained_anchor_count": int(len(retained_rows)),
        }
    observations = _collect_lifted_detections(
        anchors=anchors,
        image_ids=image_ids,
        cache_dir=Path(args.cache_dir),
        pose_by_image=pose_by_image,
        camera_by_image=camera_by_image,
        retained_anchor_rows=retained_rows,
        args=args,
    )
    feature_candidates: list[_FeatureAnchor] = []
    for anchor_row in retained_rows.tolist():
        feature_candidates.extend(
            _cluster_source_anchor(
                anchor_row=int(anchor_row),
                anchors=anchors,
                observations=observations[int(anchor_row)],
                args=args,
            )
        )
    candidates_by_maplet: dict[int, list[_FeatureAnchor]] = {}
    for candidate in feature_candidates:
        candidates_by_maplet.setdefault(candidate.owner_maplet_id, []).append(
            candidate
        )
    selected_by_maplet = {
        int(maplet_id): _select_maplet_anchors(
            values,
            maximum_count=int(args.maximum_anchors_per_maplet),
            minimum_separation=float(args.minimum_anchor_separation_m),
        )
        for maplet_id, values in candidates_by_maplet.items()
    }
    rebuilt_maplets, rebuilt_anchors, local_bank = _rebuild_artifacts(
        maplets=maplets,
        anchors=anchors,
        selected_by_maplet=selected_by_maplet,
        pose_by_image=pose_by_image,
        args=args,
    )
    rebuilt_maplets.save_npz(Path(args.output_maplets))
    rebuilt_anchors.save_npz(Path(args.output_anchors))
    local_bank.save_npz(Path(args.output_local_descriptor_bank))
    counts = np.diff(rebuilt_maplets.anchor_offsets)
    summary = {
        "stage": "build_feature_aligned_surface_anchors",
        "input_maplet_count": int(len(maplets)),
        "input_anchor_count": int(len(anchors)),
        "clean_prior": clean_summary,
        "source_anchors_with_lifted_detections": int(
            sum(bool(values) for values in observations)
        ),
        "lifted_detection_count": int(sum(len(values) for values in observations)),
        "feature_cluster_candidate_count": int(len(feature_candidates)),
        "output_maplet_count": int(len(rebuilt_maplets)),
        "output_anchor_count": int(len(rebuilt_anchors)),
        "output_descriptor_count": int(len(local_bank.descriptors)),
        "anchors_per_maplet": {
            "min": int(np.min(counts)) if len(counts) else 0,
            "median": float(np.median(counts)) if len(counts) else 0.0,
            "mean": float(np.mean(counts)) if len(counts) else 0.0,
            "max": int(np.max(counts)) if len(counts) else 0,
        },
        "cache": cache_summary,
        "production_contract": {
            "map_representation": "feature_aligned_2dgs_surface_map",
            "anchor_identity": "multiview_alike_detection_cluster_on_2dgs_surface",
            "stores_mapping_rgb": False,
            "uses_mapping_rgb_at_inference": False,
            "uses_pairwise_image_matching": False,
            "uses_loftr": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_radio_intermediate": False,
        },
        "outputs": {
            "maplets": str(args.output_maplets),
            "anchors": str(args.output_anchors),
            "local_descriptor_bank": str(args.output_local_descriptor_bank),
        },
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
