"""Localize query images against a feature-only 2DGS surface map.

Runtime inputs are deliberately limited to query RGB/RADIO-final tokens and
frozen map artifacts.  Mapping RGB, SfM points/tracks, pairwise image matchers,
and view-depth lifting are not accepted by this program.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
from scipy.special import logsumexp

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import ColmapCamera, read_colmap_cameras_binary
from feature_extract.vfm.localization.real_image_observation_features import (
    AlikeDenseObservationExtractor,
)
from feature_extract.vfm.localization.anchor_feature_contract import (
    ALIKE_RADIO_FINAL_ANCHOR_FEATURE,
    RADIO_FINAL_ANCHOR_FEATURE,
    anchor_feature_kind,
    compose_anchor_query_descriptors,
)
from feature_extract.vfm.localization.surface_anchor_set_matcher import (
    load_surface_anchor_set_matcher,
    match_query_to_surface_anchors,
)
from feature_extract.vfm.localization.surface_localization import (
    AnchorLocalDescriptorBank,
    LocalFeatureFrame,
    SurfaceMapletMatchConfig,
    SurfacePoseConfig,
    SurfacePoseResult,
    VfmSurfaceObservationIndex,
    build_maplet_conditioned_surface_anchor_candidate_pool,
    build_pose_guided_surface_anchor_candidate_pool,
    build_seeded_surface_anchor_candidate_pool,
    build_vfm_surface_observation_candidate_pool,
    estimate_vfm_query_to_support_layout,
    generate_grouped_surface_pose_hypotheses,
    match_radio_final_regions_to_maplets,
    predict_vfm_support_anchor_query_points,
    score_surface_pose,
    select_vfm_surface_feature_modes,
)
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
    select_spatially_balanced_radio_final_regions,
)
from feature_extract.vfm.surface_maplet_bank import (
    RadioFinalRegionConfig,
    StableSurfaceAnchorMap,
    VfmSurfaceMapletBank,
    encode_radio_final_regions,
)
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion
from feature_extract.vfm.vfm_2dgs_mapping import Vfm2DgsObservationBank


FORBIDDEN_RUNTIME_FLAGS = (
    "uses_mapping_rgb_at_inference",
    "uses_mapping_image_retrieval",
    "uses_pairwise_image_matching",
    "uses_sfm_points",
    "uses_sfm_tracks",
    "uses_radio_intermediate",
    "uses_view_depth_at_inference",
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_image_root", required=True)
    parser.add_argument("--radio_final_layer", default="radio_final")
    parser.add_argument("--surface_mapper_checkpoint", required=True)
    parser.add_argument("--maplets", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--local_descriptor_bank", required=True)
    parser.add_argument(
        "--radio_surface_feature_bank",
        required=True,
        help=(
            "RADIO-final features attached to 2DGS surface observations; "
            "contains no mapping RGB."
        ),
    )
    parser.add_argument("--surface_anchor_matcher_checkpoint", required=True)
    parser.add_argument("--camera_model_dir", required=True)
    parser.add_argument(
        "--query_camera_manifest",
        default="",
        help=(
            "Optional per-query calibration-only JSON. It must not contain "
            "camera poses, SfM points, or tracks."
        ),
    )
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--matcha_repo", default="/root/matcha")
    parser.add_argument("--alike_model_name", default="alike-t")
    parser.add_argument("--query_region_rows", type=int, default=8)
    parser.add_argument("--query_region_cols", type=int, default=8)
    parser.add_argument("--query_regions_per_cell", type=int, default=2)
    parser.add_argument("--maplet_top_k", type=int, default=8)
    parser.add_argument("--maximum_support_views", type=int, default=64)
    parser.add_argument("--maximum_layout_models", type=int, default=4096)
    parser.add_argument(
        "--disable_maplet_support_layout",
        action="store_true",
        help="Disable stored support-mode layout evidence during coarse maplet retrieval.",
    )
    parser.add_argument("--maximum_feature_modes", type=int, default=8)
    parser.add_argument("--maximum_global_feature_modes", type=int, default=64)
    parser.add_argument(
        "--feature_mode_spatial_rerank",
        action="store_true",
        help=(
            "Rerank the frozen VFM mode shortlist by query-to-support "
            "spatial-layout consensus before ALIKE anchor measurement."
        ),
    )
    parser.add_argument("--feature_mode_minimum_similarity", type=float, default=0.20)
    parser.add_argument("--local_top_k", type=int, default=2048)
    parser.add_argument("--local_candidate_top_k", type=int, default=8192)
    parser.add_argument("--local_nms_radius_px", type=float, default=2.0)
    parser.add_argument(
        "--augment_alike_with_radio_final",
        action="store_true",
        help=(
            "Concatenate normalized RADIO-final sampled at query ALIKE "
            "points; requires an equally augmented map descriptor bank."
        ),
    )
    parser.add_argument("--anchor_top_l", type=int, default=20)
    parser.add_argument(
        "--matcher_feature_preferred_base_fill_target",
        type=int,
        default=0,
        help=(
            "When positive, learned set matching uses every feature-aligned "
            "anchor (stable IDs >=1e9) and only enough geometry-first anchors "
            "to reach this per-maplet target. The full map remains available "
            "to feature-only layout, verification, and refinement."
        ),
    )
    parser.add_argument("--verification_anchor_top_l", type=int, default=10)
    parser.add_argument("--layout_guided_maximum_points", type=int, default=256)
    parser.add_argument("--layout_guided_search_radius_px", type=int, default=18)
    parser.add_argument("--layout_guided_seed_prior_logit", type=float, default=32.0)
    parser.add_argument("--disable_layout_guided_measurement", action="store_true")
    parser.add_argument(
        "--always_direct_surface_candidate",
        action="store_true",
        help=(
            "Let query RADIO-final to 2DGS observation-field pose compete "
            "with set/mode candidates instead of waiting for total failure."
        ),
    )
    parser.add_argument("--hypothesis_count", type=int, default=256)
    parser.add_argument("--minimum_fit_groups", type=int, default=8)
    parser.add_argument("--minimum_verification_groups", type=int, default=4)
    parser.add_argument("--heldout_stride", type=int, default=5)
    parser.add_argument(
        "--set_pose_maximum_mean_null",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--set_pose_minimum_confident_groups",
        type=int,
        default=12,
    )
    parser.add_argument("--pose_refinement_radius_px", type=float, default=24.0)
    parser.add_argument("--pose_refinement_maximum_points", type=int, default=768)
    parser.add_argument(
        "--featuremetric_refinement_radius_px", type=int, default=14
    )
    parser.add_argument(
        "--featuremetric_refinement_maximum_anchors", type=int, default=256
    )
    parser.add_argument(
        "--featuremetric_refinement_minimum_similarity",
        type=float,
        default=0.65,
    )
    parser.add_argument(
        "--disable_featuremetric_refinement", action="store_true"
    )
    parser.add_argument(
        "--feature_pose_evidence_maximum_anchors", type=int, default=384
    )
    parser.add_argument(
        "--feature_pose_evidence_weight", type=float, default=0.75
    )
    parser.add_argument("--disable_pose_refinement", action="store_true")
    parser.add_argument(
        "--emit_candidate_pose_trace",
        action="store_true",
        help="Store candidate poses for offline branch/hypothesis oracle audit.",
    )
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _camera_from_model(model_dir: Path) -> ColmapCamera:
    """Load intrinsics only; no points, tracks, images, or poses are read."""

    cameras = read_colmap_cameras_binary(Path(model_dir) / "cameras.bin")
    if not cameras:
        raise ValueError("camera model contains no cameras")
    structural = {
        (
            int(camera.model_id),
            int(camera.width),
            int(camera.height),
            len(camera.params),
        )
        for camera in cameras.values()
    }
    if len(structural) != 1:
        raise ValueError("camera model has incompatible intrinsic families")
    model_id, width, height, parameter_count = next(iter(structural))
    parameters = np.asarray(
        [camera.params for camera in cameras.values()], dtype=np.float64
    ).reshape(-1, int(parameter_count))
    return ColmapCamera(
        camera_id=0,
        model_id=int(model_id),
        width=int(width),
        height=int(height),
        params=tuple(
            float(value) for value in np.median(parameters, axis=0).tolist()
        ),
    )


def _load_query_camera_manifest(
    path: Path,
) -> tuple[dict[str, ColmapCamera], dict[str, object]]:
    payload = json.loads(Path(path).read_text())
    if payload.get("format") != "per_query_colmap_calibration_only_v1":
        raise ValueError("unsupported query camera manifest format")
    contract = dict(payload.get("production_contract") or {})
    if (
        bool(contract.get("contains_camera_pose", True))
        or bool(contract.get("contains_sfm_points", True))
        or bool(contract.get("contains_sfm_tracks", True))
    ):
        raise ValueError("query camera manifest contains forbidden geometry")
    output: dict[str, ColmapCamera] = {}
    for row, (image_id, record) in enumerate(
        sorted(dict(payload["cameras"]).items())
    ):
        output[str(image_id)] = ColmapCamera(
            camera_id=int(row),
            model_id=int(record["model_id"]),
            width=int(record["width"]),
            height=int(record["height"]),
            params=tuple(float(value) for value in record["params"]),
        )
    return output, dict(payload.get("intrinsic_audit") or {})


def _load_raw_final(path: Path, layer_name: str) -> np.ndarray:
    with np.load(Path(path)) as data:
        if str(layer_name) not in data:
            raise ValueError(
                f"RADIO-final layer {layer_name!r} is absent from {path}"
            )
        feature = np.asarray(data[str(layer_name)], dtype=np.float32)
    if feature.ndim == 4 and int(feature.shape[0]) == 1:
        feature = feature[0]
    if feature.ndim != 3:
        raise ValueError("RADIO-final query feature must have shape (C,H,W)")
    return feature


def _sample_mapped_vfm_at_pixels(
    mapped_feature: np.ndarray,
    xy: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    align_corners: bool = True,
) -> np.ndarray:
    feature = np.asarray(mapped_feature, dtype=np.float32)
    if feature.ndim != 3:
        raise ValueError("mapped VFM feature must have shape (C,H,W)")
    points = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    if bool(align_corners):
        grid_x = (
            points[:, 0]
            * max(int(feature.shape[2]) - 1, 1)
            / max(int(image_width) - 1, 1)
        ).astype(np.float32)
        grid_y = (
            points[:, 1]
            * max(int(feature.shape[1]) - 1, 1)
            / max(int(image_height) - 1, 1)
        ).astype(np.float32)
    else:
        grid_x = (
            (points[:, 0] + 0.5)
            * int(feature.shape[2])
            / max(int(image_width), 1)
            - 0.5
        ).astype(np.float32)
        grid_y = (
            (points[:, 1] + 0.5)
            * int(feature.shape[1])
            / max(int(image_height), 1)
            - 0.5
        ).astype(np.float32)
    # OpenCV remap has implementation-dependent limits for very large channel
    # counts (raw RADIO-final has 1280). Chunking preserves the exact spatial
    # sampling protocol without routing raw features through a retrieval head.
    chunks = []
    for begin in range(0, int(feature.shape[0]), 128):
        end = min(begin + 128, int(feature.shape[0]))
        value = cv2.remap(
            feature[begin:end].transpose(1, 2, 0),
            grid_x.reshape(-1, 1),
            grid_y.reshape(-1, 1),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        chunks.append(value.reshape(len(points), end - begin))
    sampled = np.concatenate(chunks, axis=1)
    sampled /= np.maximum(
        np.linalg.norm(sampled, axis=1, keepdims=True),
        1e-8,
    )
    return sampled.astype(np.float32)


def _match_radio_final_descriptor_points(
    *,
    mapped_feature: np.ndarray,
    predicted_xy: np.ndarray,
    support_descriptors: np.ndarray,
    image_width: int,
    image_height: int,
    search_radius_px: int,
    search_step_px: int = 2,
    point_chunk_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure map descriptors by local RADIO-final correlation only."""

    predicted = np.asarray(predicted_xy, dtype=np.float32).reshape(-1, 2)
    support = np.asarray(support_descriptors, dtype=np.float32)
    if support.ndim != 2 or support.shape[0] != predicted.shape[0]:
        raise ValueError(
            "support descriptors must align with predicted points"
        )
    if support.shape[1] != int(np.asarray(mapped_feature).shape[0]):
        raise ValueError(
            "support and query RADIO-final descriptor dimensions differ"
        )
    radius = int(search_radius_px)
    step = int(search_step_px)
    if radius < 0 or step <= 0 or int(point_chunk_size) <= 0:
        raise ValueError("RADIO-final local-search limits are invalid")
    axis = np.arange(-radius, radius + 1, step, dtype=np.float32)
    if not np.any(axis == 0.0):
        axis = np.unique(np.concatenate([axis, np.zeros((1,), np.float32)]))
    offset_y, offset_x = np.meshgrid(axis, axis, indexing="ij")
    offsets = np.stack([offset_x.reshape(-1), offset_y.reshape(-1)], axis=1)
    measured_xy: list[np.ndarray] = []
    measured_descriptors: list[np.ndarray] = []
    measured_scores: list[np.ndarray] = []
    for begin in range(0, len(predicted), int(point_chunk_size)):
        end = min(begin + int(point_chunk_size), len(predicted))
        candidates = predicted[begin:end, None, :] + offsets[None, :, :]
        candidates[..., 0] = np.clip(
            candidates[..., 0], 0.0, float(int(image_width) - 1)
        )
        candidates[..., 1] = np.clip(
            candidates[..., 1], 0.0, float(int(image_height) - 1)
        )
        flat_descriptors = _sample_mapped_vfm_at_pixels(
            mapped_feature,
            candidates.reshape(-1, 2),
            image_width=int(image_width),
            image_height=int(image_height),
        )
        candidate_descriptors = flat_descriptors.reshape(
            end - begin, len(offsets), -1
        )
        similarities = np.einsum(
            "nkd,nd->nk",
            candidate_descriptors,
            support[begin:end],
            optimize=True,
        )
        best = np.argmax(similarities, axis=1)
        row = np.arange(end - begin)
        measured_xy.append(candidates[row, best])
        measured_descriptors.append(candidate_descriptors[row, best])
        measured_scores.append(similarities[row, best])
    return (
        np.concatenate(measured_xy, axis=0).astype(np.float32),
        np.concatenate(measured_descriptors, axis=0).astype(np.float32),
        np.concatenate(measured_scores, axis=0).astype(np.float32),
    )


def _measure_anchor_descriptor_points(
    *,
    feature_kind: str,
    alike: AlikeDenseObservationExtractor,
    image_path: Path,
    mapped_feature: np.ndarray,
    predicted_xy: np.ndarray,
    support_descriptors: np.ndarray,
    alike_descriptor_dim: int,
    image_width: int,
    image_height: int,
    search_radius_px: int,
    search_step_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str | None]:
    """Use exactly the descriptor family declared by the persistent map."""

    if str(feature_kind) == RADIO_FINAL_ANCHOR_FEATURE:
        xy, descriptors, scores = _match_radio_final_descriptor_points(
            mapped_feature=mapped_feature,
            predicted_xy=predicted_xy,
            support_descriptors=support_descriptors,
            image_width=int(image_width),
            image_height=int(image_height),
            search_radius_px=int(search_radius_px),
            search_step_px=max(2, int(search_step_px)),
        )
        return xy, descriptors, scores, None
    measured_xy, measured_alike, measured_scores, image_hash = (
        alike.match_descriptor_points(
            image_path,
            predicted_xy,
            np.asarray(support_descriptors, dtype=np.float32)[
                :, : int(alike_descriptor_dim)
            ],
            image_width=int(image_width),
            image_height=int(image_height),
            search_radius_px=int(search_radius_px),
            search_step_px=int(search_step_px),
        )
    )
    measured_radio = (
        _sample_mapped_vfm_at_pixels(
            mapped_feature,
            measured_xy,
            image_width=int(image_width),
            image_height=int(image_height),
        )
        if str(feature_kind) == ALIKE_RADIO_FINAL_ANCHOR_FEATURE
        else None
    )
    measured = compose_anchor_query_descriptors(
        alike_descriptors=measured_alike,
        radio_final_descriptors=measured_radio,
        feature_kind=str(feature_kind),
        expected_dim=int(np.asarray(support_descriptors).shape[1]),
    )
    return measured_xy, measured, measured_scores, image_hash


def _validate_metadata(metadata: Mapping[str, object], name: str) -> None:
    illegal = [key for key in FORBIDDEN_RUNTIME_FLAGS if bool(metadata.get(key))]
    if illegal:
        raise ValueError(f"{name} violates map-only runtime contract: {illegal}")


def _retrieved_anchor_ids(
    maplet_match,
    maplets: VfmSurfaceMapletBank,
) -> np.ndarray:
    maplet_rows = {
        int(maplet_id): int(row)
        for row, maplet_id in enumerate(maplets.maplet_ids.tolist())
    }
    retrieved = {
        int(value)
        for value in maplet_match.candidate_maplet_ids.reshape(-1).tolist()
        if int(value) >= 0
    }
    output: set[int] = set()
    for maplet_id in retrieved:
        row = maplet_rows.get(maplet_id)
        if row is None:
            continue
        start = int(maplets.anchor_offsets[row])
        end = int(maplets.anchor_offsets[row + 1])
        output.update(int(value) for value in maplets.anchor_ids[start:end])
    return np.asarray(sorted(output), dtype=np.int64)


def _retrieved_support_mode_ids(
    maplet_match,
    maplets: VfmSurfaceMapletBank,
) -> tuple[str, ...]:
    row_by_id = {
        int(maplet_id): int(row)
        for row, maplet_id in enumerate(maplets.maplet_ids.tolist())
    }
    output: set[str] = set()
    for maplet_id_value in maplet_match.candidate_maplet_ids.reshape(-1):
        row = row_by_id.get(int(maplet_id_value))
        if row is None:
            continue
        start = int(maplets.view_offsets[row])
        end = int(maplets.view_offsets[row + 1])
        output.update(maplets.view_image_ids[start:end])
    return tuple(sorted(output))


def _fixed_pose_evidence(
    pool,
    result: SurfacePoseResult,
    camera: ColmapCamera,
    config: SurfacePoseConfig,
) -> tuple[float, int]:
    if not result.success or not result.hypotheses or len(pool) == 0:
        return float("-inf"), 0
    score, residuals, posterior = score_surface_pose(
        pool,
        result.pose_w2c,
        camera,
        np.ones((len(pool),), dtype=bool),
        config,
    )
    valid_residuals = np.where(pool.valid_mask, residuals, np.inf)
    identity_supported = np.where(
        pool.valid_mask
        & (
            valid_residuals
            <= float(config.inlier_threshold_px)
        ),
        posterior,
        0.0,
    )
    inlier_count = int(
        np.sum(np.max(identity_supported, axis=1) >= 0.05)
    )
    return float(score / max(len(pool), 1)), inlier_count


def _fixed_evidence_is_usable(
    score: float,
    inlier_count: int,
    config: SurfacePoseConfig,
) -> bool:
    """Require independent fixed-map support before promoting a pose mode."""

    return bool(
        np.isfinite(float(score))
        and int(inlier_count) >= int(config.minimum_selected_inliers)
    )


def _feature_pose_anchor_rows(
    *,
    candidate_anchor_ids: Sequence[int],
    anchors: StableSurfaceAnchorMap,
    descriptor_bank: AnchorLocalDescriptorBank,
    maximum_anchors: int,
) -> np.ndarray:
    anchor_row_by_id = anchors.row_by_id()
    bank_row_by_id = {
        int(anchor_id): int(row)
        for row, anchor_id in enumerate(descriptor_bank.anchor_ids.tolist())
    }
    records: list[tuple[float, int]] = []
    for anchor_id_value in candidate_anchor_ids:
        anchor_id = int(anchor_id_value)
        anchor_row = anchor_row_by_id.get(anchor_id)
        bank_row = bank_row_by_id.get(anchor_id)
        if anchor_row is None or bank_row is None:
            continue
        start = int(descriptor_bank.descriptor_offsets[bank_row])
        end = int(descriptor_bank.descriptor_offsets[bank_row + 1])
        if end <= start:
            continue
        local_quality = float(
            np.median(descriptor_bank.descriptor_quality[start:end])
        )
        quality = (
            max(float(anchors.quality_scores[anchor_row]), 1e-8)
            * max(local_quality, 1e-8)
            * np.log1p(end - start)
        )
        records.append((float(quality), anchor_id))
    records.sort(key=lambda item: (-item[0], item[1]))
    return np.asarray(
        [
            anchor_row_by_id[anchor_id]
            for _quality, anchor_id in records[: int(maximum_anchors)]
        ],
        dtype=np.int64,
    )


def _feature_map_pose_evidence(
    *,
    result: SurfacePoseResult,
    anchor_rows: np.ndarray,
    anchors: StableSurfaceAnchorMap,
    descriptor_bank: AnchorLocalDescriptorBank,
    alike: AlikeDenseObservationExtractor,
    image_path: Path,
    camera: ColmapCamera,
    support_view_id: str | None,
    feature_kind: str,
    mapped_feature: np.ndarray | None = None,
) -> tuple[float, int]:
    """Independent query dense-feature likelihood at pose-projected map anchors."""

    if not result.success or len(anchor_rows) == 0:
        return float("-inf"), 0

    pose = np.asarray(result.pose_w2c, dtype=np.float64).reshape(4, 4)
    xyz = anchors.xyz[np.asarray(anchor_rows, dtype=np.int64)]
    camera_xyz = xyz @ pose[:3, :3].T + pose[:3, 3]
    matrix, distortion = camera_matrix_and_distortion(camera)
    rvec, _jacobian = cv2.Rodrigues(pose[:3, :3])
    projected, _jacobian = cv2.projectPoints(
        xyz,
        rvec,
        pose[:3, 3],
        matrix,
        distortion,
    )
    projected = projected.reshape(-1, 2)
    visible = (
        (camera_xyz[:, 2] > 1e-6)
        & (projected[:, 0] >= 0.0)
        & (projected[:, 0] <= float(camera.width - 1))
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] <= float(camera.height - 1))
    )
    rows = np.flatnonzero(visible)
    if len(rows) < 8:
        return float("-inf"), int(len(rows))
    if str(feature_kind) == RADIO_FINAL_ANCHOR_FEATURE:
        if mapped_feature is None:
            raise ValueError(
                "RADIO-only pose evidence requires the query feature field"
            )
        query_descriptors = _sample_mapped_vfm_at_pixels(
            mapped_feature,
            projected[rows],
            image_width=int(camera.width),
            image_height=int(camera.height),
        )
        detector_scores = np.ones((len(rows),), dtype=np.float32)
    else:
        query_alike, detector_scores, _image_hash = alike.sample_points(
            image_path,
            projected[rows],
            image_width=int(camera.width),
            image_height=int(camera.height),
        )
        query_radio = (
            _sample_mapped_vfm_at_pixels(
                mapped_feature,
                projected[rows],
                image_width=int(camera.width),
                image_height=int(camera.height),
            )
            if str(feature_kind) == ALIKE_RADIO_FINAL_ANCHOR_FEATURE
            and mapped_feature is not None
            else None
        )
        query_descriptors = compose_anchor_query_descriptors(
            alike_descriptors=query_alike,
            radio_final_descriptors=query_radio,
            feature_kind=str(feature_kind),
            expected_dim=int(descriptor_bank.feature_dim),
        )
    if int(query_descriptors.shape[1]) != int(descriptor_bank.feature_dim):
        raise ValueError(
            "pose-evidence query and map descriptor dimensions differ"
        )
    bank_row_by_id = {
        int(anchor_id): int(row)
        for row, anchor_id in enumerate(descriptor_bank.anchor_ids.tolist())
    }
    camera_center = -pose[:3, :3].T @ pose[:3, 3]
    view_directions = camera_center[None, :] - xyz[rows]
    view_directions /= np.maximum(
        np.linalg.norm(view_directions, axis=1, keepdims=True),
        1e-12,
    )
    mixture_log_match: list[float] = []
    for query_row, anchor_row in enumerate(anchor_rows[rows].tolist()):
        anchor_id = int(anchors.anchor_ids[int(anchor_row)])
        bank_row = bank_row_by_id[anchor_id]
        start = int(descriptor_bank.descriptor_offsets[bank_row])
        end = int(descriptor_bank.descriptor_offsets[bank_row + 1])
        descriptor_rows = np.arange(start, end, dtype=np.int64)
        if support_view_id is not None:
            mode_rows = descriptor_rows[
                np.asarray(
                    [
                        descriptor_bank.support_image_ids[int(row)]
                        == str(support_view_id)
                        for row in descriptor_rows.tolist()
                    ],
                    dtype=bool,
                )
            ]
            # A mode-specific hypothesis with no support descriptor is
            # missing evidence.  Falling back to descriptors from another
            # view mode leaks appearance across competing explanations and
            # makes repeated structures look artificially certain.
            if not len(mode_rows):
                mixture_log_match.append(float(np.log(1e-3)))
                continue
            descriptor_rows = mode_rows
        edge_similarity = (
            descriptor_bank.descriptors[descriptor_rows]
            @ query_descriptors[query_row]
        )
        edge_log_match = -np.logaddexp(
            0.0,
            -(edge_similarity - 0.65) / 0.08,
        )
        prior_logits = np.log(
            np.maximum(
                descriptor_bank.descriptor_quality[descriptor_rows],
                1e-8,
            )
        ).astype(np.float64)
        support_directions = descriptor_bank.support_view_directions[
            descriptor_rows
        ].astype(np.float64)
        valid_direction = (
            np.linalg.norm(support_directions, axis=1) > 0.5
        )
        if np.any(valid_direction):
            geometry_cosine = (
                support_directions @ view_directions[query_row]
            )
            prior_logits += np.where(
                valid_direction,
                geometry_cosine / 0.10,
                np.min(geometry_cosine[valid_direction]) / 0.10,
            )
        log_weights = prior_logits - logsumexp(prior_logits)
        mixture_log_match.append(
            float(logsumexp(log_weights + edge_log_match))
        )
    log_match = np.asarray(mixture_log_match, dtype=np.float64)
    # Dense detector score contributes only a bounded repeatability term.
    score_term = np.log1p(
        np.maximum(np.asarray(detector_scores, dtype=np.float64), 0.0)
        * 1000.0
    )
    repeatability = np.tanh(score_term / 3.0)
    visible_fraction = float(len(rows) / max(len(anchor_rows), 1))
    score = float(
        np.mean(log_match + 0.15 * repeatability)
        - 0.25 * (1.0 - visible_fraction)
    )
    supported = int(
        np.sum((log_match >= np.log(0.5)) & (repeatability >= 0.05))
    )
    return score, supported


def _strict_pose_gate_group_count(
    matcher_diagnostics: dict[str, object],
) -> int:
    """Return groups whose retained identity mass beats conditional null.

    A merely matchable group has non-zero anchor mass but may still be almost
    entirely null.  Counting those groups as confident forces PnP to sample
    from ambiguous detector nodes and can turn abstainable queries into
    arbitrary poses.
    """

    return int(
        matcher_diagnostics["conditionally_confident_group_count"]
    )


def _surface_pose_gate_passes(
    matcher_diagnostics: Mapping[str, object],
    *,
    minimum_confident_groups: int,
    maximum_mean_null_probability: float,
) -> tuple[bool, str]:
    """Require both conditional identity and absolute retained mass.

    Conditioning on ``candidate / (candidate + null)`` is useful for deciding
    whether a group is *matchable*, but it is not sufficient for starting PnP:
    a group with candidate mass 1e-6 and null mass 0.999999 would otherwise
    look perfectly confident after conditioning.  The absolute null mass is
    therefore a separate, conservative gate.  Keeping the two tests separate
    also makes the reason for an abstention auditable.
    """

    minimum = int(minimum_confident_groups)
    maximum_null = float(maximum_mean_null_probability)
    if minimum <= 0 or not 0.0 <= maximum_null <= 1.0:
        raise ValueError("pose-gate limits are invalid")
    confident = int(
        matcher_diagnostics.get("conditionally_confident_group_count", 0)
    )
    mean_null = float(
        matcher_diagnostics.get("mean_null_probability", 1.0)
    )
    if not np.isfinite(mean_null):
        return False, "nonfinite_mean_null_probability"
    if confident < minimum:
        return False, "insufficient_conditionally_confident_groups"
    if mean_null > maximum_null:
        return False, "absolute_null_mass_gate"
    return True, "passed"


def _pose_projected_anchor_measurements(
    *,
    pose_w2c: np.ndarray,
    candidate_anchor_ids: Sequence[int],
    support_view_id: str | None,
    anchors: StableSurfaceAnchorMap,
    descriptor_bank: AnchorLocalDescriptorBank,
    camera: ColmapCamera,
    maximum_anchors: int,
    projection_margin_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project high-quality map anchors and retain a spatially balanced set."""

    import cv2

    anchor_row_by_id = anchors.row_by_id()
    bank_row_by_id = {
        int(anchor_id): int(row)
        for row, anchor_id in enumerate(descriptor_bank.anchor_ids.tolist())
    }
    records: list[tuple[float, int, np.ndarray]] = []
    for anchor_id_value in candidate_anchor_ids:
        anchor_id = int(anchor_id_value)
        anchor_row = anchor_row_by_id.get(anchor_id)
        bank_row = bank_row_by_id.get(anchor_id)
        if anchor_row is None or bank_row is None:
            continue
        descriptor_rows = np.arange(
            int(descriptor_bank.descriptor_offsets[bank_row]),
            int(descriptor_bank.descriptor_offsets[bank_row + 1]),
            dtype=np.int64,
        )
        if len(descriptor_rows) == 0:
            continue
        if support_view_id is not None:
            mode_rows = descriptor_rows[
                np.asarray(
                    [
                        descriptor_bank.support_image_ids[int(row)]
                        == str(support_view_id)
                        for row in descriptor_rows.tolist()
                    ],
                    dtype=bool,
                )
            ]
            if not len(mode_rows):
                continue
            descriptor_rows = mode_rows
        weights = np.maximum(
            descriptor_bank.descriptor_quality[descriptor_rows], 1e-8
        )
        prototype = np.sum(
            descriptor_bank.descriptors[descriptor_rows] * weights[:, None],
            axis=0,
        )
        prototype /= max(float(np.linalg.norm(prototype)), 1e-8)
        quality = (
            max(float(anchors.quality_scores[anchor_row]), 1e-8)
            * max(float(np.median(weights)), 1e-8)
            * np.log1p(len(descriptor_rows))
        )
        records.append((float(quality), anchor_id, prototype.astype(np.float32)))
    if not records:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, descriptor_bank.feature_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    ids = np.asarray([item[1] for item in records], dtype=np.int64)
    rows = np.asarray([anchor_row_by_id[int(value)] for value in ids], dtype=np.int64)
    xyz = anchors.xyz[rows]
    camera_xyz = xyz @ pose[:3, :3].T + pose[:3, 3]
    matrix, distortion = camera_matrix_and_distortion(camera)
    rvec, _jacobian = cv2.Rodrigues(pose[:3, :3])
    projected, _jacobian = cv2.projectPoints(
        xyz, rvec, pose[:3, 3], matrix, distortion
    )
    projected = projected.reshape(-1, 2)
    margin = float(projection_margin_px)
    visible = (
        (camera_xyz[:, 2] > 1e-6)
        & (projected[:, 0] >= -margin)
        & (projected[:, 0] <= float(camera.width - 1) + margin)
        & (projected[:, 1] >= -margin)
        & (projected[:, 1] <= float(camera.height - 1) + margin)
    )
    visible_rows = np.flatnonzero(visible)
    ranked = sorted(
        visible_rows.tolist(),
        key=lambda row: (-records[row][0], int(ids[row])),
    )
    cell_quota = max(1, int(np.ceil(int(maximum_anchors) / 64.0)))
    cell_counts: dict[tuple[int, int], int] = {}
    selected: list[int] = []
    for use_quota in (True, False):
        for row in ranked:
            if row in selected or len(selected) >= int(maximum_anchors):
                continue
            cell = (
                int(np.clip(projected[row, 1] / max(camera.height, 1) * 8, 0, 7)),
                int(np.clip(projected[row, 0] / max(camera.width, 1) * 8, 0, 7)),
            )
            if use_quota and cell_counts.get(cell, 0) >= cell_quota:
                continue
            selected.append(int(row))
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
    selected_rows = np.asarray(selected, dtype=np.int64)
    return (
        projected[selected_rows].astype(np.float32),
        np.stack([records[row][2] for row in selected]).astype(np.float32),
        ids[selected_rows],
    )


def _unique_featuremetric_measurements(
    *,
    xy: np.ndarray,
    descriptors: np.ndarray,
    similarities: np.ndarray,
    anchor_ids: np.ndarray,
    minimum_similarity: float,
    minimum_separation_px: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-np.asarray(similarities), kind="mergesort")
    selected: list[int] = []
    for row in order.tolist():
        if float(similarities[row]) < float(minimum_similarity):
            continue
        if selected:
            distance = np.linalg.norm(
                xy[np.asarray(selected, dtype=np.int64)] - xy[row], axis=1
            )
            if float(np.min(distance)) < float(minimum_separation_px):
                continue
        selected.append(int(row))
    rows = np.asarray(selected, dtype=np.int64)
    return xy[rows], descriptors[rows], similarities[rows], anchor_ids[rows]


def _result_key(result: SurfacePoseResult) -> tuple[float, ...]:
    if not result.hypotheses:
        return (float(result.success), float("-inf"), 0.0, float("-inf"))
    best = result.hypotheses[0]
    verification_count = max(int(np.sum(result.verification_mask)), 1)
    fit_count = max(int(np.sum(result.fit_mask)), 1)
    return (
        float(result.success),
        float(best.inlier_count),
        float(best.verification_score) / verification_count,
        float(best.generation_score) / fit_count,
    )


def _pose_record(
    *,
    image_id: str,
    result: SurfacePoseResult,
    branch: str,
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
    hypotheses = [
        {
            "pose_w2c": hypothesis.pose_w2c.reshape(-1).tolist(),
            "generation_score": float(hypothesis.generation_score),
            "verification_score": float(hypothesis.verification_score),
            "inlier_count": int(hypothesis.inlier_count),
            "source": str(hypothesis.source),
        }
        for hypothesis in result.hypotheses[:16]
    ]
    return {
        "image_id": str(image_id),
        "success": bool(result.success),
        "pose_w2c": (
            result.pose_w2c.reshape(-1).tolist() if result.success else None
        ),
        "failure_reason": result.failure_reason,
        "branch": str(branch),
        "fit_group_count": int(np.sum(result.fit_mask)),
        "verification_group_count": int(np.sum(result.verification_mask)),
        "hypothesis_count": len(result.hypotheses),
        "hypotheses": hypotheses,
        "diagnostics": dict(diagnostics),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json)
    if (output_path.exists() or summary_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite localization outputs")
    if (
        int(args.local_candidate_top_k) < int(args.local_top_k)
        or int(args.maplet_top_k) <= 0
        or int(args.anchor_top_l) <= 0
        or int(args.verification_anchor_top_l) <= 0
        or int(args.maximum_feature_modes) <= 0
        or int(args.maximum_global_feature_modes) <= 0
        or float(args.feature_mode_minimum_similarity) <= 0.0
        or not 0.0 <= float(args.set_pose_maximum_mean_null) <= 1.0
        or int(args.set_pose_minimum_confident_groups) <= 0
    ):
        raise ValueError("localization candidate limits are invalid")

    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate()
    records = list(manifest.records)
    if int(args.max_queries) > 0:
        records = records[: int(args.max_queries)]
    if not records:
        raise ValueError("query manifest is empty")
    query_root = Path(args.query_image_root).resolve(strict=True)
    query_image_paths: dict[str, Path] = {}
    for record in records:
        relative = Path(record.image_id)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"query image escapes query_image_root: {record.image_id}"
            )
        path = query_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"query image is absent: {path}")
        query_image_paths[record.image_id] = path

    maplets = VfmSurfaceMapletBank.load_npz(Path(args.maplets))
    anchors = StableSurfaceAnchorMap.load_npz(Path(args.anchors))
    local_bank = AnchorLocalDescriptorBank.load_npz(
        Path(args.local_descriptor_bank)
    )
    surface_features = Vfm2DgsObservationBank.load_npz(
        Path(args.radio_surface_feature_bank)
    )
    surface_feature_index = VfmSurfaceObservationIndex.from_bank(
        surface_features
    )
    mapper, mapper_metadata = load_surface_maplet_mapper(
        Path(args.surface_mapper_checkpoint), device=str(args.device)
    )
    anchor_matcher, anchor_matcher_payload = load_surface_anchor_set_matcher(
        Path(args.surface_anchor_matcher_checkpoint), device=str(args.device)
    )
    for metadata, name in (
        (maplets.metadata, "maplet bank"),
        (anchors.metadata, "anchor map"),
        (local_bank.metadata, "local descriptor bank"),
        (surface_features.metadata, "RADIO surface feature bank"),
        (mapper_metadata, "surface mapper"),
        (anchor_matcher_payload.get("metadata") or {}, "surface anchor matcher"),
    ):
        _validate_metadata(metadata, name)
    if int(local_bank.feature_dim) != int(anchor_matcher.config.descriptor_dim):
        raise ValueError("anchor matcher and local descriptor dimensions differ")
    sparse_anchor_feature = anchor_feature_kind(local_bank.metadata)
    sparse_branch_label = (
        "radio_final_at_alike_detections"
        if sparse_anchor_feature == RADIO_FINAL_ANCHOR_FEATURE
        else "alike_anchor_features"
    )
    initial_sparse_branch = (
        f"radio_maplet_{sparse_branch_label}_set_ot_grouped_pnp"
    )
    if bool(args.augment_alike_with_radio_final) and (
        sparse_anchor_feature != ALIKE_RADIO_FINAL_ANCHOR_FEATURE
    ):
        raise ValueError(
            "--augment_alike_with_radio_final conflicts with the explicit "
            f"map descriptor contract {sparse_anchor_feature!r}"
        )
    if set(local_bank.anchor_ids.tolist()) - set(anchors.anchor_ids.tolist()):
        raise ValueError("local descriptor bank contains unknown surface anchors")

    fallback_camera = _camera_from_model(Path(args.camera_model_dir))
    if str(args.query_camera_manifest):
        query_camera_by_image, intrinsic_audit = (
            _load_query_camera_manifest(Path(args.query_camera_manifest))
        )
        missing_query_cameras = sorted(
            {record.image_id for record in records}
            - set(query_camera_by_image)
        )
        if missing_query_cameras:
            raise ValueError(
                f"query camera manifest is missing {missing_query_cameras[0]}"
            )
    else:
        query_camera_by_image = {}
        intrinsic_audit = {
            "uses_per_query_calibration": False,
            "fallback_is_parameter_median": True,
        }
    pool_sizes = tuple(
        int(value) for value in mapper_metadata.get("pool_sizes", (1, 3, 5, 9))
    )
    pool_weights = tuple(
        float(value)
        for value in mapper_metadata.get(
            "pool_weights", (0.4, 0.3, 0.2, 0.1)
        )
    )
    region_config = RadioFinalRegionConfig(
        pool_sizes=pool_sizes, pool_weights=pool_weights
    )
    maplet_config = SurfaceMapletMatchConfig(
        top_k=int(args.maplet_top_k),
        maximum_support_views=int(args.maximum_support_views),
        maximum_layout_models=int(args.maximum_layout_models),
        enable_support_layout=not bool(args.disable_maplet_support_layout),
    )
    query_aligned_maplet_config = SurfaceMapletMatchConfig(
        top_k=int(args.maplet_top_k),
        maximum_support_views=int(args.maximum_support_views),
        maximum_layout_models=int(args.maximum_layout_models),
        enable_support_layout=False,
    )
    pose_config = SurfacePoseConfig(
        hypothesis_count=int(args.hypothesis_count),
        minimum_fit_groups=int(args.minimum_fit_groups),
        minimum_verification_groups=int(args.minimum_verification_groups),
        heldout_stride=int(args.heldout_stride),
    )
    alike = AlikeDenseObservationExtractor(
        device=str(args.device),
        matcha_repo=Path(args.matcha_repo),
        model_name=str(args.alike_model_name),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    successes = 0
    failure_counts: dict[str, int] = {}
    branch_counts: dict[str, int] = {}
    query_file_audit: list[dict[str, object]] = []
    with output_path.open("w") as output:
        for query_index, record in enumerate(records):
            query_started = time.time()
            camera = query_camera_by_image.get(
                record.image_id, fallback_camera
            )
            stage_started = time.time()
            raw = _load_raw_final(
                record.token_path, str(args.radio_final_layer)
            )
            mapped = mapper.project(raw).coarse_descriptors
            _indices, region_xy = select_spatially_balanced_radio_final_regions(
                raw,
                grid_rows=int(args.query_region_rows),
                grid_cols=int(args.query_region_cols),
                regions_per_cell=int(args.query_regions_per_cell),
            )
            region_descriptors = encode_radio_final_regions(
                mapped, region_xy, region_config
            )
            grid_size = (int(mapped.shape[2]), int(mapped.shape[1]))
            maplet_match = match_radio_final_regions_to_maplets(
                region_xy,
                region_descriptors,
                grid_size,
                maplets,
                maplet_config,
            )
            coarse_runtime = time.time() - stage_started
            stage_started = time.time()
            image_path = query_image_paths[record.image_id]
            detected = alike.detect(
                image_path,
                image_width=int(camera.width),
                image_height=int(camera.height),
                top_k=int(args.local_top_k),
                candidate_top_k=int(args.local_candidate_top_k),
                nms_radius_px=float(args.local_nms_radius_px),
                grid_rows=8,
                grid_cols=8,
            )
            detected_radio = (
                _sample_mapped_vfm_at_pixels(
                    mapped,
                    detected.xy,
                    image_width=int(camera.width),
                    image_height=int(camera.height),
                )
                if sparse_anchor_feature
                in (
                    ALIKE_RADIO_FINAL_ANCHOR_FEATURE,
                    RADIO_FINAL_ANCHOR_FEATURE,
                )
                else None
            )
            query = LocalFeatureFrame(
                image_id=record.image_id,
                keypoints_xy=detected.xy,
                descriptors=compose_anchor_query_descriptors(
                    alike_descriptors=detected.descriptors,
                    radio_final_descriptors=detected_radio,
                    feature_kind=sparse_anchor_feature,
                    expected_dim=int(local_bank.feature_dim),
                ),
                scores=detected.scores,
            )
            query_grid_xy = query.keypoints_xy * np.asarray(
                [
                    max(grid_size[0] - 1, 1) / max(int(camera.width) - 1, 1),
                    max(grid_size[1] - 1, 1) / max(int(camera.height) - 1, 1),
                ],
                dtype=np.float32,
            )
            query_aligned_region_descriptors = encode_radio_final_regions(
                mapped,
                query_grid_xy,
                region_config,
            )
            query_aligned_maplet_match = match_radio_final_regions_to_maplets(
                query_grid_xy,
                query_aligned_region_descriptors,
                grid_size,
                maplets,
                query_aligned_maplet_config,
            )
            query_feature_runtime = time.time() - stage_started
            stage_started = time.time()
            anchor_pool, matcher_diagnostics = match_query_to_surface_anchors(
                model=anchor_matcher,
                query=query,
                query_image_size=(int(camera.width), int(camera.height)),
                query_region_xy=region_xy,
                query_region_grid_size=grid_size,
                maplet_match=maplet_match,
                maplets=maplets,
                anchors=anchors,
                descriptor_bank=local_bank,
                device=str(args.device),
                top_l=int(args.anchor_top_l),
                query_aligned_maplet_match=query_aligned_maplet_match,
                feature_preferred_base_fill_target=(
                    int(
                        args.matcher_feature_preferred_base_fill_target
                    )
                    if int(
                        args.matcher_feature_preferred_base_fill_target
                    )
                    > 0
                    else None
                ),
            )
            anchor_set_runtime = time.time() - stage_started
            stage_started = time.time()
            confident_set_groups = _strict_pose_gate_group_count(
                matcher_diagnostics
            )
            mean_set_null = (
                float(np.mean(anchor_pool.null_probabilities))
                if len(anchor_pool)
                else 1.0
            )
            set_pose_gate_passed, set_pose_gate_reason = _surface_pose_gate_passes(
                matcher_diagnostics,
                minimum_confident_groups=int(
                    args.set_pose_minimum_confident_groups
                ),
                maximum_mean_null_probability=float(
                    args.set_pose_maximum_mean_null
                ),
            )
            if set_pose_gate_passed:
                initial_result = generate_grouped_surface_pose_hypotheses(
                    anchor_pool, camera, pose_config
                )
            else:
                initial_result = SurfacePoseResult(
                    success=False,
                    pose_w2c=np.eye(4, dtype=np.float64),
                    hypotheses=(),
                    fit_mask=np.zeros((len(anchor_pool),), dtype=bool),
                    verification_mask=np.zeros(
                        (len(anchor_pool),), dtype=bool
                    ),
                    failure_reason=f"set_matcher_confidence_gate:{set_pose_gate_reason}",
                )
            initial_pose_runtime = time.time() - stage_started
            stage_started = time.time()
            fixed_verification_pool = (
                build_maplet_conditioned_surface_anchor_candidate_pool(
                    query,
                    query_image_size=(
                        int(camera.width),
                        int(camera.height),
                    ),
                    query_region_xy=region_xy,
                    query_region_grid_size=grid_size,
                    maplet_match=maplet_match,
                    maplets=maplets,
                    descriptor_bank=local_bank,
                    anchors=anchors,
                    top_l=int(args.verification_anchor_top_l),
                    maximum_maplets_per_region=3,
                )
            )
            fixed_pool_runtime = time.time() - stage_started
            # Do not let a PnP-generated success escape before it has passed
            # the independent fixed-map evidence gate below.  The initial
            # result is only a proposal at this point.
            result = (
                initial_result
                if not initial_result.success
                else SurfacePoseResult(
                    success=False,
                    pose_w2c=np.eye(4, dtype=np.float64),
                    hypotheses=(),
                    fit_mask=initial_result.fit_mask,
                    verification_mask=initial_result.verification_mask,
                    failure_reason="fixed_map_evidence_gate",
                )
            )
            selection_pool = fixed_verification_pool
            branch = initial_sparse_branch
            layout_pool_size = 0
            layout_success_count = 0
            selected_layout_view_id = None
            selected_layout_score = None
            refinement_pool_size = 0
            retrieved_anchor_ids = _retrieved_anchor_ids(maplet_match, maplets)
            feature_pose_anchor_rows = _feature_pose_anchor_rows(
                candidate_anchor_ids=retrieved_anchor_ids,
                anchors=anchors,
                descriptor_bank=local_bank,
                maximum_anchors=int(
                    args.feature_pose_evidence_maximum_anchors
                ),
            )
            allowed_mode_ids = _retrieved_support_mode_ids(
                maplet_match, maplets
            )
            if bool(args.disable_layout_guided_measurement):
                feature_mode_ids = ()
                feature_mode_scores = np.zeros((0,), dtype=np.float32)
            else:
                feature_mode_ids, feature_mode_scores = (
                    select_vfm_surface_feature_modes(
                        mapped,
                        surface_features,
                        maximum_global_modes=int(
                            args.maximum_global_feature_modes
                        ),
                        maximum_modes=int(args.maximum_feature_modes),
                        allowed_mode_ids=allowed_mode_ids,
                        observation_index=surface_feature_index,
                        spatial_layout_rerank=bool(
                            args.feature_mode_spatial_rerank
                        ),
                        minimum_layout_similarity=float(
                            args.feature_mode_minimum_similarity
                        ),
                    )
                )
            feature_mode_selection_runtime = (
                time.time() - stage_started - fixed_pool_runtime
            )
            stage_started = time.time()
            pose_candidates: list[
                tuple[
                    float,
                    int,
                    float,
                    int,
                    SurfacePoseResult,
                    str,
                    str | None,
                ]
            ] = []
            initial_fixed_score, initial_fixed_inliers = _fixed_pose_evidence(
                fixed_verification_pool,
                initial_result,
                camera,
                pose_config,
            )
            if initial_result.success and _fixed_evidence_is_usable(
                initial_fixed_score, initial_fixed_inliers, pose_config
            ):
                pose_candidates.append(
                    (
                        initial_fixed_score,
                        initial_fixed_inliers,
                        float("-inf"),
                        -1,
                        initial_result,
                        initial_sparse_branch,
                        None,
                    )
                )
            feature_mode_diagnostics: list[dict[str, object]] = []
            for feature_mode_rank, (
                layout_view_id,
                feature_mode_score,
            ) in enumerate(zip(feature_mode_ids, feature_mode_scores.tolist())):
                (
                    layout_matrix,
                    layout_translation,
                    layout_inliers,
                    layout_median_residual,
                ) = estimate_vfm_query_to_support_layout(
                    mapped,
                    surface_features,
                    str(layout_view_id),
                    minimum_similarity=float(
                        args.feature_mode_minimum_similarity
                    ),
                    observation_index=surface_feature_index,
                )
                mode_diagnostic: dict[str, object] = {
                    "rank": int(feature_mode_rank),
                    "mode_id": str(layout_view_id),
                    "vfm_score": float(feature_mode_score),
                    "layout_inliers": int(layout_inliers),
                    "layout_median_residual_tokens": (
                        float(layout_median_residual)
                        if np.isfinite(layout_median_residual)
                        else None
                    ),
                    "seed_count": 0,
                    "pose_success": False,
                    "pose_inliers": 0,
                    "fixed_evidence_per_group": None,
                    "fixed_evidence_inliers": 0,
                }
                feature_mode_diagnostics.append(mode_diagnostic)
                if (
                    layout_matrix is None
                    or layout_translation is None
                ):
                    continue
                (
                    predicted_xy,
                    support_descriptors,
                    seed_anchor_ids,
                ) = predict_vfm_support_anchor_query_points(
                    query_image_size=(int(camera.width), int(camera.height)),
                    query_grid_size=grid_size,
                    support_view_id=str(layout_view_id),
                    query_to_support_matrix=layout_matrix,
                    query_to_support_translation=layout_translation,
                    anchors=anchors,
                    descriptor_bank=local_bank,
                    allowed_anchor_ids=retrieved_anchor_ids,
                    maximum_points=int(args.layout_guided_maximum_points),
                )
                mode_diagnostic["seed_count"] = int(len(seed_anchor_ids))
                if len(predicted_xy) >= 4:
                    (
                        measured_xy,
                        measured_descriptors,
                        measured_scores,
                        _measured_image_hash,
                    ) = _measure_anchor_descriptor_points(
                        feature_kind=sparse_anchor_feature,
                        alike=alike,
                        image_path=image_path,
                        mapped_feature=mapped,
                        predicted_xy=predicted_xy,
                        support_descriptors=support_descriptors,
                        alike_descriptor_dim=int(
                            detected.descriptors.shape[1]
                        ),
                        image_width=int(camera.width),
                        image_height=int(camera.height),
                        search_radius_px=int(
                            args.layout_guided_search_radius_px
                        ),
                        search_step_px=1,
                    )
                    layout_query = LocalFeatureFrame(
                        image_id=record.image_id,
                        keypoints_xy=measured_xy,
                        descriptors=measured_descriptors,
                        scores=measured_scores,
                    )
                    layout_pool = build_seeded_surface_anchor_candidate_pool(
                        layout_query,
                        seed_anchor_ids,
                        descriptor_bank=local_bank,
                        anchors=anchors,
                        top_l=int(args.anchor_top_l),
                        seed_prior_logit=float(
                            args.layout_guided_seed_prior_logit
                        ),
                    )
                    layout_pool_size += len(layout_pool)
                    layout_result = generate_grouped_surface_pose_hypotheses(
                        layout_pool, camera, pose_config
                    )
                    layout_success_count += int(layout_result.success)
                    mode_diagnostic["pose_success"] = bool(
                        layout_result.success
                    )
                    mode_diagnostic["pose_inliers"] = int(
                        layout_result.hypotheses[0].inlier_count
                        if layout_result.hypotheses
                        else 0
                    )
                    fixed_score, fixed_inliers = _fixed_pose_evidence(
                        fixed_verification_pool,
                        layout_result,
                        camera,
                        pose_config,
                    )
                    mode_diagnostic["fixed_evidence_per_group"] = (
                        float(fixed_score)
                        if np.isfinite(fixed_score)
                        else None
                    )
                    mode_diagnostic["fixed_evidence_inliers"] = int(
                        fixed_inliers
                    )
                    if layout_result.success and _fixed_evidence_is_usable(
                        fixed_score, fixed_inliers, pose_config
                    ):
                        pose_candidates.append(
                            (
                                fixed_score,
                                fixed_inliers,
                                float(feature_mode_score),
                                -int(feature_mode_rank),
                                layout_result,
                                (
                                    "radio_maplet_feature_mode_"
                                    f"{sparse_branch_label}_measurement"
                                ),
                                str(layout_view_id),
                            )
                        )
            if not pose_candidates and feature_mode_ids:
                for feature_mode_rank, (
                    layout_view_id,
                    feature_mode_score,
                ) in enumerate(
                    zip(feature_mode_ids[:4], feature_mode_scores[:4].tolist())
                ):
                    (
                        layout_matrix,
                        layout_translation,
                        _layout_inliers,
                        _layout_median_residual,
                    ) = estimate_vfm_query_to_support_layout(
                        mapped,
                        surface_features,
                        str(layout_view_id),
                        minimum_similarity=float(
                            args.feature_mode_minimum_similarity
                        ),
                        observation_index=surface_feature_index,
                    )
                    if (
                        layout_matrix is None
                        or layout_translation is None
                    ):
                        continue
                    (
                        predicted_xy,
                        support_descriptors,
                        seed_anchor_ids,
                    ) = predict_vfm_support_anchor_query_points(
                        query_image_size=(
                            int(camera.width),
                            int(camera.height),
                        ),
                        query_grid_size=grid_size,
                        support_view_id=str(layout_view_id),
                        query_to_support_matrix=layout_matrix,
                        query_to_support_translation=layout_translation,
                        anchors=anchors,
                        descriptor_bank=local_bank,
                        allowed_anchor_ids=None,
                        maximum_points=int(
                            args.layout_guided_maximum_points
                        ),
                    )
                    feature_mode_diagnostics[
                        feature_mode_rank
                    ]["expanded_seed_count"] = int(len(seed_anchor_ids))
                    if len(predicted_xy) < 4:
                        continue
                    (
                        measured_xy,
                        measured_descriptors,
                        measured_scores,
                        _measured_image_hash,
                    ) = _measure_anchor_descriptor_points(
                        feature_kind=sparse_anchor_feature,
                        alike=alike,
                        image_path=image_path,
                        mapped_feature=mapped,
                        predicted_xy=predicted_xy,
                        support_descriptors=support_descriptors,
                        alike_descriptor_dim=int(
                            detected.descriptors.shape[1]
                        ),
                        image_width=int(camera.width),
                        image_height=int(camera.height),
                        search_radius_px=int(
                            args.layout_guided_search_radius_px
                        ),
                        search_step_px=1,
                    )
                    expanded_query = LocalFeatureFrame(
                        image_id=record.image_id,
                        keypoints_xy=measured_xy,
                        descriptors=measured_descriptors,
                        scores=measured_scores,
                    )
                    expanded_pool = (
                        build_seeded_surface_anchor_candidate_pool(
                            expanded_query,
                            seed_anchor_ids,
                            descriptor_bank=local_bank,
                            anchors=anchors,
                            top_l=int(args.anchor_top_l),
                            seed_prior_logit=float(
                                args.layout_guided_seed_prior_logit
                            ),
                        )
                    )
                    layout_pool_size += len(expanded_pool)
                    expanded_result = (
                        generate_grouped_surface_pose_hypotheses(
                            expanded_pool, camera, pose_config
                        )
                    )
                    layout_success_count += int(expanded_result.success)
                    feature_mode_diagnostics[
                        feature_mode_rank
                    ]["expanded_pose_success"] = bool(
                        expanded_result.success
                    )
                    feature_mode_diagnostics[
                        feature_mode_rank
                    ]["expanded_pose_inliers"] = int(
                        expanded_result.hypotheses[0].inlier_count
                        if expanded_result.hypotheses
                        else 0
                    )
                    fixed_score, fixed_inliers = _fixed_pose_evidence(
                        fixed_verification_pool,
                        expanded_result,
                        camera,
                        pose_config,
                    )
                    if expanded_result.success and _fixed_evidence_is_usable(
                        fixed_score, fixed_inliers, pose_config
                    ):
                        pose_candidates.append(
                            (
                                fixed_score,
                                fixed_inliers,
                                float(feature_mode_score),
                                -int(feature_mode_rank),
                                expanded_result,
                                (
                                    "radio_feature_mode_expanded_"
                                    f"{sparse_branch_label}_fallback"
                                ),
                                str(layout_view_id),
                            )
                        )
            direct_surface_fallback_group_count = 0
            direct_surface_fallback_inliers = 0
            if bool(args.always_direct_surface_candidate) or not pose_candidates:
                (
                    direct_surface_pool,
                    direct_surface_mode_ids,
                    direct_surface_mode_scores,
                ) = build_vfm_surface_observation_candidate_pool(
                    mapped,
                    query_image_size=(
                        int(camera.width),
                        int(camera.height),
                    ),
                    observation_bank=surface_features,
                    maximum_global_support_views=int(
                        args.maximum_global_feature_modes
                    ),
                    maximum_support_views=min(
                        4, int(args.maximum_feature_modes)
                    ),
                    maximum_query_points=768,
                    top_l=5,
                    minimum_similarity=float(
                        args.feature_mode_minimum_similarity
                    ),
                    observation_index=surface_feature_index,
                )
                direct_surface_fallback_group_count = len(
                    direct_surface_pool
                )
                direct_surface_result = (
                    generate_grouped_surface_pose_hypotheses(
                        direct_surface_pool, camera, pose_config
                    )
                )
                direct_surface_fallback_inliers = int(
                    direct_surface_result.hypotheses[0].inlier_count
                    if direct_surface_result.hypotheses
                    else 0
                )
                fixed_score, fixed_inliers = _fixed_pose_evidence(
                    fixed_verification_pool,
                    direct_surface_result,
                    camera,
                    pose_config,
                )
                if direct_surface_result.success and _fixed_evidence_is_usable(
                    fixed_score, fixed_inliers, pose_config
                ):
                    pose_candidates.append(
                        (
                            fixed_score,
                            fixed_inliers,
                            float(
                                direct_surface_mode_scores[0]
                                if len(direct_surface_mode_scores)
                                else -np.inf
                            ),
                            0,
                            direct_surface_result,
                            (
                                "radio_final_2dgs_surface_observation_"
                                "candidate"
                            ),
                            (
                                str(direct_surface_mode_ids[0])
                                if direct_surface_mode_ids
                                else None
                            ),
                        )
                    )
            feature_pose_evidence: list[dict[str, object]] = []
            if pose_candidates:
                scored_pose_candidates = []
                for candidate_index, candidate in enumerate(pose_candidates):
                    (
                        fixed_score,
                        fixed_inliers,
                        vfm_score,
                        rank_score,
                        candidate_result,
                        candidate_branch,
                        candidate_view_id,
                    ) = candidate
                    feature_score, feature_support = (
                        _feature_map_pose_evidence(
                            result=candidate_result,
                            anchor_rows=feature_pose_anchor_rows,
                            anchors=anchors,
                            descriptor_bank=local_bank,
                            alike=alike,
                            image_path=image_path,
                            camera=camera,
                            support_view_id=candidate_view_id,
                            feature_kind=sparse_anchor_feature,
                            mapped_feature=mapped,
                        )
                    )
                    combined_score = float(fixed_score)
                    if np.isfinite(feature_score):
                        combined_score += float(
                            args.feature_pose_evidence_weight
                        ) * float(feature_score)
                    scored_pose_candidates.append(
                        (
                            combined_score,
                            fixed_score,
                            fixed_inliers,
                            vfm_score,
                            rank_score,
                            candidate_result,
                            candidate_branch,
                            candidate_view_id,
                        )
                    )
                    feature_pose_evidence.append(
                        {
                            "candidate_index": int(candidate_index),
                            "branch": str(candidate_branch),
                            "support_view_id": candidate_view_id,
                            "feature_map_log_likelihood": (
                                float(feature_score)
                                if np.isfinite(feature_score)
                                else None
                            ),
                            "feature_supported_anchor_count": int(
                                feature_support
                            ),
                            "fixed_candidate_log_likelihood": (
                                float(fixed_score)
                                if np.isfinite(fixed_score)
                                else None
                            ),
                            "candidate_pose_w2c": (
                                candidate_result.pose_w2c.tolist()
                                if bool(args.emit_candidate_pose_trace)
                                and candidate_result.success
                                else None
                            ),
                            "combined_log_likelihood": (
                                float(combined_score)
                                if np.isfinite(combined_score)
                                else None
                            ),
                        }
                    )
                selected_scored = max(
                    scored_pose_candidates,
                    key=lambda item: (
                        float(item[0]),
                        float(item[1]),
                        int(item[2]),
                        float(item[3]),
                        int(item[4]),
                    ),
                )
                selected_candidate = (
                    selected_scored[1],
                    selected_scored[2],
                    selected_scored[3],
                    selected_scored[4],
                    selected_scored[5],
                    selected_scored[6],
                    selected_scored[7],
                )
                result = selected_candidate[4]
                branch = selected_candidate[5]
                selected_layout_view_id = selected_candidate[6]
                if selected_layout_view_id is not None:
                    selected_layout_score = float(selected_candidate[2])
            feature_mode_measurement_runtime = time.time() - stage_started
            stage_started = time.time()
            refinement_fixed_evidence_delta = None
            refinement_feature_evidence_delta = None
            featuremetric_seed_count = 0
            featuremetric_measurement_count = 0
            featuremetric_fixed_evidence_delta = None
            featuremetric_accepted = False
            if result.success and not bool(args.disable_pose_refinement):
                refinement_pool = (
                    build_pose_guided_surface_anchor_candidate_pool(
                        query,
                        result.pose_w2c,
                        camera,
                        local_bank,
                        anchors,
                        allowed_anchor_ids=retrieved_anchor_ids,
                        top_l=int(args.anchor_top_l),
                        search_radius_px=float(args.pose_refinement_radius_px),
                        maximum_query_points=int(
                            args.pose_refinement_maximum_points
                        ),
                    )
                )
                refinement_pool_size = len(refinement_pool)
                refinement_result = generate_grouped_surface_pose_hypotheses(
                    refinement_pool, camera, pose_config
                )
                if refinement_result.success:
                    base_fixed_score = score_surface_pose(
                        selection_pool,
                        result.pose_w2c,
                        camera,
                        np.ones((len(selection_pool),), dtype=bool),
                        pose_config,
                    )[0]
                    refinement_fixed_score = score_surface_pose(
                        selection_pool,
                        refinement_result.pose_w2c,
                        camera,
                        np.ones((len(selection_pool),), dtype=bool),
                        pose_config,
                    )[0]
                    refinement_fixed_evidence_delta = float(
                        refinement_fixed_score - base_fixed_score
                    )
                    base_refinement_feature_score, _base_support = (
                        _feature_map_pose_evidence(
                            result=result,
                            anchor_rows=feature_pose_anchor_rows,
                            anchors=anchors,
                            descriptor_bank=local_bank,
                            alike=alike,
                            image_path=image_path,
                            camera=camera,
                            support_view_id=selected_layout_view_id,
                            feature_kind=sparse_anchor_feature,
                            mapped_feature=mapped,
                        )
                    )
                    refined_feature_score, _refined_support = (
                        _feature_map_pose_evidence(
                            result=refinement_result,
                            anchor_rows=feature_pose_anchor_rows,
                            anchors=anchors,
                            descriptor_bank=local_bank,
                            alike=alike,
                            image_path=image_path,
                            camera=camera,
                            support_view_id=selected_layout_view_id,
                            feature_kind=sparse_anchor_feature,
                            mapped_feature=mapped,
                        )
                    )
                    if np.isfinite(base_refinement_feature_score) and np.isfinite(
                        refined_feature_score
                    ):
                        refinement_feature_evidence_delta = float(
                            refined_feature_score
                            - base_refinement_feature_score
                        )
                    if (
                        refinement_fixed_score > base_fixed_score
                        and (
                            not np.isfinite(base_refinement_feature_score)
                            or refined_feature_score
                            >= base_refinement_feature_score - 0.05
                        )
                    ):
                        result = refinement_result
                        branch = (
                            f"radio_maplet_{sparse_branch_label}_set_ot_"
                            "pose_guided_feature_em"
                        )
            if (
                result.success
                and not bool(args.disable_featuremetric_refinement)
            ):
                (
                    projected_anchor_xy,
                    projected_anchor_descriptors,
                    projected_anchor_ids,
                ) = _pose_projected_anchor_measurements(
                    pose_w2c=result.pose_w2c,
                    candidate_anchor_ids=retrieved_anchor_ids,
                    support_view_id=selected_layout_view_id,
                    anchors=anchors,
                    descriptor_bank=local_bank,
                    camera=camera,
                    maximum_anchors=int(
                        args.featuremetric_refinement_maximum_anchors
                    ),
                    projection_margin_px=float(
                        args.featuremetric_refinement_radius_px
                    ),
                )
                featuremetric_seed_count = len(projected_anchor_ids)
                if len(projected_anchor_ids) >= 4:
                    (
                        measured_xy,
                        measured_descriptors,
                        measured_similarities,
                        _measured_image_hash,
                    ) = _measure_anchor_descriptor_points(
                        feature_kind=sparse_anchor_feature,
                        alike=alike,
                        image_path=image_path,
                        mapped_feature=mapped,
                        predicted_xy=projected_anchor_xy,
                        support_descriptors=projected_anchor_descriptors,
                        alike_descriptor_dim=int(
                            detected.descriptors.shape[1]
                        ),
                        image_width=int(camera.width),
                        image_height=int(camera.height),
                        search_radius_px=int(
                            args.featuremetric_refinement_radius_px
                        ),
                        search_step_px=1,
                    )
                    (
                        measured_xy,
                        measured_descriptors,
                        measured_similarities,
                        measured_anchor_ids,
                    ) = _unique_featuremetric_measurements(
                        xy=measured_xy,
                        descriptors=measured_descriptors,
                        similarities=measured_similarities,
                        anchor_ids=projected_anchor_ids,
                        minimum_similarity=float(
                            args.featuremetric_refinement_minimum_similarity
                        ),
                    )
                    featuremetric_measurement_count = len(measured_anchor_ids)
                    if len(measured_anchor_ids) >= 4:
                        featuremetric_query = LocalFeatureFrame(
                            image_id=record.image_id,
                            keypoints_xy=measured_xy,
                            descriptors=measured_descriptors,
                            scores=np.maximum(
                                measured_similarities, 0.0
                            ),
                        )
                        featuremetric_pool = (
                            build_seeded_surface_anchor_candidate_pool(
                                featuremetric_query,
                                measured_anchor_ids,
                                descriptor_bank=local_bank,
                                anchors=anchors,
                                top_l=min(5, int(args.anchor_top_l)),
                                seed_prior_logit=float(
                                    args.layout_guided_seed_prior_logit
                                ),
                            )
                        )
                        featuremetric_result = (
                            generate_grouped_surface_pose_hypotheses(
                                featuremetric_pool, camera, pose_config
                            )
                        )
                        if featuremetric_result.success:
                            base_fixed_score = score_surface_pose(
                                selection_pool,
                                result.pose_w2c,
                                camera,
                                np.ones(
                                    (len(selection_pool),), dtype=bool
                                ),
                                pose_config,
                            )[0]
                            metric_fixed_score = score_surface_pose(
                                selection_pool,
                                featuremetric_result.pose_w2c,
                                camera,
                                np.ones(
                                    (len(selection_pool),), dtype=bool
                                ),
                                pose_config,
                            )[0]
                            featuremetric_fixed_evidence_delta = float(
                                metric_fixed_score - base_fixed_score
                            )
                            base_feature_score, _base_support = (
                                _feature_map_pose_evidence(
                                    result=result,
                                    anchor_rows=feature_pose_anchor_rows,
                                    anchors=anchors,
                                    descriptor_bank=local_bank,
                                    alike=alike,
                                    image_path=image_path,
                                    camera=camera,
                                    support_view_id=selected_layout_view_id,
                                    feature_kind=sparse_anchor_feature,
                                    mapped_feature=mapped,
                                )
                            )
                            metric_feature_score, _metric_support = (
                                _feature_map_pose_evidence(
                                    result=featuremetric_result,
                                    anchor_rows=feature_pose_anchor_rows,
                                    anchors=anchors,
                                    descriptor_bank=local_bank,
                                    alike=alike,
                                    image_path=image_path,
                                    camera=camera,
                                    support_view_id=selected_layout_view_id,
                                    feature_kind=sparse_anchor_feature,
                                    mapped_feature=mapped,
                                )
                            )
                            if (
                                metric_fixed_score > base_fixed_score
                                and (
                                    not np.isfinite(base_feature_score)
                                    or metric_feature_score
                                    >= base_feature_score - 0.05
                                )
                            ):
                                result = featuremetric_result
                                branch = (
                                    "radio_maplet_2dgs_feature_map_metric_"
                                    "refinement"
                                )
                                featuremetric_accepted = True
            refinement_runtime = time.time() - stage_started
            if result.success:
                successes += 1
            else:
                reason = str(result.failure_reason or "unknown")
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
            branch_counts[branch] = branch_counts.get(branch, 0) + 1
            diagnostics = {
                **matcher_diagnostics,
                "query_feature_count": len(query.keypoints_xy),
                "anchor_identity_feature": sparse_anchor_feature,
                "anchor_identity_uses_radio_final": bool(
                    sparse_anchor_feature
                    in (
                        ALIKE_RADIO_FINAL_ANCHOR_FEATURE,
                        RADIO_FINAL_ANCHOR_FEATURE,
                    )
                ),
                "anchor_identity_uses_alike_descriptor": bool(
                    sparse_anchor_feature
                    != RADIO_FINAL_ANCHOR_FEATURE
                ),
                "retrieved_maplet_count": len(
                    set(
                        int(value)
                        for value in maplet_match.candidate_maplet_ids.reshape(
                            -1
                        )
                        .tolist()
                        if int(value) >= 0
                    )
                ),
                "retrieved_anchor_count": len(retrieved_anchor_ids),
                "maplet_layout_feature_view_id": None,
                "maplet_layout_score": None,
                "initial_pool_size": len(anchor_pool),
                "set_pose_gate_passed": bool(set_pose_gate_passed),
                "set_pose_gate_reason": str(set_pose_gate_reason),
                "set_pose_confident_group_count": int(
                    confident_set_groups
                ),
                "set_pose_absolute_mean_null_probability": float(
                    mean_set_null
                ),
                "set_pose_gate_uses_conditional_identity_confidence": True,
                "set_pose_gate_uses_absolute_null_mass": True,
                "set_pose_strictly_confident_group_count": int(
                    matcher_diagnostics[
                        "conditionally_confident_group_count"
                    ]
                ),
                "fixed_verification_pool_size": len(
                    fixed_verification_pool
                ),
                "initial_fixed_evidence_per_group": (
                    float(initial_fixed_score)
                    if np.isfinite(initial_fixed_score)
                    else None
                ),
                "initial_fixed_evidence_inliers": int(initial_fixed_inliers),
                "layout_pool_size": int(layout_pool_size),
                "layout_mode_count": len(feature_mode_ids),
                "feature_mode_spatial_rerank": bool(
                    args.feature_mode_spatial_rerank
                ),
                "layout_success_count": int(layout_success_count),
                "direct_surface_fallback_group_count": int(
                    direct_surface_fallback_group_count
                ),
                "direct_surface_fallback_inliers": int(
                    direct_surface_fallback_inliers
                ),
                "direct_surface_candidate_always_enabled": bool(
                    args.always_direct_surface_candidate
                ),
                "feature_mode_diagnostics": feature_mode_diagnostics,
                "feature_pose_evidence_anchor_count": int(
                    len(feature_pose_anchor_rows)
                ),
                "feature_pose_evidence": feature_pose_evidence,
                "uses_per_query_calibration": bool(
                    str(args.query_camera_manifest)
                ),
                "selected_layout_feature_view_id": selected_layout_view_id,
                "selected_layout_feature_score": selected_layout_score,
                "refinement_pool_size": int(refinement_pool_size),
                "refinement_fixed_evidence_delta": (
                    refinement_fixed_evidence_delta
                ),
                "refinement_feature_evidence_delta": (
                    refinement_feature_evidence_delta
                ),
                "featuremetric_refinement_seed_count": int(
                    featuremetric_seed_count
                ),
                "featuremetric_refinement_measurement_count": int(
                    featuremetric_measurement_count
                ),
                "featuremetric_refinement_fixed_evidence_delta": (
                    featuremetric_fixed_evidence_delta
                ),
                "featuremetric_refinement_accepted": bool(
                    featuremetric_accepted
                ),
                "runtime_seconds": float(time.time() - query_started),
                "stage_runtime_seconds": {
                    "coarse_radio_maplet": float(coarse_runtime),
                    "query_alike": float(query_feature_runtime),
                    "anchor_set_matcher": float(anchor_set_runtime),
                    "initial_grouped_pnp": float(initial_pose_runtime),
                    "fixed_verification_pool": float(fixed_pool_runtime),
                    "feature_mode_selection": float(
                        feature_mode_selection_runtime
                    ),
                    "feature_mode_measurement_and_pnp": float(
                        feature_mode_measurement_runtime
                    ),
                    "pose_refinement": float(refinement_runtime),
                },
                "uses_mapping_rgb_at_inference": False,
                "uses_mapping_image_retrieval": False,
                "uses_pairwise_image_matching": False,
                "uses_sfm_points": False,
                "uses_sfm_tracks": False,
                "uses_radio_intermediate": False,
                "uses_view_depth_at_inference": False,
            }
            output.write(
                json.dumps(
                    _pose_record(
                        image_id=record.image_id,
                        result=result,
                        branch=branch,
                        diagnostics=diagnostics,
                    ),
                    sort_keys=True,
                )
                + "\n"
            )
            output.flush()
            query_file_audit.append(
                {
                    "image_id": record.image_id,
                    "path_relative_to_query_root": str(
                        image_path.relative_to(query_root)
                    ),
                    "sha256": detected.image_sha256,
                }
            )
            print(
                json.dumps(
                    {
                        "query": query_index + 1,
                        "query_count": len(records),
                        "image_id": record.image_id,
                        "success": result.success,
                        "branch": branch,
                        "runtime_seconds": diagnostics["runtime_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    artifacts = {
        name: {
            "path": str(path),
            "sha256": file_sha256_short(Path(path)),
        }
        for name, path in (
            ("query_manifest", args.query_manifest),
            ("surface_mapper_checkpoint", args.surface_mapper_checkpoint),
            ("maplets", args.maplets),
            ("anchors", args.anchors),
            ("local_descriptor_bank", args.local_descriptor_bank),
            ("radio_surface_feature_bank", args.radio_surface_feature_bank),
            (
                "surface_anchor_matcher_checkpoint",
                args.surface_anchor_matcher_checkpoint,
            ),
        )
    }
    if str(args.query_camera_manifest):
        artifacts["query_camera_manifest"] = {
            "path": str(args.query_camera_manifest),
            "sha256": file_sha256_short(Path(args.query_camera_manifest)),
        }
    summary = {
        "stage": "localize_2dgs_surface_queries_map_only",
        "query_count": len(records),
        "success_count": int(successes),
        "success_rate": float(successes / len(records)),
        "failure_counts": failure_counts,
        "branch_counts": branch_counts,
        "runtime_seconds": float(time.time() - started),
        "artifacts": artifacts,
        "config": {
            "radio_final_layer": str(args.radio_final_layer),
            "region_grid": [
                int(args.query_region_rows),
                int(args.query_region_cols),
                int(args.query_regions_per_cell),
            ],
            "maplet_match": maplet_config.__dict__,
            "anchor_set_matcher": anchor_matcher.config.to_dict(),
            "local_top_k": int(args.local_top_k),
            "augment_alike_with_radio_final": bool(
                args.augment_alike_with_radio_final
            ),
            "anchor_identity_feature": sparse_anchor_feature,
            "anchor_top_l": int(args.anchor_top_l),
            "matcher_feature_preferred_base_fill_target": int(
                args.matcher_feature_preferred_base_fill_target
            ),
            "set_pose_gate": {
                "minimum_confident_groups": int(
                    args.set_pose_minimum_confident_groups
                ),
                "confidence_semantics": (
                    "conditional_identity_given_matchable_with_absolute_"
                    "posterior_retained_for_pose_scoring"
                ),
                "legacy_maximum_mean_null_argument": float(
                    args.set_pose_maximum_mean_null
                ),
                "legacy_maximum_mean_null_applied": True,
            },
            "layout_guided_feature_measurement": {
                "maximum_global_feature_modes": int(
                    args.maximum_global_feature_modes
                ),
                "maximum_feature_modes": int(args.maximum_feature_modes),
                "spatial_layout_rerank": bool(
                    args.feature_mode_spatial_rerank
                ),
                "minimum_vfm_similarity": float(
                    args.feature_mode_minimum_similarity
                ),
                "maximum_points": int(args.layout_guided_maximum_points),
                "search_radius_px": int(args.layout_guided_search_radius_px),
                "seed_prior_logit": float(
                    args.layout_guided_seed_prior_logit
                ),
                "support_source": (
                    "stored_radio_final_2dgs_feature_modes_and_"
                    "stable_anchor_prototypes"
                ),
                "query_measurement": (
                    sparse_anchor_feature
                ),
            },
            "direct_surface_observation_candidate": {
                "always_enabled": bool(
                    args.always_direct_surface_candidate
                ),
                "query_feature": "radio_final",
                "map_feature": "2dgs_surface_observation_field",
            },
            "pose": pose_config.__dict__,
            "pose_refinement": {
                "enabled": not bool(args.disable_pose_refinement),
                "search_radius_px": float(args.pose_refinement_radius_px),
                "maximum_points": int(args.pose_refinement_maximum_points),
                "scope": "radio_retrieved_maplet_anchors_only",
            },
            "featuremetric_refinement": {
                "enabled": not bool(args.disable_featuremetric_refinement),
                "search_radius_px": int(
                    args.featuremetric_refinement_radius_px
                ),
                "maximum_anchors": int(
                    args.featuremetric_refinement_maximum_anchors
                ),
                "minimum_similarity": float(
                    args.featuremetric_refinement_minimum_similarity
                ),
                "acceptance": (
                    "fixed_candidate_evidence_improves_and_"
                    "dense_feature_likelihood_non_degrades"
                ),
            },
            "independent_feature_pose_evidence": {
                "maximum_anchors": int(
                    args.feature_pose_evidence_maximum_anchors
                ),
                "weight": float(args.feature_pose_evidence_weight),
                "source": (
                    f"query_{sparse_anchor_feature}_vs_stored_anchor_"
                    "prototypes"
                ),
            },
        },
        "runtime_access_audit": {
            "query_image_root": str(query_root),
            "opened_query_image_count": len(query_file_audit),
            "opened_query_images": query_file_audit,
            "mapping_image_paths_accepted_by_cli": False,
            "intrinsic_calibration": intrinsic_audit,
        },
        "production_contract": {
            "map_representation": (
                "2dgs_surface_maplets_stable_anchors_and_feature_prototypes"
            ),
            "coarse_match": "radio_final_region_to_maplet",
            "fine_match": (
                f"{sparse_anchor_feature}_query_set_to_anchor_set_"
                "cross_attention_ot_dustbin"
            ),
            "alike_role": (
                "detector_only"
                if sparse_anchor_feature == RADIO_FINAL_ANCHOR_FEATURE
                else "detector_and_identity_descriptor"
            ),
            "geometry": "grouped_pnp_with_heldout_feature_evidence",
            "support_mode_semantics": (
                "map_feature_mode_label_only_no_mapping_image_access"
            ),
            "uses_mapping_rgb_at_inference": False,
            "uses_mapping_image_retrieval": False,
            "uses_pairwise_image_matching": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_radio_intermediate": False,
            "uses_view_depth_at_inference": False,
        },
        "output_jsonl": str(output_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
