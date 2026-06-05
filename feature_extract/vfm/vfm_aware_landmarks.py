"""VFM-aware sparse landmark discovery diagnostics.

This module is intentionally independent from the canonical patch-to-3D
localization pipeline. Stage F uses it to measure whether SfM landmarks cover
VFM-distinctive image patches and to build smoke-test VFM patch tracks between
posed reference images.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from feature_extract.vfm.map_lifting import SelectedTrackFeatureBank, TrackFeature


def _normalize_rows(matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("matrix must have shape (N, C)")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(norms, eps)).astype(np.float32, copy=False)


def load_token_feature_map(path: Path, layer_name: str) -> np.ndarray:
    with np.load(Path(path)) as data:
        if layer_name not in data:
            raise ValueError(f"layer {layer_name!r} not found in {path}")
        feature_map = np.asarray(data[layer_name], dtype=np.float32)
    if feature_map.ndim != 3:
        raise ValueError("token feature map must have shape (C, H, W)")
    return feature_map


def flatten_feature_map(feature_map: np.ndarray) -> np.ndarray:
    values = np.asarray(feature_map, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    channels, height, width = values.shape
    return values.reshape(channels, height * width).T.astype(np.float32, copy=False)


def token_distinctiveness_scores(features: np.ndarray, block_size: int = 1024) -> np.ndarray:
    """Score each token by one-minus nearest-neighbor cosine ambiguity.

    A token whose descriptor is duplicated elsewhere receives a score near 0.
    A token far from every other token receives a larger score. This is a
    descriptor-only VFM saliency proxy, not a detector trained for keypoints.
    """

    descriptors = _normalize_rows(np.asarray(features, dtype=np.float32))
    count = int(descriptors.shape[0])
    if count == 0:
        return np.zeros((0,), dtype=np.float32)
    if count == 1:
        return np.ones((1,), dtype=np.float32)
    max_similarity = np.full((count,), -np.inf, dtype=np.float32)
    reference = descriptors.T.copy()
    for start in range(0, count, int(block_size)):
        end = min(start + int(block_size), count)
        scores = descriptors[start:end] @ reference
        rows = np.arange(end - start)
        scores[rows, np.arange(start, end)] = -np.inf
        max_similarity[start:end] = np.max(scores, axis=1)
    ambiguity = np.clip(max_similarity, -1.0, 1.0)
    return (1.0 - ambiguity).astype(np.float32, copy=False)


def select_distinctive_tokens(
    saliency_scores: np.ndarray,
    token_xy: np.ndarray,
    image_width: int,
    image_height: int,
    top_count: int,
    min_distance_to_boundary_px: float = 0.0,
    min_saliency: float | None = None,
) -> np.ndarray:
    """Return token indices sorted by descending VFM distinctiveness."""

    scores = np.asarray(saliency_scores, dtype=np.float32).reshape(-1)
    xy = np.asarray(token_xy, dtype=np.float64)
    if xy.shape != (scores.size, 2):
        raise ValueError("token_xy must have shape (N, 2)")
    if int(top_count) <= 0:
        return np.zeros((0,), dtype=np.int64)
    valid = np.isfinite(scores)
    if min_saliency is not None:
        valid &= scores >= float(min_saliency)
    margin = float(min_distance_to_boundary_px)
    if margin > 0.0:
        valid &= xy[:, 0] >= margin
        valid &= xy[:, 1] >= margin
        valid &= xy[:, 0] <= float(image_width - 1) - margin
        valid &= xy[:, 1] <= float(image_height - 1) - margin
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(-scores[valid_indices], kind="mergesort")
    return valid_indices[order[: int(top_count)]].astype(np.int64, copy=False)


def anchor_coverage_summary(
    saliency_scores: np.ndarray,
    has_anchor: np.ndarray,
    top_fractions: Sequence[float] = (0.1, 0.2),
) -> dict[str, float | int]:
    """Summarize whether VFM-distinctive tokens are covered by 3D anchors."""

    scores = np.asarray(saliency_scores, dtype=np.float32).reshape(-1)
    anchor_mask = np.asarray(has_anchor, dtype=bool).reshape(-1)
    if scores.shape[0] != anchor_mask.shape[0]:
        raise ValueError("saliency_scores and has_anchor must have the same length")
    token_count = int(scores.size)
    if token_count == 0:
        return {
            "token_count": 0,
            "anchor_token_fraction": 0.0,
            "mean_saliency_with_anchor": 0.0,
            "mean_saliency_without_anchor": 0.0,
        }
    summary: dict[str, float | int] = {
        "token_count": token_count,
        "anchor_token_count": int(np.sum(anchor_mask)),
        "anchor_token_fraction": float(np.mean(anchor_mask)),
        "mean_saliency": float(np.mean(scores)),
        "mean_saliency_with_anchor": 0.0 if not np.any(anchor_mask) else float(np.mean(scores[anchor_mask])),
        "mean_saliency_without_anchor": 0.0
        if np.all(anchor_mask)
        else float(np.mean(scores[~anchor_mask])),
    }
    sorted_indices = np.argsort(-scores, kind="mergesort")
    for fraction in top_fractions:
        value = float(fraction)
        if not 0.0 < value <= 1.0:
            raise ValueError("top fractions must be in (0, 1]")
        count = max(1, int(np.ceil(value * token_count)))
        selected = sorted_indices[:count]
        prefix = f"top_{int(round(value * 100))}pct"
        selected_anchor = anchor_mask[selected]
        summary[f"{prefix}_count"] = int(count)
        summary[f"{prefix}_anchor_fraction"] = float(np.mean(selected_anchor))
        summary[f"{prefix}_missing_anchor_fraction"] = float(np.mean(~selected_anchor))
        summary[f"{prefix}_mean_saliency"] = float(np.mean(scores[selected]))
    return summary


def candidate_reference_image_order(
    rows: Iterable[dict[str, object]],
    top_n: int,
    max_queries: int = 0,
) -> list[str]:
    """Return unique reference images used by a ranked candidate pool.

    The returned order is query first-seen order, then retrieval rank/order
    within each query. This mirrors reference-visibility localization pools
    while producing a single deterministic image list for Stage F pair-track
    construction.
    """

    if int(top_n) <= 0:
        raise ValueError("top_n must be positive")
    by_query: dict[str, list[tuple[int, int, str]]] = {}
    query_order: list[str] = []
    row_order = 0
    for row in rows:
        if row.get("record_type") == "header":
            continue
        if row.get("record_type", "candidate") != "candidate":
            continue
        query_id = row.get("query_id")
        reference_image = row.get("reference_image")
        if query_id is None or reference_image is None:
            continue
        query_key = str(query_id)
        if query_key not in by_query:
            if int(max_queries) > 0 and len(query_order) >= int(max_queries):
                row_order += 1
                continue
            by_query[query_key] = []
            query_order.append(query_key)
        metadata = row.get("metadata")
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        fallback_rank = len(by_query[query_key]) + 1
        rank = int(metadata_dict.get("retrieval_rank", fallback_rank))
        by_query[query_key].append((rank, row_order, str(reference_image)))
        row_order += 1

    ordered_references: list[str] = []
    seen: set[str] = set()
    for query_id in query_order:
        per_query_seen: set[str] = set()
        per_query_count = 0
        for _rank, _row_order, reference_image in sorted(by_query[query_id]):
            if reference_image in per_query_seen:
                continue
            per_query_seen.add(reference_image)
            per_query_count += 1
            if reference_image not in seen:
                ordered_references.append(reference_image)
                seen.add(reference_image)
            if per_query_count >= int(top_n):
                break
    return ordered_references


def _row_bool(row: dict[str, object], key: str) -> bool:
    return bool(row.get(key, False))


def _row_float(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def aggregate_track_match_reliability(rows: Iterable[dict[str, object]]) -> dict[int, dict[str, float | int]]:
    """Aggregate query-match diagnostics into per-track reliability metrics."""

    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        if row.get("track_id") is None:
            continue
        grouped.setdefault(int(row["track_id"]), []).append(row)

    result: dict[int, dict[str, float | int]] = {}
    for track_id, track_rows in grouped.items():
        count = int(len(track_rows))
        if count == 0:
            continue

        def mean_bool(key: str) -> float:
            return float(sum(1 for item in track_rows if _row_bool(item, key)) / count)

        def mean_float(key: str) -> float:
            values = [_row_float(item, key) for item in track_rows]
            clean = [float(value) for value in values if value is not None and np.isfinite(value)]
            return 0.0 if not clean else float(np.mean(clean))

        def median_float(key: str) -> float:
            values = [_row_float(item, key) for item in track_rows]
            clean = [float(value) for value in values if value is not None and np.isfinite(value)]
            return 0.0 if not clean else float(np.median(clean))

        result[int(track_id)] = {
            "track_id": int(track_id),
            "match_count": count,
            "patch_precision": mean_bool("patch_correct"),
            "patch_positive_precision": mean_bool("patch_positive_label"),
            "stride_precision": mean_bool("stride_positive_label"),
            "strong_positive_precision": mean_bool("strong_positive_label"),
            "pnp_inlier_rate": mean_bool("pnp_inlier"),
            "hard_negative_rate": mean_bool("hard_negative_label"),
            "mean_similarity": mean_float("similarity"),
            "median_similarity": median_float("similarity"),
            "mean_gt_reproj_error_px": mean_float("gt_reproj_error_px"),
            "median_gt_reproj_error_px": median_float("gt_reproj_error_px"),
            "mean_baseline_reproj_residual_px": mean_float("baseline_reproj_residual_px"),
            "median_baseline_reproj_residual_px": median_float("baseline_reproj_residual_px"),
            "mean_observation_count": mean_float("observation_count"),
            "mean_landmark_reprojection_error": mean_float("landmark_reprojection_error"),
            "mean_landmark_ambiguity": mean_float("landmark_ambiguity"),
        }
    return result


def filter_selected_track_bank_by_ids(
    bank: SelectedTrackFeatureBank,
    keep_track_ids: Iterable[int],
) -> SelectedTrackFeatureBank:
    keep = {int(track_id) for track_id in keep_track_ids}
    tracks = {int(track_id): track for track_id, track in bank.tracks.items() if int(track_id) in keep}
    return SelectedTrackFeatureBank(tracks=tracks, feature_dim=int(bank.feature_dim))


@dataclass(frozen=True)
class ReciprocalTokenMatch:
    source_token_index: int
    target_token_index: int
    similarity: float
    source_margin: float
    target_margin: float


@dataclass(frozen=True)
class VfmPatchTrackObservation:
    image_id: str
    token_index: int
    xy: np.ndarray


@dataclass(frozen=True)
class VfmPatchTrack:
    observations: tuple[VfmPatchTrackObservation, ...]
    pair_count: int
    mean_similarity: float


def _top2_margin(scores: np.ndarray, best_index: int) -> float:
    if scores.size <= 1:
        return float("inf")
    masked = np.array(scores, dtype=np.float32, copy=True)
    masked[int(best_index)] = -np.inf
    second = float(np.max(masked))
    return float(scores[int(best_index)] - second)


def reciprocal_token_matches(
    source_descriptors: np.ndarray,
    target_descriptors: np.ndarray,
    source_token_indices: np.ndarray | None = None,
    target_token_indices: np.ndarray | None = None,
    min_similarity: float = 0.0,
) -> list[ReciprocalTokenMatch]:
    """Build mutual nearest VFM patch correspondences between two token sets."""

    source = _normalize_rows(np.asarray(source_descriptors, dtype=np.float32))
    target = _normalize_rows(np.asarray(target_descriptors, dtype=np.float32))
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target descriptors must have the same feature dimension")
    source_count = int(source.shape[0])
    target_count = int(target.shape[0])
    if source_count == 0 or target_count == 0:
        return []
    src_ids = (
        np.arange(source_count, dtype=np.int64)
        if source_token_indices is None
        else np.asarray(source_token_indices, dtype=np.int64).reshape(-1)
    )
    tgt_ids = (
        np.arange(target_count, dtype=np.int64)
        if target_token_indices is None
        else np.asarray(target_token_indices, dtype=np.int64).reshape(-1)
    )
    if src_ids.shape[0] != source_count or tgt_ids.shape[0] != target_count:
        raise ValueError("token index arrays must match descriptor counts")
    scores = source @ target.T
    source_best = np.argmax(scores, axis=1)
    target_best = np.argmax(scores, axis=0)
    matches: list[ReciprocalTokenMatch] = []
    for source_row, target_row in enumerate(source_best.tolist()):
        if int(target_best[int(target_row)]) != int(source_row):
            continue
        similarity = float(scores[source_row, int(target_row)])
        if similarity < float(min_similarity):
            continue
        matches.append(
            ReciprocalTokenMatch(
                source_token_index=int(src_ids[source_row]),
                target_token_index=int(tgt_ids[int(target_row)]),
                similarity=similarity,
                source_margin=_top2_margin(scores[source_row], int(target_row)),
                target_margin=_top2_margin(scores[:, int(target_row)], int(source_row)),
            )
        )
    matches.sort(key=lambda item: (int(item.source_token_index), int(item.target_token_index)))
    return matches


def observation_records_from_matches(
    matches: Iterable[ReciprocalTokenMatch],
    source_image_id: str,
    target_image_id: str,
    source_xy_by_token: np.ndarray,
    target_xy_by_token: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for match in matches:
        source_xy = np.asarray(source_xy_by_token[int(match.source_token_index)], dtype=np.float64)
        target_xy = np.asarray(target_xy_by_token[int(match.target_token_index)], dtype=np.float64)
        rows.append(
            {
                "source_image_id": str(source_image_id),
                "target_image_id": str(target_image_id),
                "source_token_index": int(match.source_token_index),
                "target_token_index": int(match.target_token_index),
                "source_xy": [float(source_xy[0]), float(source_xy[1])],
                "target_xy": [float(target_xy[0]), float(target_xy[1])],
                "similarity": float(match.similarity),
                "source_margin": float(match.source_margin),
                "target_margin": float(match.target_margin),
            }
        )
    return rows


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def fundamental_from_w2c_poses(
    source_pose_w2c: np.ndarray,
    target_pose_w2c: np.ndarray,
    source_intrinsic: np.ndarray,
    target_intrinsic: np.ndarray,
) -> np.ndarray:
    """Compute the pixel-coordinate fundamental matrix from two w2c poses."""

    source_pose = np.asarray(source_pose_w2c, dtype=np.float64).reshape(4, 4)
    target_pose = np.asarray(target_pose_w2c, dtype=np.float64).reshape(4, 4)
    source_k = np.asarray(source_intrinsic, dtype=np.float64).reshape(3, 3)
    target_k = np.asarray(target_intrinsic, dtype=np.float64).reshape(3, 3)
    source_rotation = source_pose[:3, :3]
    target_rotation = target_pose[:3, :3]
    source_translation = source_pose[:3, 3]
    target_translation = target_pose[:3, 3]
    relative_rotation = target_rotation @ source_rotation.T
    relative_translation = target_translation - relative_rotation @ source_translation
    essential = _skew(relative_translation) @ relative_rotation
    fundamental = np.linalg.inv(target_k).T @ essential @ np.linalg.inv(source_k)
    norm = float(np.linalg.norm(fundamental))
    if norm > 1e-12:
        fundamental = fundamental / norm
    return fundamental.astype(np.float64, copy=False)


def sampson_epipolar_errors_px(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    fundamental: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return sqrt Sampson errors for corresponding pixel coordinates."""

    source = np.asarray(source_xy, dtype=np.float64)
    target = np.asarray(target_xy, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("source_xy and target_xy must both have shape (N, 2)")
    matrix = np.asarray(fundamental, dtype=np.float64).reshape(3, 3)
    ones = np.ones((source.shape[0], 1), dtype=np.float64)
    x1 = np.concatenate([source, ones], axis=1)
    x2 = np.concatenate([target, ones], axis=1)
    fx1 = (matrix @ x1.T).T
    ftx2 = (matrix.T @ x2.T).T
    residual = np.sum(x2 * fx1, axis=1)
    denom = fx1[:, 0] * fx1[:, 0] + fx1[:, 1] * fx1[:, 1] + ftx2[:, 0] * ftx2[:, 0] + ftx2[:, 1] * ftx2[:, 1]
    return np.sqrt((residual * residual) / np.maximum(denom, float(eps))).astype(np.float64, copy=False)


def project_points_w2c(points_xyz: np.ndarray, pose_w2c: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3)")
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    matrix = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    camera_points = (pose[:3, :3] @ points.T + pose[:3, 3:4]).T
    pixels_h = (matrix @ camera_points.T).T
    z = np.maximum(np.abs(pixels_h[:, 2:3]), 1e-12) * np.sign(pixels_h[:, 2:3] + 1e-12)
    return (pixels_h[:, :2] / z).astype(np.float64, copy=False)


def triangulate_two_view_dlt(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    source_pose_w2c: np.ndarray,
    target_pose_w2c: np.ndarray,
    source_intrinsic: np.ndarray,
    target_intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate 2D correspondences with linear DLT.

    Distortion is intentionally not modeled here. Stage F uses this as a
    geometry smoke test before investing in a full reconstruction backend.
    """

    source = np.asarray(source_xy, dtype=np.float64)
    target = np.asarray(target_xy, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("source_xy and target_xy must both have shape (N, 2)")
    source_pose = np.asarray(source_pose_w2c, dtype=np.float64).reshape(4, 4)
    target_pose = np.asarray(target_pose_w2c, dtype=np.float64).reshape(4, 4)
    source_k = np.asarray(source_intrinsic, dtype=np.float64).reshape(3, 3)
    target_k = np.asarray(target_intrinsic, dtype=np.float64).reshape(3, 3)
    source_projection = source_k @ source_pose[:3, :]
    target_projection = target_k @ target_pose[:3, :]
    xyz = np.zeros((source.shape[0], 3), dtype=np.float64)
    for idx, (xy_a, xy_b) in enumerate(zip(source, target)):
        x_a, y_a = float(xy_a[0]), float(xy_a[1])
        x_b, y_b = float(xy_b[0]), float(xy_b[1])
        system = np.stack(
            [
                x_a * source_projection[2] - source_projection[0],
                y_a * source_projection[2] - source_projection[1],
                x_b * target_projection[2] - target_projection[0],
                y_b * target_projection[2] - target_projection[1],
            ],
            axis=0,
        )
        _u, _s, vh = np.linalg.svd(system)
        point_h = vh[-1]
        xyz[idx] = point_h[:3] / max(abs(float(point_h[3])), 1e-12) * (1.0 if point_h[3] >= 0.0 else -1.0)
    source_projected = project_points_w2c(xyz, source_pose, source_k)
    target_projected = project_points_w2c(xyz, target_pose, target_k)
    source_error = np.linalg.norm(source_projected - source, axis=1)
    target_error = np.linalg.norm(target_projected - target, axis=1)
    return xyz, source_error.astype(np.float64), target_error.astype(np.float64)


def triangulate_multiview_dlt(
    observations_xy: Sequence[np.ndarray],
    poses_w2c: Sequence[np.ndarray],
    intrinsics: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate one point from 2+ calibrated observations."""

    if len(observations_xy) != len(poses_w2c) or len(observations_xy) != len(intrinsics):
        raise ValueError("observations_xy, poses_w2c, and intrinsics must have the same length")
    if len(observations_xy) < 2:
        raise ValueError("at least two observations are required")
    rows = []
    for xy, pose_w2c, intrinsic in zip(observations_xy, poses_w2c, intrinsics):
        point = np.asarray(xy, dtype=np.float64).reshape(2)
        pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
        matrix = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
        projection = matrix @ pose[:3, :]
        rows.append(float(point[0]) * projection[2] - projection[0])
        rows.append(float(point[1]) * projection[2] - projection[1])
    system = np.stack(rows, axis=0)
    _u, _s, vh = np.linalg.svd(system)
    point_h = vh[-1]
    xyz = point_h[:3] / max(abs(float(point_h[3])), 1e-12) * (1.0 if point_h[3] >= 0.0 else -1.0)
    projected_errors = []
    for xy, pose_w2c, intrinsic in zip(observations_xy, poses_w2c, intrinsics):
        projected = project_points_w2c(xyz.reshape(1, 3), pose_w2c, intrinsic)[0]
        projected_errors.append(float(np.linalg.norm(projected - np.asarray(xy, dtype=np.float64).reshape(2))))
    return xyz.astype(np.float64, copy=False), np.asarray(projected_errors, dtype=np.float64)


def triangulation_angles_deg(points_xyz: np.ndarray, source_pose_w2c: np.ndarray, target_pose_w2c: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3)")
    source_pose = np.asarray(source_pose_w2c, dtype=np.float64).reshape(4, 4)
    target_pose = np.asarray(target_pose_w2c, dtype=np.float64).reshape(4, 4)
    source_center = -source_pose[:3, :3].T @ source_pose[:3, 3]
    target_center = -target_pose[:3, :3].T @ target_pose[:3, 3]
    ray_a = points - source_center.reshape(1, 3)
    ray_b = points - target_center.reshape(1, 3)
    ray_a = ray_a / np.maximum(np.linalg.norm(ray_a, axis=1, keepdims=True), 1e-12)
    ray_b = ray_b / np.maximum(np.linalg.norm(ray_b, axis=1, keepdims=True), 1e-12)
    cosines = np.clip(np.sum(ray_a * ray_b, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cosines)).astype(np.float64, copy=False)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(self, item: tuple[str, int]) -> tuple[str, int]:
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            next_item = self.parent[item]
            self.parent[item] = root
            item = next_item
        return root

    def union(self, a: tuple[str, int], b: tuple[str, int]) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def link_vfm_patch_pair_rows(
    rows: Sequence[dict[str, object]],
    min_observations: int = 3,
    max_observations: int | None = None,
) -> list[VfmPatchTrack]:
    """Link pairwise VFM patch correspondences into connected multi-view tracks."""

    if int(min_observations) <= 1:
        raise ValueError("min_observations must be greater than 1")
    if max_observations is not None and int(max_observations) < int(min_observations):
        raise ValueError("max_observations must be >= min_observations")
    union_find = _UnionFind()
    xy_by_key: dict[tuple[str, int], np.ndarray] = {}
    pair_count_by_root: dict[tuple[str, int], int] = {}
    similarity_by_root: dict[tuple[str, int], list[float]] = {}
    pair_keys: list[tuple[tuple[str, int], tuple[str, int], float]] = []
    for row in rows:
        source_key = (str(row["source_image_id"]), int(row["source_token_index"]))
        target_key = (str(row["target_image_id"]), int(row["target_token_index"]))
        xy_by_key.setdefault(source_key, np.asarray(row["source_xy"], dtype=np.float64).reshape(2))
        xy_by_key.setdefault(target_key, np.asarray(row["target_xy"], dtype=np.float64).reshape(2))
        union_find.union(source_key, target_key)
        pair_keys.append((source_key, target_key, float(row.get("similarity", 0.0))))
    for source_key, _target_key, similarity in pair_keys:
        root = union_find.find(source_key)
        pair_count_by_root[root] = pair_count_by_root.get(root, 0) + 1
        similarity_by_root.setdefault(root, []).append(float(similarity))
    grouped: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for key in xy_by_key:
        grouped.setdefault(union_find.find(key), []).append(key)
    tracks: list[VfmPatchTrack] = []
    for root, keys in grouped.items():
        unique_keys = sorted(set(keys), key=lambda item: (item[0], item[1]))
        if len(unique_keys) < int(min_observations):
            continue
        if max_observations is not None and len(unique_keys) > int(max_observations):
            continue
        observations = tuple(
            VfmPatchTrackObservation(
                image_id=str(image_id),
                token_index=int(token_index),
                xy=np.asarray(xy_by_key[(image_id, token_index)], dtype=np.float64).copy(),
            )
            for image_id, token_index in unique_keys
        )
        similarities = similarity_by_root.get(root, [])
        tracks.append(
            VfmPatchTrack(
                observations=observations,
                pair_count=int(pair_count_by_root.get(root, 0)),
                mean_similarity=0.0 if not similarities else float(np.mean(similarities)),
            )
        )
    tracks.sort(key=lambda item: (-len(item.observations), -item.pair_count, item.observations[0].image_id))
    return tracks


def combine_selected_track_banks(
    banks: Sequence[SelectedTrackFeatureBank],
    track_id_offsets: Sequence[int] | None = None,
) -> SelectedTrackFeatureBank:
    """Combine independent track banks without modifying the source banks."""

    if not banks:
        return SelectedTrackFeatureBank(tracks={}, feature_dim=0)
    feature_dim = int(banks[0].feature_dim)
    if any(int(bank.feature_dim) != feature_dim for bank in banks):
        raise ValueError("all banks must have the same feature_dim")
    offsets = [0] * len(banks) if track_id_offsets is None else [int(item) for item in track_id_offsets]
    if len(offsets) != len(banks):
        raise ValueError("track_id_offsets must have one value per bank")
    tracks: dict[int, TrackFeature] = {}
    for bank, offset in zip(banks, offsets):
        for original_track_id, track in bank.tracks.items():
            track_id = int(original_track_id) + int(offset)
            if track_id in tracks:
                raise ValueError(f"duplicate combined track id: {track_id}")
            tracks[track_id] = TrackFeature(
                track_id=track_id,
                mean_feature=np.asarray(track.mean_feature, dtype=np.float32).copy(),
                variance=np.asarray(track.variance, dtype=np.float32).copy(),
                observation_count=int(track.observation_count),
                mean_utility=float(track.mean_utility),
                observation_image_ids=tuple(str(item) for item in track.observation_image_ids),
            )
    return SelectedTrackFeatureBank(tracks=tracks, feature_dim=feature_dim)


def project_selected_track_bank(
    bank: SelectedTrackFeatureBank,
    projector,
    output_dim: int,
    device: str = "cpu",
    batch_size: int = 4096,
    active_group_mask: np.ndarray | None = None,
) -> SelectedTrackFeatureBank:
    """Project aggregated track mean features with a descriptor selector."""

    try:
        import torch
        from torch.nn import functional as F
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required to project selected track banks") from exc
    if int(output_dim) <= 0:
        raise ValueError("output_dim must be positive")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if not bank.tracks:
        return SelectedTrackFeatureBank(tracks={}, feature_dim=int(output_dim))
    track_ids = np.asarray(sorted(bank.tracks), dtype=np.int64)
    features = np.stack([bank.tracks[int(track_id)].mean_feature for track_id in track_ids], axis=0).astype(np.float32)
    torch_device = torch.device(device)
    model = projector.to(torch_device) if hasattr(projector, "to") else projector
    if hasattr(model, "eval"):
        model.eval()
    mask_tensor = None
    if active_group_mask is not None:
        mask_tensor = torch.as_tensor(np.asarray(active_group_mask, dtype=np.float32), device=torch_device)
    projected_chunks = []
    with torch.no_grad():
        for start in range(0, features.shape[0], int(batch_size)):
            tensor = torch.as_tensor(features[start : start + int(batch_size)], dtype=torch.float32, device=torch_device)
            try:
                if mask_tensor is None:
                    projected = model(tensor)
                else:
                    projected = model(tensor, active_group_mask=mask_tensor)
            except TypeError:
                projected = model(tensor)
            projected = F.normalize(projected, p=2, dim=1, eps=1e-6)
            projected_chunks.append(projected.detach().cpu().numpy().astype(np.float32, copy=False))
    projected_features = np.concatenate(projected_chunks, axis=0)
    if projected_features.shape != (track_ids.size, int(output_dim)):
        raise ValueError(
            f"projected features have shape {projected_features.shape}, expected {(track_ids.size, int(output_dim))}"
        )
    tracks: dict[int, TrackFeature] = {}
    for idx, track_id in enumerate(track_ids):
        original = bank.tracks[int(track_id)]
        tracks[int(track_id)] = TrackFeature(
            track_id=int(track_id),
            mean_feature=projected_features[idx],
            variance=np.zeros((int(output_dim),), dtype=np.float32),
            observation_count=int(original.observation_count),
            mean_utility=float(original.mean_utility),
            observation_image_ids=tuple(str(item) for item in original.observation_image_ids),
        )
    return SelectedTrackFeatureBank(tracks=tracks, feature_dim=int(output_dim))
