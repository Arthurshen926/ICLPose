"""Low-level 2D measurement refinement sidecar for fixed VFM matches."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
import json

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapTrackObservation
from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch


@dataclass(frozen=True)
class LowLevelOffsetSidecarConfig:
    mode: str = "gray_ncc"
    template_radius_px: int = 8
    search_radius_px: int = 8
    search_step_px: int = 1
    max_offset_px: float | None = None
    min_score: float = 0.05
    min_confidence: float = 0.02

    def __post_init__(self) -> None:
        if self.mode not in {"gray_ncc", "sobel_ncc"}:
            raise ValueError("mode must be 'gray_ncc' or 'sobel_ncc'")
        if int(self.template_radius_px) <= 0 or int(self.search_radius_px) < 0 or int(self.search_step_px) <= 0:
            raise ValueError("template_radius_px/search_radius_px/search_step_px are invalid")
        if self.max_offset_px is not None and float(self.max_offset_px) < 0.0:
            raise ValueError("max_offset_px must be non-negative")


@dataclass(frozen=True)
class LowLevelOffsetResult:
    offset_xy: np.ndarray
    score: float
    confidence: float
    applied: bool
    support_image_id: str | None = None
    support_xy: tuple[float, float] | None = None
    reason: str = ""


@dataclass(frozen=True)
class SuperPointSnapConfig:
    max_offset_px: float = 8.0
    min_score: float = 0.005
    selection_strategy: str = "highest_score"

    def __post_init__(self) -> None:
        if float(self.max_offset_px) < 0.0:
            raise ValueError("max_offset_px must be non-negative")
        if float(self.min_score) < 0.0:
            raise ValueError("min_score must be non-negative")
        if self.selection_strategy not in {"highest_score", "nearest"}:
            raise ValueError("selection_strategy must be 'highest_score' or 'nearest'")


@dataclass(frozen=True)
class SuperPointKeypointSet:
    xy: np.ndarray
    scores: np.ndarray
    descriptors: np.ndarray | None = None

    def __post_init__(self) -> None:
        xy = np.asarray(self.xy)
        scores = np.asarray(self.scores)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must be Nx2")
        if scores.ndim != 1 or scores.shape[0] != xy.shape[0]:
            raise ValueError("scores must be N")
        if self.descriptors is not None:
            descriptors = np.asarray(self.descriptors)
            if descriptors.ndim != 2 or descriptors.shape[0] != xy.shape[0]:
                raise ValueError("descriptors must be NxD")


@dataclass(frozen=True)
class SupportSuperPointKeypoint:
    track_id: int
    support_image_id: str
    xy: np.ndarray
    descriptor: np.ndarray
    score: float
    distance_to_observation_px: float
    observation_xy: np.ndarray | None = None


@dataclass(frozen=True)
class LandmarkConditionedKeypointSelectorConfig:
    candidate_radius_px: float = 8.0
    support_radius_px: float = 16.0
    min_query_score: float = 0.005
    min_support_score: float = 0.005
    descriptor_weight: float = 1.0
    query_score_weight: float = 0.25
    center_penalty_weight: float = 0.15
    support_distance_penalty_weight: float = 0.10
    score_threshold: float = 0.2
    support_selection_strategy: str = "nearest"

    def __post_init__(self) -> None:
        if float(self.candidate_radius_px) < 0.0 or float(self.support_radius_px) < 0.0:
            raise ValueError("candidate/support radius must be non-negative")
        if float(self.min_query_score) < 0.0 or float(self.min_support_score) < 0.0:
            raise ValueError("min scores must be non-negative")
        if self.support_selection_strategy not in {"nearest", "highest_score"}:
            raise ValueError("support_selection_strategy must be 'nearest' or 'highest_score'")


@dataclass(frozen=True)
class LowLevelSupportBank:
    by_track: Mapping[int, tuple[ColmapTrackObservation, ...]]

    @classmethod
    def from_observations(cls, observations: Sequence[ColmapTrackObservation]) -> "LowLevelSupportBank":
        grouped: dict[int, list[ColmapTrackObservation]] = defaultdict(list)
        for obs in observations:
            grouped[int(obs.track_id)].append(obs)
        return cls(
            by_track={
                track_id: tuple(sorted(items, key=lambda item: (str(item.image_id), int(item.point2d_idx))))
                for track_id, items in grouped.items()
            }
        )

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "LowLevelSupportBank":
        observations = []
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            observations.append(
                ColmapTrackObservation(
                    track_id=int(item["track_id"]),
                    image_id=str(item["image_id"]),
                    point2d_idx=int(item.get("point2d_idx", 0)),
                    xy=(float(item["xy"][0]), float(item["xy"][1])),
                    xyz=np.asarray(item.get("xyz", [0.0, 0.0, 0.0]), dtype=np.float64),
                    track_length=int(item.get("track_length", 1)),
                    reprojection_error=float(item.get("reprojection_error", 0.0)),
                    camera_id=item.get("camera_id"),
                    image_width=item.get("image_width"),
                    image_height=item.get("image_height"),
                    camera_center=None
                    if item.get("camera_center") is None
                    else np.asarray(item.get("camera_center"), dtype=np.float64),
                    viewing_ray=None
                    if item.get("viewing_ray") is None
                    else np.asarray(item.get("viewing_ray"), dtype=np.float64),
                )
            )
        return cls.from_observations(observations)


def select_support_observation(
    support_bank: LowLevelSupportBank,
    track_id: int,
    query_viewing_ray: np.ndarray | None = None,
) -> ColmapTrackObservation | None:
    observations = support_bank.by_track.get(int(track_id), ())
    if not observations:
        return None
    if query_viewing_ray is None:
        return observations[0]
    query_ray = np.asarray(query_viewing_ray, dtype=np.float64).reshape(3)
    query_ray = query_ray / max(float(np.linalg.norm(query_ray)), 1e-12)

    def score(obs: ColmapTrackObservation) -> float:
        if obs.viewing_ray is None:
            return -1.0
        ray = np.asarray(obs.viewing_ray, dtype=np.float64).reshape(3)
        ray = ray / max(float(np.linalg.norm(ray)), 1e-12)
        return float(np.dot(query_ray, ray))

    return max(observations, key=score)


def _to_gray(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim == 2:
        gray = values.astype(np.float32)
    elif values.ndim == 3 and values.shape[2] >= 3:
        rgb = values[..., :3].astype(np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    else:
        raise ValueError("image must be HxW or HxWx3")
    if gray.max(initial=0.0) > 2.0:
        gray = gray / 255.0
    return gray.astype(np.float32, copy=False)


def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for sobel_ncc") from exc
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def preprocess_lowlevel_image(image: np.ndarray, mode: str = "gray_ncc") -> np.ndarray:
    gray = _to_gray(image)
    if mode == "gray_ncc":
        return gray
    if mode == "sobel_ncc":
        return _sobel_magnitude(gray)
    raise ValueError("unsupported low-level mode")


def _extract_crop(image: np.ndarray, xy: np.ndarray, radius: int) -> np.ndarray | None:
    values = np.asarray(image, dtype=np.float32)
    x = int(round(float(xy[0])))
    y = int(round(float(xy[1])))
    r = int(radius)
    if x - r < 0 or y - r < 0 or x + r >= values.shape[1] or y + r >= values.shape[0]:
        return None
    return values[y - r : y + r + 1, x - r : x + r + 1].astype(np.float32, copy=True)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    aa = aa - float(np.mean(aa))
    bb = bb - float(np.mean(bb))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-8:
        return -1.0
    return float(np.dot(aa, bb) / denom)


def estimate_ncc_patch_offset(
    query_image: np.ndarray,
    support_image: np.ndarray,
    query_xy: np.ndarray,
    support_xy: np.ndarray,
    config: LowLevelOffsetSidecarConfig,
) -> LowLevelOffsetResult:
    query_values = preprocess_lowlevel_image(query_image, config.mode)
    support_values = preprocess_lowlevel_image(support_image, config.mode)
    return _estimate_ncc_patch_offset_preprocessed(query_values, support_values, query_xy, support_xy, config)


def _estimate_ncc_patch_offset_preprocessed(
    query_values: np.ndarray,
    support_values: np.ndarray,
    query_xy: np.ndarray,
    support_xy: np.ndarray,
    config: LowLevelOffsetSidecarConfig,
) -> LowLevelOffsetResult:
    radius = int(config.template_radius_px)
    template = _extract_crop(support_values, np.asarray(support_xy, dtype=np.float64), radius)
    if template is None:
        return LowLevelOffsetResult(np.zeros((2,), dtype=np.float32), -1.0, 0.0, False, reason="support_crop_oob")
    scores: list[tuple[float, int, int]] = []
    search = int(config.search_radius_px)
    step = int(config.search_step_px)
    max_offset = float(config.max_offset_px) if config.max_offset_px is not None else float(search)
    for dy in range(-search, search + 1, step):
        for dx in range(-search, search + 1, step):
            if max(abs(float(dx)), abs(float(dy))) > max_offset:
                continue
            crop = _extract_crop(query_values, np.asarray(query_xy, dtype=np.float64) + np.asarray([dx, dy]), radius)
            if crop is None:
                continue
            scores.append((_ncc(crop, template), int(dx), int(dy)))
    if not scores:
        return LowLevelOffsetResult(np.zeros((2,), dtype=np.float32), -1.0, 0.0, False, reason="query_search_oob")
    scores.sort(key=lambda item: item[0], reverse=True)
    best_score, best_dx, best_dy = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else -1.0
    confidence = float(best_score - second_score)
    applied = bool(best_score >= float(config.min_score) and confidence >= float(config.min_confidence))
    reason = "applied" if applied else "low_confidence"
    return LowLevelOffsetResult(
        offset_xy=np.asarray([best_dx, best_dy], dtype=np.float32),
        score=float(best_score),
        confidence=confidence,
        applied=applied,
        reason=reason,
    )


def snap_xy_to_superpoint_keypoint(
    query_xy: np.ndarray,
    keypoints: SuperPointKeypointSet,
    config: SuperPointSnapConfig,
) -> LowLevelOffsetResult:
    xy = np.asarray(query_xy, dtype=np.float64).reshape(2)
    keypoint_xy = np.asarray(keypoints.xy, dtype=np.float64).reshape(-1, 2)
    scores = np.asarray(keypoints.scores, dtype=np.float64).reshape(-1)
    if keypoint_xy.shape[0] == 0:
        return LowLevelOffsetResult(np.zeros((2,), dtype=np.float32), -1.0, 0.0, False, reason="no_keypoints")
    offsets = keypoint_xy - xy[None, :]
    distances = np.linalg.norm(offsets, axis=1)
    keep = (distances <= float(config.max_offset_px)) & (scores >= float(config.min_score))
    if not np.any(keep):
        return LowLevelOffsetResult(
            np.zeros((2,), dtype=np.float32),
            -1.0,
            0.0,
            False,
            reason="no_keypoint_within_radius",
        )
    kept_indices = np.flatnonzero(keep)
    if config.selection_strategy == "nearest":
        order = sorted(kept_indices.tolist(), key=lambda idx: (float(distances[idx]), -float(scores[idx])))
    else:
        order = sorted(kept_indices.tolist(), key=lambda idx: (-float(scores[idx]), float(distances[idx])))
    best_idx = int(order[0])
    second_score = float(scores[int(order[1])]) if len(order) > 1 else 0.0
    confidence = float(scores[best_idx] - second_score)
    return LowLevelOffsetResult(
        offset_xy=offsets[best_idx].astype(np.float32),
        score=float(scores[best_idx]),
        confidence=confidence,
        applied=True,
        reason="applied",
    )


def _nearest_superpoint_to_xy(
    xy: np.ndarray,
    keypoints: SuperPointKeypointSet,
) -> tuple[int | None, float | None]:
    keypoint_xy = np.asarray(keypoints.xy, dtype=np.float64).reshape(-1, 2)
    if keypoint_xy.shape[0] == 0:
        return None, None
    query_xy = np.asarray(xy, dtype=np.float64).reshape(2)
    distances = np.linalg.norm(keypoint_xy - query_xy[None, :], axis=1)
    idx = int(np.argmin(distances))
    return idx, float(distances[idx])


def _normalized_descriptor(descriptor: np.ndarray) -> np.ndarray:
    values = np.asarray(descriptor, dtype=np.float32).reshape(-1)
    norm = max(float(np.linalg.norm(values)), 1e-12)
    return (values / norm).astype(np.float32, copy=False)


def _require_descriptors(keypoints: SuperPointKeypointSet) -> np.ndarray:
    if keypoints.descriptors is None:
        raise ValueError("SuperPoint descriptors are required for landmark-conditioned selection")
    descriptors = np.asarray(keypoints.descriptors, dtype=np.float32)
    if descriptors.shape[0] != np.asarray(keypoints.xy).shape[0]:
        raise ValueError("descriptor count does not match keypoint count")
    return descriptors


def select_support_superpoint_keypoint(
    track_id: int,
    support_image_id: str,
    observation_xy: np.ndarray,
    keypoints: SuperPointKeypointSet,
    max_distance_px: float,
    min_score: float = 0.005,
    strategy: str = "nearest",
) -> SupportSuperPointKeypoint | None:
    descriptors = _require_descriptors(keypoints)
    keypoint_xy = np.asarray(keypoints.xy, dtype=np.float64).reshape(-1, 2)
    scores = np.asarray(keypoints.scores, dtype=np.float64).reshape(-1)
    if keypoint_xy.shape[0] == 0:
        return None
    xy = np.asarray(observation_xy, dtype=np.float64).reshape(2)
    distances = np.linalg.norm(keypoint_xy - xy[None, :], axis=1)
    keep = (distances <= float(max_distance_px)) & (scores >= float(min_score))
    if not np.any(keep):
        return None
    indices = np.flatnonzero(keep).tolist()
    if strategy == "highest_score":
        order = sorted(indices, key=lambda idx: (-float(scores[idx]), float(distances[idx])))
    elif strategy == "nearest":
        order = sorted(indices, key=lambda idx: (float(distances[idx]), -float(scores[idx])))
    else:
        raise ValueError("strategy must be 'nearest' or 'highest_score'")
    best = int(order[0])
    return SupportSuperPointKeypoint(
        track_id=int(track_id),
        support_image_id=str(support_image_id),
        xy=keypoint_xy[best].astype(np.float64),
        descriptor=_normalized_descriptor(descriptors[best]),
        score=float(scores[best]),
        distance_to_observation_px=float(distances[best]),
        observation_xy=xy.astype(np.float64),
    )


def _candidate_descriptor_details(
    query_xy: np.ndarray,
    query_keypoints: SuperPointKeypointSet,
    support: SupportSuperPointKeypoint,
    config: LandmarkConditionedKeypointSelectorConfig,
) -> dict[str, np.ndarray]:
    descriptors = _require_descriptors(query_keypoints)
    keypoint_xy = np.asarray(query_keypoints.xy, dtype=np.float64).reshape(-1, 2)
    scores = np.asarray(query_keypoints.scores, dtype=np.float64).reshape(-1)
    xy = np.asarray(query_xy, dtype=np.float64).reshape(2)
    distances = np.linalg.norm(keypoint_xy - xy[None, :], axis=1)
    keep = (distances <= float(config.candidate_radius_px)) & (scores >= float(config.min_query_score))
    indices = np.flatnonzero(keep).astype(np.int64)
    if indices.size == 0:
        return {
            "indices": indices,
            "selector_scores": np.zeros((0,), dtype=np.float64),
            "descriptor_similarities": np.zeros((0,), dtype=np.float64),
            "distances": distances,
            "scores": scores,
        }
    query_desc = descriptors[indices].astype(np.float32, copy=False)
    query_desc = query_desc / np.maximum(np.linalg.norm(query_desc, axis=1, keepdims=True), 1e-12)
    support_desc = _normalized_descriptor(support.descriptor)
    similarities = query_desc @ support_desc
    norm_center_dist = distances[indices] / max(float(config.candidate_radius_px), 1e-6)
    norm_support_dist = float(support.distance_to_observation_px) / max(float(config.support_radius_px), 1e-6)
    selector_scores = (
        float(config.descriptor_weight) * similarities.astype(np.float64)
        + float(config.query_score_weight) * scores[indices]
        - float(config.center_penalty_weight) * norm_center_dist
        - float(config.support_distance_penalty_weight) * norm_support_dist
    )
    return {
        "indices": indices,
        "selector_scores": selector_scores.astype(np.float64),
        "descriptor_similarities": similarities.astype(np.float64),
        "distances": distances,
        "scores": scores,
    }


def _candidate_descriptor_scores(
    query_xy: np.ndarray,
    query_keypoints: SuperPointKeypointSet,
    support: SupportSuperPointKeypoint,
    config: LandmarkConditionedKeypointSelectorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    details = _candidate_descriptor_details(query_xy, query_keypoints, support, config)
    return (
        details["indices"].astype(np.int64),
        details["selector_scores"].astype(np.float64),
        details["distances"].astype(np.float64),
        details["scores"].astype(np.float64),
    )


def select_landmark_conditioned_keypoint(
    query_xy: np.ndarray,
    query_keypoints: SuperPointKeypointSet,
    support: SupportSuperPointKeypoint,
    config: LandmarkConditionedKeypointSelectorConfig,
) -> LowLevelOffsetResult:
    indices, selector_scores, distances, scores = _candidate_descriptor_scores(
        query_xy,
        query_keypoints,
        support,
        config,
    )
    if indices.size == 0:
        return LowLevelOffsetResult(np.zeros((2,), dtype=np.float32), -1.0, 0.0, False, reason="no_query_candidates")
    order = indices[np.argsort(-selector_scores)]
    best = int(order[0])
    best_score = float(selector_scores[np.argsort(-selector_scores)[0]])
    if best_score < float(config.score_threshold):
        return LowLevelOffsetResult(
            np.zeros((2,), dtype=np.float32),
            best_score,
            0.0,
            False,
            reason="selector_below_threshold",
        )
    sorted_scores = np.sort(selector_scores)[::-1]
    confidence = float(sorted_scores[0] - sorted_scores[1]) if sorted_scores.size > 1 else float(sorted_scores[0])
    keypoint_xy = np.asarray(query_keypoints.xy, dtype=np.float64).reshape(-1, 2)
    offset = keypoint_xy[best] - np.asarray(query_xy, dtype=np.float64).reshape(2)
    return LowLevelOffsetResult(
        offset_xy=offset.astype(np.float32),
        score=best_score,
        confidence=confidence,
        applied=True,
        reason="applied",
    )


def _rank_of_gt_nearest_candidate(
    gt_xy: np.ndarray,
    query_xy: np.ndarray,
    query_keypoints: SuperPointKeypointSet,
    support: SupportSuperPointKeypoint,
    config: LandmarkConditionedKeypointSelectorConfig,
    positive_radius_px: float = 8.0,
) -> int | None:
    indices, selector_scores, _distances, _scores = _candidate_descriptor_scores(query_xy, query_keypoints, support, config)
    if indices.size == 0:
        return None
    keypoint_xy = np.asarray(query_keypoints.xy, dtype=np.float64).reshape(-1, 2)
    gt = np.asarray(gt_xy, dtype=np.float64).reshape(2)
    candidate_distances = np.linalg.norm(keypoint_xy[indices] - gt[None, :], axis=1)
    positive_positions = np.flatnonzero(candidate_distances <= float(positive_radius_px))
    if positive_positions.size == 0:
        return None
    best_positive_position = int(positive_positions[np.argmin(candidate_distances[positive_positions])])
    sorted_positions = np.argsort(-selector_scores)
    rank_positions = {int(position): rank for rank, position in enumerate(sorted_positions.tolist(), start=1)}
    return int(rank_positions[best_positive_position])


def build_landmark_conditioned_superpoint_candidate_rows(
    matches: Sequence[QueryTo3DMatch],
    query_keypoints: SuperPointKeypointSet,
    support_by_track: Mapping[int, SupportSuperPointKeypoint],
    inlier_mask: np.ndarray,
    config: LandmarkConditionedKeypointSelectorConfig,
    gt_xy_by_match: Sequence[np.ndarray | None] | None = None,
    query_id: str | None = None,
    stride_px: float | None = None,
    baseline_reproj_residual_by_match: Sequence[float | None] | None = None,
) -> list[dict[str, object]]:
    """Build per-action rows for landmark-conditioned keypoint diagnostics.

    Each first-pass inlier receives one explicit ``no_snap`` action. If a
    support SuperPoint keypoint and query candidates exist, the function also
    emits one ``snap`` action per candidate keypoint. GT-derived fields are
    diagnostics/training labels only; all feature fields are available at
    inference time after first-pass PnP.
    """

    mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != len(matches):
        raise ValueError("inlier_mask must have one value per match")
    if gt_xy_by_match is not None and len(gt_xy_by_match) != len(matches):
        raise ValueError("gt_xy_by_match must have one value per match")
    if baseline_reproj_residual_by_match is not None and len(baseline_reproj_residual_by_match) != len(matches):
        raise ValueError("baseline_reproj_residual_by_match must have one value per match")

    keypoint_xy = np.asarray(query_keypoints.xy, dtype=np.float64).reshape(-1, 2)
    keypoint_scores = np.asarray(query_keypoints.scores, dtype=np.float64).reshape(-1)
    rows: list[dict[str, object]] = []
    for match_idx, match in enumerate(matches):
        if not bool(mask[match_idx]):
            continue
        base_xy = np.asarray(match.xy, dtype=np.float64).reshape(2)
        gt_xy = None if gt_xy_by_match is None else gt_xy_by_match[match_idx]
        gt_xy_arr = None if gt_xy is None else np.asarray(gt_xy, dtype=np.float64).reshape(2)
        center_error = None if gt_xy_arr is None else float(np.linalg.norm(base_xy - gt_xy_arr))
        residual_value = (
            None
            if baseline_reproj_residual_by_match is None
            else baseline_reproj_residual_by_match[match_idx]
        )
        common = {
            "query_id": None if query_id is None else str(query_id),
            "match_index": int(match_idx),
            "token_index": int(match.token_index),
            "track_id": int(match.track_id),
            "base_xy": [float(base_xy[0]), float(base_xy[1])],
            "xyz": [float(value) for value in np.asarray(match.xyz, dtype=np.float64).reshape(3)],
            "gt_xy": None if gt_xy_arr is None else [float(gt_xy_arr[0]), float(gt_xy_arr[1])],
            "center_error_px": center_error,
            "stride_px": None if stride_px is None else float(stride_px),
            "match_similarity": float(match.similarity),
            "match_ratio": float(match.ratio),
            "similarity_margin": None if match.similarity_margin is None else float(match.similarity_margin),
            "landmark_variance": float(match.landmark_variance),
            "landmark_reprojection_error": None
            if match.landmark_reprojection_error is None
            else float(match.landmark_reprojection_error),
            "landmark_quality": None if match.landmark_quality is None else float(match.landmark_quality),
            "landmark_ambiguity": None if match.landmark_ambiguity is None else float(match.landmark_ambiguity),
            "observation_count": None if match.observation_count is None else int(match.observation_count),
            "visibility_count": None if match.visibility_count is None else int(match.visibility_count),
            "baseline_reproj_residual_px": None if residual_value is None else float(residual_value),
        }
        rows.append(
            {
                **common,
                "action": "no_snap",
                "candidate_index": -1,
                "candidate_rank": -1,
                "candidate_xy": [float(base_xy[0]), float(base_xy[1])],
                "residual_xy": [float(base_xy[0]), float(base_xy[1])],
                "candidate_error_px": center_error,
                "residual_error_px": center_error,
                "snap_improvement_px": 0.0,
                "support_available": False,
                "selected_by_heuristic": False,
                "heuristic_applied": False,
            }
        )
        support = support_by_track.get(int(match.track_id))
        if support is None:
            continue
        details = _candidate_descriptor_details(base_xy, query_keypoints, support, config)
        indices = details["indices"].astype(np.int64)
        selector_scores = details["selector_scores"].astype(np.float64)
        descriptor_similarities = details["descriptor_similarities"].astype(np.float64)
        distances = details["distances"].astype(np.float64)
        scores = details["scores"].astype(np.float64)
        if indices.size == 0:
            continue
        sorted_positions = np.argsort(-selector_scores)
        sorted_scores = selector_scores[sorted_positions]
        best_score = float(sorted_scores[0])
        heuristic_applied = bool(best_score >= float(config.score_threshold))
        support_xy = np.asarray(support.xy, dtype=np.float64).reshape(2)
        observation_xy = None if support.observation_xy is None else np.asarray(support.observation_xy, dtype=np.float64).reshape(2)
        support_delta = np.zeros((2,), dtype=np.float64) if observation_xy is None else observation_xy - support_xy
        norm_support_dist = float(support.distance_to_observation_px) / max(float(config.support_radius_px), 1e-6)
        rank_by_position = {int(position): rank for rank, position in enumerate(sorted_positions.tolist(), start=1)}
        best_position = int(sorted_positions[0])
        for position, kp_idx in enumerate(indices.tolist()):
            candidate_xy = keypoint_xy[int(kp_idx)].astype(np.float64)
            residual_xy = candidate_xy + support_delta
            candidate_error = None if gt_xy_arr is None else float(np.linalg.norm(candidate_xy - gt_xy_arr))
            residual_error = None if gt_xy_arr is None else float(np.linalg.norm(residual_xy - gt_xy_arr))
            improvement = (
                None
                if center_error is None or candidate_error is None
                else float(center_error - candidate_error)
            )
            residual_improvement = (
                None
                if center_error is None or residual_error is None
                else float(center_error - residual_error)
            )
            rows.append(
                {
                    **common,
                    "action": "snap",
                    "candidate_index": int(kp_idx),
                    "candidate_rank": int(rank_by_position[int(position)]),
                    "candidate_xy": [float(candidate_xy[0]), float(candidate_xy[1])],
                    "residual_xy": [float(residual_xy[0]), float(residual_xy[1])],
                    "candidate_error_px": candidate_error,
                    "residual_error_px": residual_error,
                    "snap_improvement_px": improvement,
                    "residual_improvement_px": residual_improvement,
                    "snap_improves_center": False
                    if improvement is None
                    else bool(improvement > 0.0),
                    "snap_correct_at_4px": False
                    if candidate_error is None
                    else bool(candidate_error <= 4.0),
                    "snap_correct_at_8px": False
                    if candidate_error is None
                    else bool(candidate_error <= 8.0),
                    "residual_correct_at_8px": False
                    if residual_error is None
                    else bool(residual_error <= 8.0),
                    "selected_by_heuristic": bool(position == best_position),
                    "heuristic_applied": bool(position == best_position and heuristic_applied),
                    "selector_score": float(selector_scores[position]),
                    "selector_margin": float(sorted_scores[0] - sorted_scores[1]) if sorted_scores.size > 1 else float(sorted_scores[0]),
                    "descriptor_similarity": float(descriptor_similarities[position]),
                    "query_keypoint_score": float(scores[int(kp_idx)]),
                    "support_keypoint_score": float(support.score),
                    "distance_to_center_px": float(distances[int(kp_idx)]),
                    "distance_to_center_norm": float(distances[int(kp_idx)] / max(float(config.candidate_radius_px), 1e-6)),
                    "support_available": True,
                    "support_image_id": str(support.support_image_id),
                    "support_xy": [float(support_xy[0]), float(support_xy[1])],
                    "support_observation_xy": None
                    if observation_xy is None
                    else [float(observation_xy[0]), float(observation_xy[1])],
                    "support_delta_xy": [float(support_delta[0]), float(support_delta[1])],
                    "support_bias_px": float(support.distance_to_observation_px),
                    "support_bias_norm": norm_support_dist,
                }
            )
    return rows


def apply_landmark_conditioned_superpoint_snaps_to_matches(
    matches: Sequence[QueryTo3DMatch],
    query_keypoints: SuperPointKeypointSet,
    support_by_track: Mapping[int, SupportSuperPointKeypoint],
    inlier_mask: np.ndarray,
    config: LandmarkConditionedKeypointSelectorConfig,
    gt_xy_by_match: Sequence[np.ndarray | None] | None = None,
) -> tuple[list[QueryTo3DMatch], dict[str, object]]:
    refined = list(matches)
    mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != len(matches):
        raise ValueError("inlier_mask must have one value per match")
    if gt_xy_by_match is not None and len(gt_xy_by_match) != len(matches):
        raise ValueError("gt_xy_by_match must have one value per match")
    applied_count = 0
    inlier_count = 0
    skipped_by_inlier = 0
    missing_support = 0
    low_confidence = 0
    offsets = []
    selector_scores = []
    confidences = []
    snap_errors = []
    top1_flags = []
    top5_flags = []
    for idx, match in enumerate(matches):
        if not bool(mask[idx]):
            skipped_by_inlier += 1
            continue
        inlier_count += 1
        support = support_by_track.get(int(match.track_id))
        if support is None:
            missing_support += 1
            continue
        result = select_landmark_conditioned_keypoint(np.asarray(match.xy, dtype=np.float64), query_keypoints, support, config)
        gt_xy = None if gt_xy_by_match is None else gt_xy_by_match[idx]
        rank = None
        if gt_xy is not None:
            rank = _rank_of_gt_nearest_candidate(
                np.asarray(gt_xy, dtype=np.float64),
                np.asarray(match.xy, dtype=np.float64),
                query_keypoints,
                support,
                config,
                positive_radius_px=8.0,
            )
            if rank is not None:
                top1_flags.append(1.0 if rank == 1 else 0.0)
                top5_flags.append(1.0 if rank <= 5 else 0.0)
        if not result.applied:
            low_confidence += 1
            continue
        new_xy = np.asarray(match.xy, dtype=np.float64).reshape(2) + result.offset_xy.astype(np.float64)
        refined[idx] = replace(match, xy=new_xy)
        applied_count += 1
        offsets.append(float(np.linalg.norm(result.offset_xy)))
        selector_scores.append(float(result.score))
        confidences.append(float(result.confidence))
        if gt_xy is not None:
            snap_errors.append(float(np.linalg.norm(new_xy - np.asarray(gt_xy, dtype=np.float64).reshape(2))))
    return refined, {
        "mode": "landmark_conditioned_superpoint",
        "refined_count": int(applied_count),
        "applied_count": int(applied_count),
        "inlier_count": int(inlier_count),
        "offset_applied_ratio": float(applied_count / max(inlier_count, 1)),
        "skipped_by_inlier_count": int(skipped_by_inlier),
        "missing_support_count": int(missing_support),
        "low_confidence_count": int(low_confidence),
        "mean_offset_px": None if not offsets else float(np.mean(offsets)),
        "mean_score": None if not selector_scores else float(np.mean(selector_scores)),
        "mean_confidence": None if not confidences else float(np.mean(confidences)),
        "mean_snap_error_to_gt_px": None if not snap_errors else float(np.mean(snap_errors)),
        "snap_selection_accuracy_at_4px": None if not snap_errors else float(np.mean([1.0 if value <= 4.0 else 0.0 for value in snap_errors])),
        "snap_selection_accuracy_at_8px": None if not snap_errors else float(np.mean([1.0 if value <= 8.0 else 0.0 for value in snap_errors])),
        "sp_descriptor_top1_accuracy": None if not top1_flags else float(np.mean(top1_flags)),
        "sp_descriptor_top5_accuracy": None if not top5_flags else float(np.mean(top5_flags)),
        "keypoint_count": int(np.asarray(query_keypoints.xy).shape[0]),
        "support_count": int(len(support_by_track)),
        "candidate_radius_px": float(config.candidate_radius_px),
        "support_radius_px": float(config.support_radius_px),
        "score_threshold": float(config.score_threshold),
    }


def superpoint_availability_summary(
    gt_xy_by_match: Sequence[np.ndarray | None],
    keypoints: SuperPointKeypointSet,
    inlier_mask: np.ndarray,
    radii_px: Sequence[float] = (2.0, 4.0, 8.0, 12.0, 16.0),
) -> dict[str, object]:
    mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != len(gt_xy_by_match):
        raise ValueError("inlier_mask must have one value per GT projection")
    distances = []
    missing_gt = 0
    for idx, gt_xy in enumerate(gt_xy_by_match):
        if not bool(mask[idx]):
            continue
        if gt_xy is None:
            missing_gt += 1
            continue
        _nearest_idx, distance = _nearest_superpoint_to_xy(np.asarray(gt_xy, dtype=np.float64), keypoints)
        if distance is None:
            continue
        distances.append(float(distance))
    summary: dict[str, object] = {
        "availability_count": int(len(distances)),
        "availability_missing_gt_count": int(missing_gt),
        "nearest_sp_distance_px_mean": None if not distances else float(np.mean(distances)),
        "nearest_sp_distance_px_median": None if not distances else float(np.median(distances)),
        "nearest_sp_distance_px_p90": None if not distances else float(np.quantile(distances, 0.90)),
    }
    for radius in radii_px:
        radius_value = float(radius)
        key = f"sp_availability_at_{int(radius_value) if radius_value.is_integer() else radius_value:g}px"
        summary[key] = None if not distances else float(np.mean([1.0 if value <= radius_value else 0.0 for value in distances]))
    return summary


def apply_superpoint_gt_oracle_snaps_to_matches(
    matches: Sequence[QueryTo3DMatch],
    keypoints: SuperPointKeypointSet,
    inlier_mask: np.ndarray,
    gt_xy_by_match: Sequence[np.ndarray | None],
    max_distance_px: float | None,
    patch_positive_by_token: Mapping[int, set[int]] | None = None,
    require_patch_positive: bool = False,
) -> tuple[list[QueryTo3DMatch], dict[str, object]]:
    refined = list(matches)
    mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != len(matches):
        raise ValueError("inlier_mask must have one value per match")
    if len(gt_xy_by_match) != len(matches):
        raise ValueError("gt_xy_by_match must have one value per match")
    patch_positive_by_token = patch_positive_by_token or {}
    applied_count = 0
    skipped_by_inlier = 0
    skipped_by_missing_gt = 0
    skipped_by_patch_positive = 0
    no_keypoint = 0
    too_far = 0
    inlier_count = 0
    offsets = []
    gt_distances = []
    scores = []
    for idx, match in enumerate(matches):
        if not bool(mask[idx]):
            skipped_by_inlier += 1
            continue
        inlier_count += 1
        if require_patch_positive and int(match.track_id) not in patch_positive_by_token.get(int(match.token_index), set()):
            skipped_by_patch_positive += 1
            continue
        gt_xy = gt_xy_by_match[idx]
        if gt_xy is None:
            skipped_by_missing_gt += 1
            continue
        nearest_idx, distance = _nearest_superpoint_to_xy(np.asarray(gt_xy, dtype=np.float64), keypoints)
        if nearest_idx is None or distance is None:
            no_keypoint += 1
            continue
        if max_distance_px is not None and float(distance) > float(max_distance_px):
            too_far += 1
            continue
        target_xy = np.asarray(keypoints.xy, dtype=np.float64).reshape(-1, 2)[int(nearest_idx)]
        offset = target_xy - np.asarray(match.xy, dtype=np.float64).reshape(2)
        refined[idx] = replace(match, xy=target_xy.astype(np.float64))
        applied_count += 1
        offsets.append(float(np.linalg.norm(offset)))
        gt_distances.append(float(distance))
        scores.append(float(np.asarray(keypoints.scores, dtype=np.float64).reshape(-1)[int(nearest_idx)]))
    return refined, {
        "mode": "superpoint_gt_oracle_patch_positive" if require_patch_positive else "superpoint_gt_oracle",
        "refined_count": int(applied_count),
        "applied_count": int(applied_count),
        "inlier_count": int(inlier_count),
        "offset_applied_ratio": float(applied_count / max(inlier_count, 1)),
        "skipped_by_inlier_count": int(skipped_by_inlier),
        "skipped_by_missing_gt_count": int(skipped_by_missing_gt),
        "skipped_by_patch_positive_count": int(skipped_by_patch_positive),
        "no_keypoint_count": int(no_keypoint),
        "too_far_count": int(too_far),
        "mean_offset_px": None if not offsets else float(np.mean(offsets)),
        "mean_gt_to_sp_distance_px": None if not gt_distances else float(np.mean(gt_distances)),
        "median_gt_to_sp_distance_px": None if not gt_distances else float(np.median(gt_distances)),
        "mean_score": None if not scores else float(np.mean(scores)),
        "keypoint_count": int(np.asarray(keypoints.xy).shape[0]),
        "max_distance_px": None if max_distance_px is None else float(max_distance_px),
    }


def apply_superpoint_snaps_to_matches(
    matches: Sequence[QueryTo3DMatch],
    keypoints: SuperPointKeypointSet,
    inlier_mask: np.ndarray,
    config: SuperPointSnapConfig,
) -> tuple[list[QueryTo3DMatch], dict[str, object]]:
    refined = list(matches)
    mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != len(matches):
        raise ValueError("inlier_mask must have one value per match")
    applied_count = 0
    skipped_by_inlier = 0
    inlier_count = 0
    low_confidence = 0
    offsets = []
    scores = []
    confidences = []
    for idx, match in enumerate(matches):
        if not bool(mask[idx]):
            skipped_by_inlier += 1
            continue
        inlier_count += 1
        result = snap_xy_to_superpoint_keypoint(np.asarray(match.xy, dtype=np.float64), keypoints, config)
        if not result.applied:
            low_confidence += 1
            continue
        refined[idx] = replace(
            match,
            xy=np.asarray(match.xy, dtype=np.float64).reshape(2) + result.offset_xy.astype(np.float64),
        )
        applied_count += 1
        offsets.append(float(np.linalg.norm(result.offset_xy)))
        scores.append(float(result.score))
        confidences.append(float(result.confidence))
    return refined, {
        "mode": "superpoint_snap",
        "refined_count": int(applied_count),
        "applied_count": int(applied_count),
        "inlier_count": int(inlier_count),
        "offset_applied_ratio": float(applied_count / max(inlier_count, 1)),
        "skipped_by_inlier_count": int(skipped_by_inlier),
        "low_confidence_count": int(low_confidence),
        "mean_offset_px": None if not offsets else float(np.mean(offsets)),
        "mean_score": None if not scores else float(np.mean(scores)),
        "mean_confidence": None if not confidences else float(np.mean(confidences)),
        "keypoint_count": int(np.asarray(keypoints.xy).shape[0]),
        "max_offset_px": float(config.max_offset_px),
        "min_score": float(config.min_score),
        "selection_strategy": config.selection_strategy,
    }


class HLocSuperPointKeypointDetector:
    def __init__(
        self,
        *,
        device: str = "cuda:0",
        nms_radius: int = 4,
        keypoint_threshold: float = 0.005,
        max_keypoints: int = -1,
        remove_borders: int = 4,
    ) -> None:
        try:
            import torch
            from hloc.extractors.superpoint import SuperPoint
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("HLoc SuperPoint is required for SuperPoint snapping") from exc
        self._torch = torch
        self.device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
        self.model = SuperPoint(
            {
                "nms_radius": int(nms_radius),
                "keypoint_threshold": float(keypoint_threshold),
                "max_keypoints": int(max_keypoints),
                "remove_borders": int(remove_borders),
                "fix_sampling": True,
            }
        ).eval().to(self.device)

    def detect(self, image: np.ndarray) -> SuperPointKeypointSet:
        gray = _to_gray(image)
        tensor = self._torch.from_numpy(gray[None, None].astype(np.float32)).to(self.device)
        with self._torch.no_grad():
            output = self.model({"image": tensor})
        xy = output["keypoints"][0].detach().cpu().numpy().astype(np.float32, copy=False)
        scores = output["scores"][0].detach().cpu().numpy().astype(np.float32, copy=False)
        descriptors = output["descriptors"][0].detach().cpu().numpy().astype(np.float32, copy=False).T
        return SuperPointKeypointSet(xy=xy, scores=scores, descriptors=descriptors)


def apply_lowlevel_offsets_to_matches(
    matches: Sequence[QueryTo3DMatch],
    query_image_id: str,
    image_by_id: Mapping[str, np.ndarray],
    support_bank: LowLevelSupportBank,
    inlier_mask: np.ndarray,
    config: LowLevelOffsetSidecarConfig,
    query_viewing_ray_by_track: Mapping[int, np.ndarray] | None = None,
) -> tuple[list[QueryTo3DMatch], dict[str, object]]:
    refined = list(matches)
    mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != len(matches):
        raise ValueError("inlier_mask must have one value per match")
    query_image = image_by_id.get(str(query_image_id))
    if query_image is None:
        raise ValueError(f"query image missing: {query_image_id}")
    processed_cache: dict[str, np.ndarray] = {
        str(query_image_id): preprocess_lowlevel_image(query_image, config.mode),
    }
    query_viewing_ray_by_track = query_viewing_ray_by_track or {}
    applied_count = 0
    skipped_by_inlier = 0
    inlier_count = 0
    missing_support = 0
    missing_image = 0
    low_confidence = 0
    offsets = []
    scores = []
    confidences = []
    for idx, match in enumerate(matches):
        if not bool(mask[idx]):
            skipped_by_inlier += 1
            continue
        inlier_count += 1
        support = select_support_observation(
            support_bank,
            int(match.track_id),
            query_viewing_ray=query_viewing_ray_by_track.get(int(match.track_id)),
        )
        if support is None:
            missing_support += 1
            continue
        support_image = image_by_id.get(str(support.image_id))
        if support_image is None:
            missing_image += 1
            continue
        support_key = str(support.image_id)
        if support_key not in processed_cache:
            processed_cache[support_key] = preprocess_lowlevel_image(support_image, config.mode)
        result = _estimate_ncc_patch_offset_preprocessed(
            query_values=processed_cache[str(query_image_id)],
            support_values=processed_cache[support_key],
            query_xy=np.asarray(match.xy, dtype=np.float64),
            support_xy=np.asarray(support.xy, dtype=np.float64),
            config=config,
        )
        if not result.applied:
            low_confidence += 1
            continue
        refined[idx] = replace(
            match,
            xy=np.asarray(match.xy, dtype=np.float64).reshape(2) + result.offset_xy.astype(np.float64),
        )
        applied_count += 1
        offsets.append(float(np.linalg.norm(result.offset_xy)))
        scores.append(float(result.score))
        confidences.append(float(result.confidence))
    return refined, {
        "mode": config.mode,
        "refined_count": int(applied_count),
        "applied_count": int(applied_count),
        "inlier_count": int(inlier_count),
        "offset_applied_ratio": float(applied_count / max(inlier_count, 1)),
        "skipped_by_inlier_count": int(skipped_by_inlier),
        "missing_support_count": int(missing_support),
        "missing_image_count": int(missing_image),
        "low_confidence_count": int(low_confidence),
        "mean_offset_px": None if not offsets else float(np.mean(offsets)),
        "mean_score": None if not scores else float(np.mean(scores)),
        "mean_confidence": None if not confidences else float(np.mean(confidences)),
        "template_radius_px": int(config.template_radius_px),
        "search_radius_px": int(config.search_radius_px),
        "min_score": float(config.min_score),
        "min_confidence": float(config.min_confidence),
    }
