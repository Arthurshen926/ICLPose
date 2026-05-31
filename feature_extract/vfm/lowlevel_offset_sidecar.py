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

    def __post_init__(self) -> None:
        xy = np.asarray(self.xy)
        scores = np.asarray(self.scores)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must be Nx2")
        if scores.ndim != 1 or scores.shape[0] != xy.shape[0]:
            raise ValueError("scores must be N")


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
        return SuperPointKeypointSet(xy=xy, scores=scores)


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
