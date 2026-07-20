"""Target-free global-alignment evidence from frozen LoFTR pair caches.

The local LoFTR anchor probe asks whether a cached correspondence is near one
query/support anchor pair.  This module tests a different observation: after
fitting one robust *image-pair* support-to-query homography from every cached
LoFTR correspondence, does a fixed candidate support anchor agree with the
whole-image geometric phase?  The fit never receives a landmark identity,
candidate score, pose, target, or support-image shortlist.

The values returned here are raw, per-view features.  A failed or unreliable
model is explicit unknown data (NaN with ``usable=False``), never negative
evidence and never a fallback geometric score.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from feature_extract.vfm.localization.frozen_loftr_pair_cache import FrozenLoFTRPairCache


FROZEN_LOFTR_GLOBAL_ALIGNMENT_EVIDENCE_FORMAT = (
    "frozen_loftr_global_alignment_evidence_v1"
)
LOFTR_GLOBAL_ALIGNMENT_FEATURE_NAMES = (
    "loftr_homography_anchor_forward_error_px",
    "loftr_homography_anchor_symmetric_error_px",
    "loftr_homography_anchor_nearest_inlier_support_px",
    "loftr_homography_inlier_ratio",
    "loftr_homography_inlier_count_log1p",
    "loftr_homography_median_inlier_error_px",
    "loftr_homography_query_inlier_hull_area_fraction",
    "loftr_homography_support_inlier_hull_area_fraction",
    "loftr_homography_pair_match_count_log1p",
)


@dataclass(frozen=True)
class FrozenLoFTRHomographyConfig:
    """Predeclared robust fitting policy for all query/support image pairs."""

    ransac_reprojection_threshold_px: float = 8.0
    max_iterations: int = 2048
    confidence: float = 0.995
    min_match_count: int = 16
    min_inlier_count: int = 12
    seed_namespace: str = "frozen_loftr_global_alignment_ransac_v1"

    def __post_init__(self) -> None:
        if (
            not np.isfinite(float(self.ransac_reprojection_threshold_px))
            or float(self.ransac_reprojection_threshold_px) <= 0.0
            or int(self.max_iterations) <= 0
            or not np.isfinite(float(self.confidence))
            or not 0.0 < float(self.confidence) < 1.0
            or int(self.min_match_count) < 4
            or int(self.min_inlier_count) < 4
            or int(self.min_inlier_count) > int(self.min_match_count)
            or not str(self.seed_namespace)
        ):
            raise ValueError("frozen LoFTR homography configuration is invalid")

    def metadata(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "model": "support_to_query_homography_ransac_v1",
            "fit_uses_all_cached_matches": True,
            "fit_match_order": "descending_confidence_then_cache_index_v1",
            "invalid_model_policy": "explicit_unknown_v1",
        }


@dataclass(frozen=True)
class FrozenLoFTRHomographyModel:
    """One robust, target-free image-pair transform and its reliability data."""

    support_to_query: np.ndarray
    query_to_support: np.ndarray
    inlier_support_xy: np.ndarray
    inlier_query_xy: np.ndarray
    inlier_ratio: float
    inlier_count: int
    median_inlier_error_px: float
    query_inlier_hull_area_fraction: float
    support_inlier_hull_area_fraction: float
    pair_match_count: int


def _as_points(value: np.ndarray, *, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or np.any(~np.isfinite(points)):
        raise ValueError(f"{name} must be finite [N, 2] coordinates")
    return points


def _source_size(value: Sequence[int | float]) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError("source image size must be width,height")
    width, height = (float(item) for item in value)
    if not np.isfinite([width, height]).all() or width <= 0.0 or height <= 0.0:
        raise ValueError("source image size is invalid")
    return width, height


def _deterministic_seed(*, namespace: str, support_image_id: str) -> int:
    payload = f"{namespace}\0{support_image_id}".encode("utf8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") % (2**31 - 1)


def _apply_homography(homography: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(homography, dtype=np.float64)
    xy = _as_points(points, name="homography points")
    if matrix.shape != (3, 3) or np.any(~np.isfinite(matrix)):
        raise ValueError("homography matrix is invalid")
    homogeneous = np.concatenate(
        [xy, np.ones((len(xy), 1), dtype=np.float64)], axis=1
    )
    projected = homogeneous @ matrix.T
    denominator = projected[:, 2]
    valid = np.isfinite(denominator) & (np.abs(denominator) > 1e-8)
    output = np.full((len(xy), 2), np.nan, dtype=np.float64)
    output[valid] = projected[valid, :2] / denominator[valid, None]
    valid &= np.isfinite(output).all(axis=1)
    return output, valid


def _convex_hull_area_fraction(points: np.ndarray, *, source_size: Sequence[int | float]) -> float:
    xy = _as_points(points, name="inlier points")
    width, height = _source_size(source_size)
    if len(xy) < 3:
        return 0.0
    hull = cv2.convexHull(xy.astype(np.float32, copy=False))
    area = float(cv2.contourArea(hull))
    fraction = area / float(width * height)
    if not np.isfinite(fraction) or fraction < 0.0:
        raise RuntimeError("LoFTR homography hull area is invalid")
    return float(min(fraction, 1.0))


def fit_frozen_loftr_support_to_query_homography(
    *,
    matched_query_xy: np.ndarray,
    matched_support_xy: np.ndarray,
    match_confidence: np.ndarray,
    support_image_id: str,
    source_size: Sequence[int | float],
    config: FrozenLoFTRHomographyConfig = FrozenLoFTRHomographyConfig(),
) -> FrozenLoFTRHomographyModel | None:
    """Fit a deterministic, robust support-to-query model from one full pair.

    Sorting only fixes RANSAC input order; it deliberately keeps every cached
    match.  The returned inliers are recomputed from the final model, so cache
    consumers do not depend on OpenCV's internal mask convention.
    """

    query = _as_points(matched_query_xy, name="matched query points")
    support = _as_points(matched_support_xy, name="matched support points")
    confidence = np.asarray(match_confidence, dtype=np.float64).reshape(-1)
    _source_size(source_size)
    if (
        query.shape != support.shape
        or len(confidence) != len(query)
        or np.any(~np.isfinite(confidence))
        or np.any(confidence < 0.0)
        or np.any(confidence > 1.0)
        or not str(support_image_id)
    ):
        raise ValueError("frozen LoFTR homography pair inputs are invalid")
    if len(query) < int(config.min_match_count):
        return None

    # Do not discard a match: this ordering merely makes OpenCV RANSAC fully
    # reproducible for immutable cache arrays and emphasizes confident samples.
    order = np.lexsort((np.arange(len(query), dtype=np.int64), -confidence))
    query = query[order]
    support = support[order]
    cv2.setRNGSeed(
        _deterministic_seed(
            namespace=config.seed_namespace, support_image_id=str(support_image_id)
        )
    )
    homography, _mask = cv2.findHomography(
        support.astype(np.float32, copy=False),
        query.astype(np.float32, copy=False),
        method=cv2.RANSAC,
        ransacReprojThreshold=float(config.ransac_reprojection_threshold_px),
        maxIters=int(config.max_iterations),
        confidence=float(config.confidence),
    )
    if homography is None:
        return None
    forward = np.asarray(homography, dtype=np.float64)
    if forward.shape != (3, 3) or np.any(~np.isfinite(forward)):
        return None
    if abs(float(forward[2, 2])) <= 1e-12:
        return None
    forward = forward / float(forward[2, 2])
    try:
        inverse = np.linalg.inv(forward)
    except np.linalg.LinAlgError:
        return None
    if np.any(~np.isfinite(inverse)) or abs(float(inverse[2, 2])) <= 1e-12:
        return None
    inverse = inverse / float(inverse[2, 2])
    projected, projectable = _apply_homography(forward, support)
    errors = np.full((len(query),), np.inf, dtype=np.float64)
    errors[projectable] = np.linalg.norm(projected[projectable] - query[projectable], axis=1)
    inliers = projectable & (errors <= float(config.ransac_reprojection_threshold_px))
    inlier_count = int(np.sum(inliers))
    if inlier_count < int(config.min_inlier_count):
        return None
    inlier_errors = errors[inliers]
    ratio = float(inlier_count / len(query))
    median_error = float(np.median(inlier_errors))
    if not np.isfinite(ratio) or not np.isfinite(median_error):
        return None
    return FrozenLoFTRHomographyModel(
        support_to_query=forward.astype(np.float64, copy=False),
        query_to_support=inverse.astype(np.float64, copy=False),
        inlier_support_xy=support[inliers].astype(np.float64, copy=False),
        inlier_query_xy=query[inliers].astype(np.float64, copy=False),
        inlier_ratio=ratio,
        inlier_count=inlier_count,
        median_inlier_error_px=median_error,
        query_inlier_hull_area_fraction=_convex_hull_area_fraction(
            query[inliers], source_size=source_size
        ),
        support_inlier_hull_area_fraction=_convex_hull_area_fraction(
            support[inliers], source_size=source_size
        ),
        pair_match_count=int(len(query)),
    )


def loftr_global_alignment_pair_features(
    *,
    model: FrozenLoFTRHomographyModel,
    query_xy: np.ndarray,
    support_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate fixed anchor pairs under one already-fitted global model."""

    query = _as_points(query_xy, name="query anchors")
    support = _as_points(support_xy, name="support anchors")
    if query.shape != support.shape:
        raise ValueError("query and support anchor coordinates differ")
    output = np.full(
        (len(query), len(LOFTR_GLOBAL_ALIGNMENT_FEATURE_NAMES)), np.nan, dtype=np.float32
    )
    usable = np.zeros((len(query),), dtype=bool)
    if len(query) == 0:
        return output, usable
    predicted_query, forward_valid = _apply_homography(model.support_to_query, support)
    predicted_support, backward_valid = _apply_homography(model.query_to_support, query)
    valid = forward_valid & backward_valid
    if not np.any(valid):
        return output, usable
    forward_error = np.linalg.norm(predicted_query[valid] - query[valid], axis=1)
    backward_error = np.linalg.norm(predicted_support[valid] - support[valid], axis=1)
    symmetric_error = 0.5 * (forward_error + backward_error)
    support_delta = support[valid, None, :] - model.inlier_support_xy[None, :, :]
    nearest_support = np.sqrt(np.min(np.sum(support_delta**2, axis=2), axis=1))
    fixed_values = np.asarray(
        [
            float(model.inlier_ratio),
            float(np.log1p(model.inlier_count)),
            float(model.median_inlier_error_px),
            float(model.query_inlier_hull_area_fraction),
            float(model.support_inlier_hull_area_fraction),
            float(np.log1p(model.pair_match_count)),
        ],
        dtype=np.float64,
    )
    values = np.column_stack(
        [forward_error, symmetric_error, nearest_support, np.tile(fixed_values, (int(np.sum(valid)), 1))]
    )
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise RuntimeError("LoFTR global alignment anchor features are invalid")
    output[valid] = values.astype(np.float32, copy=False)
    usable[valid] = True
    return output, usable


def frozen_loftr_global_alignment_view_features(
    *,
    cache: FrozenLoFTRPairCache,
    query_xy: np.ndarray,
    candidate_support_xy: np.ndarray,
    candidate_support_image_ids: np.ndarray,
    candidate_view_valid: np.ndarray,
    source_size: Sequence[int | float],
    config: FrozenLoFTRHomographyConfig = FrozenLoFTRHomographyConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Materialize global alignment evidence for fixed candidate support views.

    The cache is indexed only by support image ids already present in the
    immutable maplet mixture.  A single fitted homography is shared by every
    anchor belonging to that image; no candidate-specific model is fitted.
    """

    query = _as_points(query_xy, name="query coordinates")
    support_xy = np.asarray(candidate_support_xy, dtype=np.float64)
    support_ids = np.asarray(candidate_support_image_ids).astype(str)
    valid = np.asarray(candidate_view_valid, dtype=bool)
    _source_size(source_size)
    if (
        support_xy.ndim != 4
        or support_xy.shape != (*support_ids.shape, 2)
        or support_ids.shape != valid.shape
        or support_ids.shape[0] != len(query)
        or np.any(~np.isfinite(support_xy[valid]))
        or np.any(support_ids[valid] == "")
    ):
        raise ValueError("frozen LoFTR global alignment view inputs are invalid")
    feature_count = len(LOFTR_GLOBAL_ALIGNMENT_FEATURE_NAMES)
    features = np.full((*support_ids.shape, feature_count), np.nan, dtype=np.float32)
    usable = np.zeros(support_ids.shape, dtype=bool)
    pair_match_counts = np.zeros(support_ids.shape, dtype=np.int32)
    model_valid = np.zeros(support_ids.shape, dtype=bool)
    flat_ids = support_ids.reshape(-1)
    flat_valid = valid.reshape(-1)
    flat_support_xy = support_xy.reshape(-1, 2)
    point_indices = np.broadcast_to(
        np.arange(len(query), dtype=np.int64)[:, None, None], support_ids.shape
    ).reshape(-1)
    for image_id in np.unique(flat_ids[flat_valid]).tolist():
        selected = np.flatnonzero(flat_valid & (flat_ids == str(image_id)))
        pair_index = cache.image_index(str(image_id))
        matched_query, matched_support, confidence = cache.matches_for_index(pair_index)
        pair_match_counts.reshape(-1)[selected] = int(len(confidence))
        model = fit_frozen_loftr_support_to_query_homography(
            matched_query_xy=matched_query,
            matched_support_xy=matched_support,
            match_confidence=confidence,
            support_image_id=str(image_id),
            source_size=source_size,
            config=config,
        )
        if model is None:
            continue
        model_valid.reshape(-1)[selected] = True
        values, available = loftr_global_alignment_pair_features(
            model=model,
            query_xy=query[point_indices[selected]],
            support_xy=flat_support_xy[selected],
        )
        features.reshape(-1, feature_count)[selected] = values
        usable.reshape(-1)[selected] = available
    if (
        np.any(usable & ~valid)
        or np.any(model_valid & ~valid)
        or np.any(~np.isfinite(features[usable]))
        or np.any(usable & ~model_valid)
    ):
        raise RuntimeError("frozen LoFTR global alignment materialization is invalid")
    return features, usable, pair_match_counts, model_valid
