"""Pose hypothesis scoring for rendered-keypoint VFM localization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import (
    QueryTo3DMatch,
    match_reprojection_errors,
    match_spatial_distribution_stats,
)


@dataclass(frozen=True)
class PoseHypothesisScore:
    score: float
    inlier_count: int
    weighted_residual: float
    confidence_mean: float
    coverage: float
    degeneracy_penalty: float


def annotate_measurement_uncertainty(
    matches: Sequence[QueryTo3DMatch],
    *,
    base_sigma_px: float,
    fine_refined: bool = False,
    min_sigma_px: float = 1.0,
    confidence_scale: float = 0.5,
) -> list[QueryTo3DMatch]:
    """Attach patch-level measurement sigma to matches without changing geometry."""

    from dataclasses import replace

    sigma0 = float(base_sigma_px) * (0.5 if bool(fine_refined) else 1.0)
    sigma0 = max(float(sigma0), float(min_sigma_px))
    annotated = []
    for match in matches:
        conf = 0.0 if match.pnp_soft_score is None else float(np.clip(match.pnp_soft_score, 0.0, 1.0))
        sigma = sigma0 * (1.0 + float(confidence_scale) * (1.0 - conf))
        annotated.append(replace(match, measurement_sigma_px=max(float(sigma), float(min_sigma_px))))
    return annotated


def _match_confidence(match: QueryTo3DMatch) -> float:
    if match.pnp_soft_score is not None and np.isfinite(float(match.pnp_soft_score)):
        return float(match.pnp_soft_score)
    return float((float(match.similarity) + 1.0) * 0.5)


def coverage_preserving_match_filter(
    matches: Sequence[QueryTo3DMatch],
    camera: ColmapCamera,
    *,
    grid_size: int = 4,
    max_per_cell: int = 8,
    min_confidence: float | None = None,
    max_total: int | None = None,
) -> list[QueryTo3DMatch]:
    """Keep confident matches while preserving coarse 2D spatial coverage."""

    values = list(matches)
    if not values:
        return []
    grid = max(int(grid_size), 1)
    per_cell = max(int(max_per_cell), 1)
    width = max(float(camera.width), 1.0)
    height = max(float(camera.height), 1.0)
    buckets: dict[tuple[int, int], list[QueryTo3DMatch]] = {}
    for match in values:
        confidence = _match_confidence(match)
        if min_confidence is not None and confidence < float(min_confidence):
            continue
        xy = np.asarray(match.xy, dtype=np.float64).reshape(2)
        gx = int(np.clip(np.floor(xy[0] / width * grid), 0, grid - 1))
        gy = int(np.clip(np.floor(xy[1] / height * grid), 0, grid - 1))
        buckets.setdefault((gx, gy), []).append(match)
    if not buckets and min_confidence is not None:
        return coverage_preserving_match_filter(
            values,
            camera,
            grid_size=grid,
            max_per_cell=per_cell,
            min_confidence=None,
            max_total=max_total,
        )
    kept: list[QueryTo3DMatch] = []
    for key in sorted(buckets):
        ranked = sorted(
            buckets[key],
            key=lambda item: (_match_confidence(item), float(item.similarity)),
            reverse=True,
        )
        kept.extend(ranked[:per_cell])
    kept.sort(key=lambda item: (_match_confidence(item), float(item.similarity)), reverse=True)
    if max_total is not None and int(max_total) > 0:
        kept = kept[: int(max_total)]
    return kept


def score_pose_hypothesis(
    matches: Sequence[QueryTo3DMatch],
    pose_w2c: np.ndarray | None,
    camera: ColmapCamera,
    *,
    inlier_threshold_px: float,
    inlier_mask: np.ndarray | None = None,
) -> PoseHypothesisScore:
    """Score a PnP hypothesis with confidence, uncertainty, coverage, and degeneracy."""

    values = list(matches)
    if pose_w2c is None or not values:
        return PoseHypothesisScore(float("-inf"), 0, float("inf"), 0.0, 0.0, 1.0)
    residuals = match_reprojection_errors(values, pose_w2c, camera)
    if inlier_mask is None:
        inliers = residuals <= float(inlier_threshold_px)
    else:
        inliers = np.asarray(inlier_mask, dtype=bool).reshape(-1)
        if inliers.shape[0] != len(values):
            raise ValueError("inlier_mask must have one value per match")
    if not np.any(inliers):
        return PoseHypothesisScore(float("-inf"), 0, float("inf"), 0.0, 0.0, 1.0)
    sigmas = np.asarray(
        [
            float(match.measurement_sigma_px)
            if match.measurement_sigma_px is not None and np.isfinite(float(match.measurement_sigma_px))
            else max(float(inlier_threshold_px) * 0.5, 1.0)
            for match in values
        ],
        dtype=np.float64,
    )
    confidences = np.asarray(
        [
            float(np.clip(match.pnp_soft_score, 1e-4, 1.0))
            if match.pnp_soft_score is not None and np.isfinite(float(match.pnp_soft_score))
            else float(np.clip((float(match.similarity) + 1.0) * 0.5, 1e-4, 1.0))
            for match in values
        ],
        dtype=np.float64,
    )
    normalized_residual = residuals[inliers] / np.maximum(sigmas[inliers], 1e-6)
    weighted_residual = float(np.mean(normalized_residual))
    confidence_mean = float(np.mean(confidences[inliers]))
    spatial = match_spatial_distribution_stats(values, int(camera.width), int(camera.height), inliers)
    coverage = float(spatial.get("grid_4x4_occupancy_frac") or 0.0)
    line_ratio = float(spatial.get("xyz_linearity_ratio") or 0.0)
    xy_ratio = float(spatial.get("xy_pca_minor_major_ratio") or 0.0)
    degeneracy_penalty = float(max(0.0, 0.12 - line_ratio) + max(0.0, 0.05 - xy_ratio))
    inlier_bonus = float(np.log1p(float(np.sum(inliers))))
    score = (
        0.70 * confidence_mean
        - 0.25 * weighted_residual
        + 0.20 * coverage
        + 0.08 * inlier_bonus
        - 0.50 * degeneracy_penalty
    )
    return PoseHypothesisScore(
        score=float(score),
        inlier_count=int(np.sum(inliers)),
        weighted_residual=weighted_residual,
        confidence_mean=confidence_mean,
        coverage=coverage,
        degeneracy_penalty=degeneracy_penalty,
    )
