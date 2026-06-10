"""Render-RGB query/render descriptor samples for geometry-aware VFM adaptation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from feature_extract.vfm.patch_selector_training import PatchSelectorTrainingSet
from feature_extract.vfm.query_to_3d_matching import normalize_rows
from feature_extract.vfm.rendered_keypoint_selector_samples import (
    RenderedKeypointSelectorSampleConfig,
    build_rendered_keypoint_selector_samples,
)


@dataclass(frozen=True)
class RenderRGBKeypointPairDiagnostics:
    query_keypoint_count: int
    render_keypoint_count: int
    finite_pair_count: int
    positive_pair_count: int
    negative_pair_count: int
    raw_top1_gt16: float
    raw_top1_gt32: float
    raw_top1_median_reprojection_px: float | None
    positive_similarity_mean: float | None
    negative_similarity_mean: float | None
    similarity_gap_mean: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _safe_mean(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.mean(values))


def render_rgb_keypoint_pair_diagnostics(
    query_descriptors: np.ndarray,
    render_descriptors: np.ndarray,
    reprojection_errors_px: np.ndarray,
    *,
    positive_threshold_px: float = 16.0,
    negative_threshold_px: float = 32.0,
) -> RenderRGBKeypointPairDiagnostics:
    """Summarize raw query/render descriptor geometry before training."""

    query = np.asarray(query_descriptors, dtype=np.float32)
    render = np.asarray(render_descriptors, dtype=np.float32)
    errors = np.asarray(reprojection_errors_px, dtype=np.float32)
    if query.ndim != 2 or render.ndim != 2:
        raise ValueError("descriptors must have shape (N, C)")
    if query.shape[1] != render.shape[1]:
        raise ValueError("query and render descriptor dimensions must match")
    if errors.shape != (query.shape[0], render.shape[0]):
        raise ValueError("reprojection_errors_px must have shape (Q, R)")
    finite = np.isfinite(errors)
    if query.shape[0] == 0 or render.shape[0] == 0:
        return RenderRGBKeypointPairDiagnostics(
            query_keypoint_count=int(query.shape[0]),
            render_keypoint_count=int(render.shape[0]),
            finite_pair_count=int(np.sum(finite)),
            positive_pair_count=0,
            negative_pair_count=0,
            raw_top1_gt16=0.0,
            raw_top1_gt32=0.0,
            raw_top1_median_reprojection_px=None,
            positive_similarity_mean=None,
            negative_similarity_mean=None,
            similarity_gap_mean=None,
        )
    query_norm, query_valid = normalize_rows(query)
    render_norm, render_valid = normalize_rows(render)
    scores = query_norm @ render_norm.T
    scores[~query_valid, :] = -np.inf
    scores[:, ~render_valid] = -np.inf
    scores[~finite] = -np.inf
    valid_rows = np.any(np.isfinite(scores), axis=1)
    if np.any(valid_rows):
        top1 = np.argmax(scores[valid_rows], axis=1)
        top1_errors = errors[valid_rows, top1]
        raw_top1_gt16 = float(np.mean(top1_errors <= 16.0))
        raw_top1_gt32 = float(np.mean(top1_errors <= 32.0))
        raw_top1_median = float(np.median(top1_errors[np.isfinite(top1_errors)]))
    else:
        raw_top1_gt16 = 0.0
        raw_top1_gt32 = 0.0
        raw_top1_median = None
    positive = finite & (errors <= float(positive_threshold_px))
    negative = finite & (errors >= float(negative_threshold_px))
    pos_mean = _safe_mean(scores[positive])
    neg_mean = _safe_mean(scores[negative])
    gap = None if pos_mean is None or neg_mean is None else float(pos_mean - neg_mean)
    return RenderRGBKeypointPairDiagnostics(
        query_keypoint_count=int(query.shape[0]),
        render_keypoint_count=int(render.shape[0]),
        finite_pair_count=int(np.sum(finite)),
        positive_pair_count=int(np.sum(positive)),
        negative_pair_count=int(np.sum(negative)),
        raw_top1_gt16=raw_top1_gt16,
        raw_top1_gt32=raw_top1_gt32,
        raw_top1_median_reprojection_px=raw_top1_median,
        positive_similarity_mean=pos_mean,
        negative_similarity_mean=neg_mean,
        similarity_gap_mean=gap,
    )


def build_render_rgb_keypoint_adapter_samples(
    query_descriptors: np.ndarray,
    render_descriptors: np.ndarray,
    reprojection_errors_px: np.ndarray,
    config: RenderedKeypointSelectorSampleConfig | None = None,
) -> PatchSelectorTrainingSet:
    """Build training samples and attach render-RGB matching diagnostics."""

    cfg = config or RenderedKeypointSelectorSampleConfig()
    diagnostics = render_rgb_keypoint_pair_diagnostics(
        query_descriptors,
        render_descriptors,
        reprojection_errors_px,
        positive_threshold_px=float(cfg.positive_threshold_px),
        negative_threshold_px=float(cfg.negative_threshold_px),
    )
    samples = build_rendered_keypoint_selector_samples(
        query_descriptors,
        render_descriptors,
        reprojection_errors_px,
        cfg,
    )
    metadata = dict(samples.metadata or {})
    metadata.update(
        {
            "source": "render_rgb_radio_keypoints",
            "diagnostics": diagnostics.to_dict(),
        }
    )
    return PatchSelectorTrainingSet(
        query_features=samples.query_features,
        positive_features=samples.positive_features,
        positive_mask=samples.positive_mask,
        negative_features=samples.negative_features,
        positive_reprojection_distances=samples.positive_reprojection_distances,
        negative_reprojection_distances=samples.negative_reprojection_distances,
        metadata=metadata,
    )
