"""Query dense VFM token to rendered Gaussian VFM map matching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feature_extract.vfm.query_to_3d_matching import (
    QueryTo3DMatch,
    _flatten_query_features,
    _query_topk_and_landmark_best,
    normalize_rows,
)


@dataclass(frozen=True)
class QueryToRenderMatchingConfig:
    top_k: int = 2
    ratio_threshold: float | None = 0.9
    min_similarity: float = 0.0
    mutual: bool = False
    query_token_step: int = 1
    max_matches: int | None = None
    block_size: int = 512

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.ratio_threshold is not None and not 0.0 < float(self.ratio_threshold) <= 1.0:
            raise ValueError("ratio_threshold must be in (0, 1]")
        if self.query_token_step <= 0:
            raise ValueError("query_token_step must be positive")
        if self.max_matches is not None and self.max_matches <= 0:
            raise ValueError("max_matches must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")


def _render_pixel_xy(width: int, height: int, image_width: int, image_height: int) -> np.ndarray:
    xs = np.linspace(0.0, float(image_width - 1), width, dtype=np.float64)
    ys = np.linspace(0.0, float(image_height - 1), height, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float64)


def match_query_tokens_to_rendered_map(
    query_feature_map: np.ndarray,
    rendered_feature_map: np.ndarray,
    rendered_xyz_map: np.ndarray,
    visibility_mask: np.ndarray,
    config: QueryToRenderMatchingConfig | None = None,
    image_width: int = 1024,
    image_height: int = 576,
) -> list[QueryTo3DMatch]:
    """Match query VFM tokens to visible rendered map pixels.

    The output reuses `QueryTo3DMatch` so existing PnP and reprojection
    precision utilities can consume rendered-map correspondences directly.
    Render pixel indices are encoded as non-negative `track_id` values.
    """

    cfg = config or QueryToRenderMatchingConfig()
    rendered_feature_map = np.asarray(rendered_feature_map, dtype=np.float32)
    rendered_xyz_map = np.asarray(rendered_xyz_map, dtype=np.float32)
    visibility_mask = np.asarray(visibility_mask, dtype=bool)
    if rendered_feature_map.ndim != 3:
        raise ValueError("rendered_feature_map must have shape (C, H, W)")
    channels, render_height, render_width = rendered_feature_map.shape
    if rendered_xyz_map.shape != (render_height, render_width, 3):
        raise ValueError("rendered_xyz_map must have shape (H, W, 3)")
    if visibility_mask.shape != (render_height, render_width):
        raise ValueError("visibility_mask must have shape (H, W)")

    query_features, query_xy, token_indices = _flatten_query_features(
        query_feature_map,
        image_width=image_width,
        image_height=image_height,
        step=cfg.query_token_step,
    )
    query_features, valid_query = normalize_rows(query_features)
    visible_flat = visibility_mask.reshape(-1)
    render_features = rendered_feature_map.reshape(channels, -1).T[visible_flat]
    render_xyz = rendered_xyz_map.reshape(-1, 3)[visible_flat].astype(np.float64)
    render_pixel_indices = np.flatnonzero(visible_flat).astype(np.int64)
    if render_features.size == 0:
        return []
    render_features, valid_render = normalize_rows(render_features)
    if not np.all(valid_render):
        render_features = render_features[valid_render]
        render_xyz = render_xyz[valid_render]
        render_pixel_indices = render_pixel_indices[valid_render]

    valid_query_indices = np.flatnonzero(valid_query)
    if valid_query_indices.size == 0 or render_features.shape[0] == 0:
        return []
    query_features = query_features[valid_query_indices]
    query_xy = query_xy[valid_query_indices]
    token_indices = token_indices[valid_query_indices]

    top_indices, top_scores, render_best_query = _query_topk_and_landmark_best(
        query_features,
        render_features,
        top_k=cfg.top_k,
        block_size=cfg.block_size,
    )
    matches: list[QueryTo3DMatch] = []
    for query_idx in range(query_features.shape[0]):
        render_idx = int(top_indices[query_idx, 0])
        if render_idx < 0:
            continue
        similarity = float(top_scores[query_idx, 0])
        if similarity < float(cfg.min_similarity):
            continue
        ratio = 0.0
        if top_scores.shape[1] >= 2:
            best_distance = max(0.0, 1.0 - similarity)
            second_distance = max(1e-6, 1.0 - float(top_scores[query_idx, 1]))
            ratio = float(best_distance / second_distance)
            if cfg.ratio_threshold is not None and ratio > float(cfg.ratio_threshold):
                continue
        if cfg.mutual and int(render_best_query[render_idx]) != query_idx:
            continue
        matches.append(
            QueryTo3DMatch(
                token_index=int(token_indices[query_idx]),
                xy=query_xy[query_idx].astype(np.float64, copy=True),
                track_id=int(render_pixel_indices[render_idx]),
                xyz=render_xyz[render_idx].astype(np.float64, copy=True),
                similarity=similarity,
                ratio=ratio,
                landmark_variance=0.0,
            )
        )
    matches.sort(key=lambda item: item.similarity, reverse=True)
    if cfg.max_matches is not None:
        matches = matches[: int(cfg.max_matches)]
    return matches
