from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocalLikelihoodResult:
    xy_px: np.ndarray
    local_log_probs: np.ndarray
    mean_xy_px: np.ndarray
    cov_query_2x2: np.ndarray
    mode_xy_px: np.ndarray
    mode_probability: float
    dustbin_probability: float
    gt_in_window: bool | None
    target_is_dustbin: bool


def _bilinear_sample(feature_map: np.ndarray, xy: np.ndarray, image_width: int, image_height: int) -> tuple[np.ndarray, np.ndarray]:
    fmap = np.asarray(feature_map, dtype=np.float64)
    if fmap.ndim != 3:
        raise ValueError("query_feature_map must have shape (C, H, W)")
    coords = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    channels, height, width = fmap.shape
    gx = coords[:, 0] * float(width - 1) / max(float(image_width - 1), 1.0)
    gy = coords[:, 1] * float(height - 1) / max(float(image_height - 1), 1.0)
    valid = (gx >= 0.0) & (gy >= 0.0) & (gx <= float(width - 1)) & (gy <= float(height - 1))
    x0 = np.floor(np.clip(gx, 0.0, float(width - 1))).astype(np.int64)
    y0 = np.floor(np.clip(gy, 0.0, float(height - 1))).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    wx = np.clip(gx - x0.astype(np.float64), 0.0, 1.0)
    wy = np.clip(gy - y0.astype(np.float64), 0.0, 1.0)
    values = (
        fmap[:, y0, x0].T * ((1.0 - wx) * (1.0 - wy))[:, None]
        + fmap[:, y0, x1].T * (wx * (1.0 - wy))[:, None]
        + fmap[:, y1, x0].T * ((1.0 - wx) * wy)[:, None]
        + fmap[:, y1, x1].T * (wx * wy)[:, None]
    )
    values[~valid] = 0.0
    return values.reshape(-1, channels), valid


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    max_value = float(np.max(values))
    shifted = values - max_value
    return shifted - np.log(np.sum(np.exp(shifted)))


def compute_local_likelihood(
    *,
    anchor_descriptor: np.ndarray,
    query_feature_map: np.ndarray,
    center_xy_px: np.ndarray,
    image_width: int,
    image_height: int,
    search_radius_px: float,
    step_px: float,
    temperature: float = 1.0,
    dustbin_logit: float | None = None,
    gt_xy_px: np.ndarray | None = None,
) -> LocalLikelihoodResult:
    """Build a query-side local likelihood around a projected anchor center."""

    radius = float(search_radius_px)
    step = float(step_px)
    if radius < 0.0 or step <= 0.0:
        raise ValueError("search_radius_px must be non-negative and step_px must be positive")
    offsets = np.arange(-radius, radius + 0.5 * step, step, dtype=np.float64)
    dx, dy = np.meshgrid(offsets, offsets)
    center = np.asarray(center_xy_px, dtype=np.float64).reshape(2)
    xy = np.stack([center[0] + dx.reshape(-1), center[1] + dy.reshape(-1)], axis=1)
    sampled, valid = _bilinear_sample(query_feature_map, xy, int(image_width), int(image_height))
    desc = np.asarray(anchor_descriptor, dtype=np.float64).reshape(-1)
    logits = sampled @ desc / max(float(temperature), 1e-8)
    logits[~valid] = -1e9
    spatial_log_probs = _log_softmax(logits if dustbin_logit is None else np.concatenate([logits, [float(dustbin_logit)]]))
    if dustbin_logit is None:
        log_probs = spatial_log_probs
        dustbin_probability = 0.0
    else:
        log_probs = spatial_log_probs[:-1]
        dustbin_probability = float(np.exp(spatial_log_probs[-1]))
    probabilities = np.exp(log_probs)
    prob_sum = max(float(np.sum(probabilities)), 1e-12)
    spatial_probs = probabilities / prob_sum if dustbin_logit is None else probabilities
    mean = np.sum(xy * spatial_probs[:, None], axis=0) / max(float(np.sum(spatial_probs)), 1e-12)
    centered = xy - mean.reshape(1, 2)
    cov = (centered.T @ (centered * spatial_probs[:, None])) / max(float(np.sum(spatial_probs)), 1e-12)
    cov = cov + np.eye(2, dtype=np.float64) * 1e-6
    mode_idx = int(np.argmax(probabilities))
    mode_probability = float(probabilities[mode_idx])
    gt_in_window = None
    target_is_dustbin = False
    if gt_xy_px is not None:
        gt = np.asarray(gt_xy_px, dtype=np.float64).reshape(2)
        gt_in_window = bool(np.any(np.linalg.norm(xy[valid] - gt.reshape(1, 2), axis=1) <= 0.5 * step + 1e-9))
        target_is_dustbin = not gt_in_window
    return LocalLikelihoodResult(
        xy_px=xy.astype(np.float64),
        local_log_probs=log_probs.astype(np.float64),
        mean_xy_px=mean.astype(np.float64),
        cov_query_2x2=cov.astype(np.float64),
        mode_xy_px=xy[mode_idx].astype(np.float64),
        mode_probability=mode_probability,
        dustbin_probability=dustbin_probability,
        gt_in_window=gt_in_window,
        target_is_dustbin=bool(target_is_dustbin),
    )
