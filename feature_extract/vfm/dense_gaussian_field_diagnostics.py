"""Diagnostics for scene-level dense Gaussian VFM fields."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from feature_extract.vfm.gaussian_vfm_field import GaussianVFMField, GaussianVFMSource
from feature_extract.vfm.query_to_3d_matching import normalize_rows
from feature_extract.vfm.semidense_anchor_map import SemiDenseAnchorMap


def encode_feature_map_with_selector(
    feature_map: np.ndarray,
    selector,
    device: str = "cpu",
    batch_size: int = 65536,
) -> np.ndarray:
    """Encode a CxHxW raw feature map with a row-wise selector, preserving HxW."""

    values = np.asarray(feature_map, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    channels, height, width = values.shape
    rows = values.reshape(channels, height * width).T
    encoded = selector.encode_rows(rows, device=device, batch_size=int(batch_size))
    encoded = np.asarray(encoded, dtype=np.float32)
    if encoded.ndim != 2 or encoded.shape[0] != height * width:
        raise ValueError("selector.encode_rows must return shape (H*W, D)")
    return encoded.T.reshape(encoded.shape[1], height, width).astype(np.float32, copy=False)


def gaussian_field_coverage_stats(
    source: GaussianVFMSource,
    field: GaussianVFMField,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Summarize how much of the Gaussian map has feature-bearing descriptors."""

    source_count = int(source.xyz.shape[0])
    feature_count = int(len(field))
    support = np.asarray(field.support_counts, dtype=np.float32)
    distances = np.asarray(field.mean_distances, dtype=np.float32)
    stats: dict[str, object] = {
        "source_gaussian_count": source_count,
        "feature_bearing_gaussian_count": feature_count,
        "feature_bearing_fraction": 0.0 if source_count == 0 else float(feature_count / source_count),
        "feature_dim": int(field.feature_dim),
        "mean_samples": 0.0 if support.size == 0 else float(np.mean(support)),
        "median_samples": 0.0 if support.size == 0 else float(np.median(support)),
        "mean_assignment_distance": 0.0 if distances.size == 0 else float(np.mean(distances)),
        "median_assignment_distance": 0.0 if distances.size == 0 else float(np.median(distances)),
    }
    if extra:
        stats.update(dict(extra))
    return stats


def gaussian_field_to_semidense_anchor_map(field: GaussianVFMField) -> SemiDenseAnchorMap:
    """Expose a feature-bearing Gaussian VFM field as a semi-dense anchor map."""

    count = int(len(field))
    features = np.asarray(field.features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("field.features must have shape (N, C)")
    normalized_features, _valid = normalize_rows(features)
    support = np.asarray(field.support_counts, dtype=np.float32).reshape(-1)
    distances = np.asarray(field.mean_distances, dtype=np.float32).reshape(-1)
    opacity = np.asarray(field.opacity, dtype=np.float32).reshape(-1)

    support_scale = float(np.percentile(support, 90.0)) if support.size else 1.0
    if support_scale <= 1e-6:
        support_scale = 1.0
    distance_scale = float(np.percentile(distances[distances > 0.0], 90.0)) if np.any(distances > 0.0) else 1.0
    if distance_scale <= 1e-6:
        distance_scale = 1.0
    support_score = np.clip(support / support_scale, 0.0, 1.0)
    distance_score = 1.0 / (1.0 + np.maximum(distances, 0.0) / distance_scale)
    opacity_score = np.clip(opacity, 0.0, 1.0)
    quality = np.clip(support_score * distance_score * opacity_score, 0.0, 1.0).astype(np.float32)

    return SemiDenseAnchorMap(
        anchor_ids=np.arange(count, dtype=np.int64),
        xyz=np.asarray(field.xyz, dtype=np.float64),
        features=normalized_features.astype(np.float32, copy=False),
        source_types=np.asarray(["gaussian_ray"] * count, dtype=str),
        source_track_ids=np.asarray(field.nearest_track_ids, dtype=np.int64),
        source_gaussian_indices=np.asarray(field.gaussian_indices, dtype=np.int64),
        support_counts=np.asarray(field.support_counts, dtype=np.int64),
        mean_distances=np.asarray(field.mean_distances, dtype=np.float32),
        feature_variances=np.zeros((count,), dtype=np.float32),
        observation_counts=np.asarray(field.support_counts, dtype=np.int64),
        visibility_counts=np.asarray(field.support_counts, dtype=np.int64),
        quality_scores=quality,
        opacity=np.asarray(field.opacity, dtype=np.float32),
        scale=np.asarray(field.scale, dtype=np.float32),
        observation_image_ids=tuple(() for _ in range(count)),
        metadata={
            "stage": "gaussian_field_to_semidense_anchor_map",
            "source_field_metadata": dict(field.metadata or {}),
        },
    )


def visibility_overlay_rgb(
    image_rgb: np.ndarray,
    visibility_mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a visibility mask over an RGB image for camera-view diagnostics."""

    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must have shape (H, W, 3)")
    mask = np.asarray(visibility_mask, dtype=bool)
    if mask.shape != image.shape[:2]:
        raise ValueError("visibility_mask must have shape (H, W)")

    alpha_value = float(np.clip(alpha, 0.0, 1.0))
    base = image.astype(np.float32, copy=True)
    if image.dtype.kind == "f" and np.nanmax(base) <= 1.0:
        base *= 255.0

    tint = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    base[mask] = (1.0 - alpha_value) * base[mask] + alpha_value * tint
    return np.clip(np.rint(base), 0, 255).astype(np.uint8)


def pca_feature_rgb(feature_map: np.ndarray, visibility_mask: np.ndarray) -> np.ndarray:
    """Convert visible CxHxW feature pixels to a PCA-colored RGB diagnostic."""

    features = np.asarray(feature_map, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    channels, height, width = features.shape
    mask = np.asarray(visibility_mask, dtype=bool)
    if mask.shape != (height, width):
        raise ValueError("visibility_mask must have shape (H, W)")

    rgb = np.zeros((height, width, 3), dtype=np.float32)
    if not np.any(mask):
        return np.zeros((height, width, 3), dtype=np.uint8)

    pixels = features.reshape(channels, height * width).T
    visible = pixels[mask.reshape(-1)]
    if visible.shape[0] == 1:
        projected = np.zeros((1, 3), dtype=np.float32)
        projected[0, : min(3, channels)] = visible[0, : min(3, channels)]
    else:
        centered = visible - visible.mean(axis=0, keepdims=True)
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[: min(3, vt.shape[0])].T
        projected = centered @ basis
        if projected.shape[1] < 3:
            projected = np.pad(projected, ((0, 0), (0, 3 - projected.shape[1])), mode="constant")
    lo = np.percentile(projected, 1.0, axis=0, keepdims=True)
    hi = np.percentile(projected, 99.0, axis=0, keepdims=True)
    projected = (projected - lo) / np.maximum(hi - lo, 1e-6)
    projected = np.clip(projected, 0.0, 1.0)
    rgb.reshape(-1, 3)[mask.reshape(-1)] = projected[:, :3].astype(np.float32, copy=False)
    return np.asarray(np.rint(rgb * 255.0), dtype=np.uint8)
