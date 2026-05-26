"""Control experiments for feature selection claims."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


METADATA_SCORE_WHITELIST = {
    "candidate_rank",
    "retrieval_score",
    "pnp_inliers",
    "reprojection_median",
    "score_margin",
    "candidate_prior",
}


def shuffle_features(features: np.ndarray, axis: int, seed: int) -> np.ndarray:
    """Shuffle features along one axis with deterministic RNG."""

    array = np.asarray(features).copy()
    rng = np.random.default_rng(seed)
    indices = np.arange(array.shape[axis])
    rng.shuffle(indices)
    return np.take(array, indices, axis=axis)


def random_channel_projection(features: np.ndarray, output_dim: int, seed: int) -> np.ndarray:
    """Project channel-first features with a deterministic Gaussian matrix."""

    array = np.asarray(features, dtype=np.float32)
    if array.ndim < 2:
        raise ValueError("features must be channel-first with at least 2 dimensions")
    channels = array.shape[0]
    if not 0 < output_dim <= channels:
        raise ValueError("output_dim must be in (0, channels]")
    rng = np.random.default_rng(seed)
    projection = rng.normal(0.0, 1.0 / np.sqrt(output_dim), size=(output_dim, channels))
    flat = array.reshape(channels, -1)
    return (projection @ flat).reshape(output_dim, *array.shape[1:]).astype(np.float32)


def pca_channel_projection(features: np.ndarray, output_dim: int) -> np.ndarray:
    """Project channel-first features to principal components using SVD."""

    array = np.asarray(features, dtype=np.float32)
    if array.ndim < 2:
        raise ValueError("features must be channel-first with at least 2 dimensions")
    channels = array.shape[0]
    if not 0 < output_dim <= channels:
        raise ValueError("output_dim must be in (0, channels]")
    flat = array.reshape(channels, -1)
    centered = flat - flat.mean(axis=1, keepdims=True)
    u, _, _ = np.linalg.svd(centered, full_matrices=False)
    projected = u[:, :output_dim].T @ centered
    return projected.reshape(output_dim, *array.shape[1:]).astype(np.float32)


def mask_channels_by_utility(
    features: np.ndarray,
    utility: np.ndarray,
    fraction: float,
    remove: str,
) -> np.ndarray:
    """Zero high- or low-utility channels for counterfactual controls."""

    array = np.asarray(features).copy()
    channel_utility = np.asarray(utility, dtype=np.float64).reshape(-1)
    if array.shape[0] != channel_utility.size:
        raise ValueError("first feature dimension must match utility length")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    count = max(1, int(round(channel_utility.size * fraction)))
    order = np.argsort(channel_utility, kind="mergesort")
    if remove == "high":
        selected = order[-count:]
    elif remove == "low":
        selected = order[:count]
    else:
        raise ValueError("remove must be 'high' or 'low'")
    array[selected] = 0
    return array


def metadata_only_scores(
    metadata_rows: Iterable[Mapping[str, float]],
    weights: Mapping[str, float],
) -> np.ndarray:
    """Score candidates using only whitelisted non-oracle metadata fields."""

    for field in weights:
        if field not in METADATA_SCORE_WHITELIST:
            raise ValueError(f"metadata field '{field}' is not whitelisted")
    scores = []
    for row in metadata_rows:
        total = 0.0
        for field, weight in weights.items():
            total += float(weight) * float(row.get(field, 0.0))
        scores.append(total)
    return np.asarray(scores, dtype=np.float64)
