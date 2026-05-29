"""Non-learned feature compression baselines for Stage C0."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from feature_extract.vfm.map_lifting import SelectedTrackFeatureBank, TrackFeature


_EPS = 1e-8


@dataclass(frozen=True)
class FeatureCompressionTransform:
    method: str
    input_dim: int
    output_dim: int
    mean: np.ndarray
    matrix: np.ndarray | None = None
    selected_channels: np.ndarray | None = None
    channel_scores: np.ndarray | None = None
    l2_normalize: bool = False

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32).reshape(-1)
        if mean.shape[0] != int(self.input_dim):
            raise ValueError("mean must have shape (input_dim,)")
        matrix = None if self.matrix is None else np.asarray(self.matrix, dtype=np.float32)
        selected = None if self.selected_channels is None else np.asarray(self.selected_channels, dtype=np.int64).reshape(-1)
        scores = None if self.channel_scores is None else np.asarray(self.channel_scores, dtype=np.float32).reshape(-1)
        if matrix is not None and matrix.shape != (int(self.input_dim), int(self.output_dim)):
            raise ValueError("matrix must have shape (input_dim, output_dim)")
        if selected is not None and selected.shape[0] != int(self.output_dim):
            raise ValueError("selected_channels must have shape (output_dim,)")
        if scores is not None and scores.shape[0] != int(self.input_dim):
            raise ValueError("channel_scores must have shape (input_dim,)")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "selected_channels", selected)
        object.__setattr__(self, "channel_scores", scores)

    def apply_rows(self, features: np.ndarray) -> np.ndarray:
        rows = np.asarray(features, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != int(self.input_dim):
            raise ValueError("features must have shape (N, input_dim)")
        if self.selected_channels is not None:
            projected = rows[:, self.selected_channels]
        elif self.matrix is not None:
            centered = rows - self.mean.reshape(1, -1)
            projected = centered @ self.matrix
        else:
            projected = rows.copy()
        projected = projected.astype(np.float32, copy=False)
        if self.l2_normalize:
            projected = _l2_normalize_rows(projected)
        return projected

    def apply_variances(self, variances: np.ndarray) -> np.ndarray:
        rows = np.asarray(variances, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != int(self.input_dim):
            raise ValueError("variances must have shape (N, input_dim)")
        if self.selected_channels is not None:
            return rows[:, self.selected_channels].astype(np.float32, copy=False)
        if self.matrix is not None:
            return (rows @ np.square(self.matrix)).astype(np.float32, copy=False)
        return rows.copy()

    def to_npz(self, path: Path, metadata: Mapping[str, object] | None = None) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "vfm_feature_compression_transform_v1",
            "method": self.method,
            "input_dim": int(self.input_dim),
            "output_dim": int(self.output_dim),
            "l2_normalize": bool(self.l2_normalize),
            "metadata": dict(metadata or {}),
        }
        arrays = {
            "metadata": np.asarray(json.dumps(payload, sort_keys=True)),
            "mean": self.mean.astype(np.float32),
            "matrix": np.zeros((0, 0), dtype=np.float32) if self.matrix is None else self.matrix.astype(np.float32),
            "selected_channels": np.zeros((0,), dtype=np.int64)
            if self.selected_channels is None
            else self.selected_channels.astype(np.int64),
            "channel_scores": np.zeros((0,), dtype=np.float32)
            if self.channel_scores is None
            else self.channel_scores.astype(np.float32),
        }
        np.savez_compressed(output, **arrays)

    @classmethod
    def from_npz(cls, path: Path) -> "FeatureCompressionTransform":
        with np.load(Path(path)) as data:
            payload = json.loads(str(data["metadata"].item()))
            if payload.get("format") != "vfm_feature_compression_transform_v1":
                raise ValueError(f"unsupported feature compression transform format in {path}")
            matrix = np.asarray(data["matrix"], dtype=np.float32)
            selected = np.asarray(data["selected_channels"], dtype=np.int64)
            scores = np.asarray(data["channel_scores"], dtype=np.float32)
            return cls(
                method=str(payload["method"]),
                input_dim=int(payload["input_dim"]),
                output_dim=int(payload["output_dim"]),
                mean=np.asarray(data["mean"], dtype=np.float32),
                matrix=None if matrix.size == 0 else matrix,
                selected_channels=None if selected.size == 0 else selected,
                channel_scores=None if scores.size == 0 else scores,
                l2_normalize=bool(payload.get("l2_normalize", False)),
            )


def _l2_normalize_rows(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return (features / np.maximum(norms, _EPS)).astype(np.float32, copy=False)


def _sample_rows(features: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    rows = np.asarray(features, dtype=np.float32)
    if rows.ndim != 2:
        raise ValueError("features must have shape (N, C)")
    if max_samples <= 0 or rows.shape[0] <= max_samples:
        return rows
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(rows.shape[0], size=int(max_samples), replace=False)
    return rows[np.sort(indices)]


def _stable_topk(scores: np.ndarray, output_dim: int) -> np.ndarray:
    indexed = np.lexsort((np.arange(scores.size), -np.asarray(scores, dtype=np.float64)))
    return indexed[:output_dim].astype(np.int64)


def _fit_pca_matrix(features: np.ndarray, output_dim: int) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0).astype(np.float32)
    centered = features - mean.reshape(1, -1)
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64, copy=False))
    order = np.argsort(eigvals)[::-1][:output_dim]
    matrix = eigvecs[:, order].astype(np.float32)
    for col in range(matrix.shape[1]):
        pivot = int(np.argmax(np.abs(matrix[:, col])))
        if matrix[pivot, col] < 0:
            matrix[:, col] *= -1.0
    return mean, matrix


def _idf_scores(features: np.ndarray, threshold: float) -> np.ndarray:
    active = np.abs(features) > float(threshold)
    df = active.sum(axis=0).astype(np.float32)
    idf = np.log((features.shape[0] + 1.0) / (df + 1.0))
    return np.where(df > 0, idf, 0.0).astype(np.float32)


def _fisher_scores(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    label_values = np.asarray(labels).reshape(-1)
    if label_values.shape[0] != features.shape[0]:
        raise ValueError("labels must have shape (N,)")
    classes = np.unique(label_values)
    if classes.size < 2:
        raise ValueError("fisher selection requires at least two classes")
    global_mean = features.mean(axis=0)
    between = np.zeros((features.shape[1],), dtype=np.float64)
    within = np.zeros((features.shape[1],), dtype=np.float64)
    for cls in classes:
        subset = features[label_values == cls]
        if subset.size == 0:
            continue
        cls_mean = subset.mean(axis=0)
        between += float(subset.shape[0]) * np.square(cls_mean - global_mean)
        within += np.square(subset - cls_mean.reshape(1, -1)).sum(axis=0)
    return (between / np.maximum(within, _EPS)).astype(np.float32)


def fit_feature_compression(
    features: np.ndarray,
    method: str,
    output_dim: int,
    seed: int = 0,
    max_fit_samples: int = 0,
    labels: np.ndarray | None = None,
    idf_threshold: float = 0.0,
    l2_normalize: bool = False,
) -> FeatureCompressionTransform:
    sampled = _sample_rows(features, int(max_fit_samples), int(seed))
    if sampled.ndim != 2:
        raise ValueError("features must have shape (N, C)")
    input_dim = int(sampled.shape[1])
    if method == "identity":
        output_dim = input_dim
    if not 0 < int(output_dim) <= input_dim:
        raise ValueError("output_dim must be in (0, input_dim]")
    output_dim = int(output_dim)
    zero_mean = np.zeros((input_dim,), dtype=np.float32)

    if method == "identity":
        return FeatureCompressionTransform(method, input_dim, input_dim, zero_mean, l2_normalize=l2_normalize)
    if method == "first_channels":
        selected = np.arange(output_dim, dtype=np.int64)
        return FeatureCompressionTransform(
            method,
            input_dim,
            output_dim,
            zero_mean,
            selected_channels=selected,
            l2_normalize=l2_normalize,
        )
    if method == "random":
        rng = np.random.default_rng(int(seed))
        matrix = rng.normal(0.0, 1.0 / np.sqrt(output_dim), size=(input_dim, output_dim)).astype(np.float32)
        return FeatureCompressionTransform(method, input_dim, output_dim, zero_mean, matrix=matrix, l2_normalize=l2_normalize)
    if method == "pca":
        mean, matrix = _fit_pca_matrix(sampled, output_dim)
        return FeatureCompressionTransform(
            method,
            input_dim,
            output_dim,
            mean,
            matrix=matrix,
            l2_normalize=l2_normalize,
        )
    if method == "channel_variance":
        scores = sampled.var(axis=0).astype(np.float32)
        selected = _stable_topk(scores, output_dim)
        return FeatureCompressionTransform(
            method,
            input_dim,
            output_dim,
            zero_mean,
            selected_channels=selected,
            channel_scores=scores,
            l2_normalize=l2_normalize,
        )
    if method == "idf":
        scores = _idf_scores(sampled, threshold=float(idf_threshold))
        selected = _stable_topk(scores, output_dim)
        return FeatureCompressionTransform(
            method,
            input_dim,
            output_dim,
            zero_mean,
            selected_channels=selected,
            channel_scores=scores,
            l2_normalize=l2_normalize,
        )
    if method == "fisher":
        if labels is None:
            raise ValueError("fisher selection requires labels")
        sampled_labels = np.asarray(labels)
        if max_fit_samples > 0 and np.asarray(features).shape[0] > int(max_fit_samples):
            raise ValueError("fisher selection does not support max_fit_samples without sampled labels")
        scores = _fisher_scores(sampled, sampled_labels)
        selected = _stable_topk(scores, output_dim)
        return FeatureCompressionTransform(
            method,
            input_dim,
            output_dim,
            zero_mean,
            selected_channels=selected,
            channel_scores=scores,
            l2_normalize=l2_normalize,
        )
    raise ValueError(f"unsupported compression method: {method}")


def slice_feature_compression(
    transform: FeatureCompressionTransform,
    output_dim: int,
) -> FeatureCompressionTransform:
    if not 0 < int(output_dim) <= int(transform.output_dim):
        raise ValueError("output_dim must be in (0, transform.output_dim]")
    output_dim = int(output_dim)
    if output_dim == int(transform.output_dim):
        return transform
    matrix = None if transform.matrix is None else transform.matrix[:, :output_dim]
    selected = None if transform.selected_channels is None else transform.selected_channels[:output_dim]
    return FeatureCompressionTransform(
        method=transform.method,
        input_dim=transform.input_dim,
        output_dim=output_dim,
        mean=transform.mean,
        matrix=matrix,
        selected_channels=selected,
        channel_scores=transform.channel_scores,
        l2_normalize=transform.l2_normalize,
    )


def apply_feature_compression_to_channel_first(
    feature_map: np.ndarray,
    transform: FeatureCompressionTransform,
) -> np.ndarray:
    values = np.asarray(feature_map, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != int(transform.input_dim):
        raise ValueError("feature_map must have shape (input_dim, H, W)")
    channels, height, width = values.shape
    rows = values.reshape(channels, height * width).T
    transformed = transform.apply_rows(rows)
    return transformed.T.reshape(transform.output_dim, height, width).astype(np.float32, copy=False)


def transform_track_feature_bank(
    bank: SelectedTrackFeatureBank,
    transform: FeatureCompressionTransform,
) -> SelectedTrackFeatureBank:
    if int(bank.feature_dim) != int(transform.input_dim):
        raise ValueError("track bank feature_dim does not match transform input_dim")
    if not bank.tracks:
        return SelectedTrackFeatureBank(tracks={}, feature_dim=transform.output_dim)
    track_ids = sorted(bank.tracks)
    means = np.stack([bank.tracks[track_id].mean_feature for track_id in track_ids], axis=0).astype(np.float32)
    variances = np.stack([bank.tracks[track_id].variance for track_id in track_ids], axis=0).astype(np.float32)
    transformed_means = transform.apply_rows(means)
    transformed_variances = transform.apply_variances(variances)
    tracks = {}
    for idx, track_id in enumerate(track_ids):
        source = bank.tracks[track_id]
        tracks[int(track_id)] = TrackFeature(
            track_id=int(track_id),
            mean_feature=transformed_means[idx],
            variance=transformed_variances[idx],
            observation_count=int(source.observation_count),
            mean_utility=float(source.mean_utility),
            observation_image_ids=tuple(source.observation_image_ids),
        )
    return SelectedTrackFeatureBank(tracks=tracks, feature_dim=transform.output_dim)
