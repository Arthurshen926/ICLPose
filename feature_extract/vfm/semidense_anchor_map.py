"""Semi-dense reliable anchor maps for patch-level VFM localization.

Stage E is intentionally a side path: it builds an expanded map support from
existing sparse SfM landmarks and nearby reliable Gaussian centers, without
changing the canonical sparse matching pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree

from feature_extract.vfm.gaussian_vfm_field import GaussianVFMSource
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, normalize_rows


@dataclass(frozen=True)
class SemiDenseAnchorConfig:
    max_distance: float = 0.05
    k_neighbors: int = 2
    min_support: int = 1
    min_opacity: float = 0.0
    max_gaussian_scale: float | None = None
    max_landmark_variance: float | None = None
    min_landmark_observations: int = 2
    include_sparse: bool = True
    l2_normalize_features: bool = True
    gaussian_track_id_offset: int = 100_000_000

    def __post_init__(self) -> None:
        if float(self.max_distance) <= 0.0:
            raise ValueError("max_distance must be positive")
        if int(self.k_neighbors) <= 0:
            raise ValueError("k_neighbors must be positive")
        if int(self.min_support) <= 0:
            raise ValueError("min_support must be positive")
        if float(self.min_opacity) < 0.0:
            raise ValueError("min_opacity must be non-negative")
        if self.max_gaussian_scale is not None and float(self.max_gaussian_scale) <= 0.0:
            raise ValueError("max_gaussian_scale must be positive")
        if int(self.min_landmark_observations) <= 0:
            raise ValueError("min_landmark_observations must be positive")
        if int(self.gaussian_track_id_offset) <= 0:
            raise ValueError("gaussian_track_id_offset must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_distance": float(self.max_distance),
            "k_neighbors": int(self.k_neighbors),
            "min_support": int(self.min_support),
            "min_opacity": float(self.min_opacity),
            "max_gaussian_scale": None if self.max_gaussian_scale is None else float(self.max_gaussian_scale),
            "max_landmark_variance": None
            if self.max_landmark_variance is None
            else float(self.max_landmark_variance),
            "min_landmark_observations": int(self.min_landmark_observations),
            "include_sparse": bool(self.include_sparse),
            "l2_normalize_features": bool(self.l2_normalize_features),
            "gaussian_track_id_offset": int(self.gaussian_track_id_offset),
        }


@dataclass(frozen=True)
class SemiDenseAnchorMap:
    anchor_ids: np.ndarray
    xyz: np.ndarray
    features: np.ndarray
    source_types: np.ndarray
    source_track_ids: np.ndarray
    source_gaussian_indices: np.ndarray
    support_counts: np.ndarray
    mean_distances: np.ndarray
    feature_variances: np.ndarray
    observation_counts: np.ndarray
    visibility_counts: np.ndarray
    quality_scores: np.ndarray
    opacity: np.ndarray
    scale: np.ndarray
    observation_image_ids: tuple[tuple[str, ...], ...] | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        anchor_ids = np.asarray(self.anchor_ids, dtype=np.int64).reshape(-1)
        xyz = np.asarray(self.xyz, dtype=np.float64)
        features = np.asarray(self.features, dtype=np.float32)
        row_count = int(anchor_ids.shape[0])
        if xyz.shape != (row_count, 3):
            raise ValueError("xyz must have shape (N, 3)")
        if features.ndim != 2 or features.shape[0] != row_count:
            raise ValueError("features must have shape (N, C)")
        object.__setattr__(self, "anchor_ids", anchor_ids)
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "source_types", np.asarray(self.source_types, dtype=str).reshape(row_count))
        for name in (
            "source_track_ids",
            "source_gaussian_indices",
            "support_counts",
            "observation_counts",
            "visibility_counts",
        ):
            value = np.asarray(getattr(self, name), dtype=np.int64).reshape(-1)
            if value.shape != (row_count,):
                raise ValueError(f"{name} must have shape (N,)")
            object.__setattr__(self, name, value)
        for name in ("mean_distances", "feature_variances", "quality_scores", "opacity", "scale"):
            value = np.asarray(getattr(self, name), dtype=np.float32).reshape(-1)
            if value.shape != (row_count,):
                raise ValueError(f"{name} must have shape (N,)")
            object.__setattr__(self, name, value)
        if self.observation_image_ids is None:
            observation_image_ids = tuple(() for _ in range(row_count))
        else:
            observation_image_ids = tuple(
                tuple(str(image_id) for image_id in image_ids)
                for image_ids in self.observation_image_ids
            )
            if len(observation_image_ids) != row_count:
                raise ValueError("observation_image_ids must have length N")
        object.__setattr__(self, "observation_image_ids", observation_image_ids)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1]) if self.features.ndim == 2 else 0

    def __len__(self) -> int:
        return int(self.anchor_ids.shape[0])

    def to_landmark_index(self) -> LandmarkMapIndex:
        track_ids = np.asarray(self.source_track_ids, dtype=np.int64).copy()
        gaussian_rows = np.flatnonzero(np.asarray(self.source_types) != "sfm")
        if gaussian_rows.size:
            track_ids[gaussian_rows] = -100_000_000 - np.arange(gaussian_rows.size, dtype=np.int64)
        return LandmarkMapIndex(
            track_ids=track_ids,
            xyz=self.xyz,
            features=self.features,
            mean_variances=self.feature_variances,
            observation_counts=self.observation_counts,
            observation_image_ids=self.observation_image_ids,
            reprojection_errors=self.mean_distances,
        )

    def save_npz(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            anchor_ids=self.anchor_ids.astype(np.int64, copy=False),
            xyz=self.xyz.astype(np.float32, copy=False),
            features=self.features.astype(np.float32, copy=False),
            source_types=self.source_types.astype(str, copy=False),
            source_track_ids=self.source_track_ids.astype(np.int64, copy=False),
            source_gaussian_indices=self.source_gaussian_indices.astype(np.int64, copy=False),
            support_counts=self.support_counts.astype(np.int64, copy=False),
            mean_distances=self.mean_distances.astype(np.float32, copy=False),
            feature_variances=self.feature_variances.astype(np.float32, copy=False),
            observation_counts=self.observation_counts.astype(np.int64, copy=False),
            visibility_counts=self.visibility_counts.astype(np.int64, copy=False),
            quality_scores=self.quality_scores.astype(np.float32, copy=False),
            opacity=self.opacity.astype(np.float32, copy=False),
            scale=self.scale.astype(np.float32, copy=False),
            observation_image_ids=np.asarray(
                [json.dumps(list(image_ids), sort_keys=True) for image_ids in self.observation_image_ids],
                dtype=str,
            ),
            metadata_json=np.asarray(json.dumps(dict(self.metadata or {}), sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "SemiDenseAnchorMap":
        with np.load(Path(path), allow_pickle=True) as data:
            metadata = {}
            if "metadata_json" in data:
                metadata = json.loads(str(data["metadata_json"].tolist()))
            if "observation_image_ids" in data:
                observation_image_ids = tuple(
                    tuple(str(image_id) for image_id in json.loads(str(item)))
                    for item in data["observation_image_ids"].tolist()
                )
            else:
                observation_image_ids = None
            return cls(
                anchor_ids=np.asarray(data["anchor_ids"], dtype=np.int64),
                xyz=np.asarray(data["xyz"], dtype=np.float64),
                features=np.asarray(data["features"], dtype=np.float32),
                source_types=np.asarray(data["source_types"], dtype=str),
                source_track_ids=np.asarray(data["source_track_ids"], dtype=np.int64),
                source_gaussian_indices=np.asarray(data["source_gaussian_indices"], dtype=np.int64),
                support_counts=np.asarray(data["support_counts"], dtype=np.int64),
                mean_distances=np.asarray(data["mean_distances"], dtype=np.float32),
                feature_variances=np.asarray(data["feature_variances"], dtype=np.float32),
                observation_counts=np.asarray(data["observation_counts"], dtype=np.int64),
                visibility_counts=np.asarray(data["visibility_counts"], dtype=np.int64),
                quality_scores=np.asarray(data["quality_scores"], dtype=np.float32),
                opacity=np.asarray(data["opacity"], dtype=np.float32),
                scale=np.asarray(data["scale"], dtype=np.float32),
                observation_image_ids=observation_image_ids,
                metadata=metadata,
            )


def _filtered_landmarks(landmarks: LandmarkMapIndex, config: SemiDenseAnchorConfig) -> LandmarkMapIndex:
    if len(landmarks) == 0:
        return landmarks
    mask = landmarks.observation_counts >= int(config.min_landmark_observations)
    if config.max_landmark_variance is not None:
        mask &= landmarks.mean_variances <= float(config.max_landmark_variance)
    return landmarks.subset(mask)


def _quality_from_components(
    support_counts: np.ndarray,
    distances: np.ndarray,
    variances: np.ndarray,
    opacity: np.ndarray,
    config: SemiDenseAnchorConfig,
) -> np.ndarray:
    support_score = np.clip(np.asarray(support_counts, dtype=np.float32) / max(float(config.k_neighbors), 1.0), 0.0, 1.0)
    distance_score = 1.0 - np.clip(np.asarray(distances, dtype=np.float32) / max(float(config.max_distance), 1e-6), 0.0, 1.0)
    variance = np.maximum(np.asarray(variances, dtype=np.float32), 0.0)
    variance_scale = float(np.percentile(variance, 90.0)) if variance.size else 1.0
    if variance_scale <= 1e-8:
        variance_scale = 1.0
    variance_score = 1.0 / (1.0 + variance / variance_scale)
    opacity_score = np.clip(np.asarray(opacity, dtype=np.float32), 0.0, 1.0)
    return np.clip(support_score * distance_score * variance_score * opacity_score, 0.0, 1.0).astype(np.float32)


def build_sfm_guided_semidense_anchor_map(
    landmarks: LandmarkMapIndex,
    gaussians: GaussianVFMSource,
    config: SemiDenseAnchorConfig | None = None,
) -> SemiDenseAnchorMap:
    """Expand sparse SfM VFM landmarks to nearby reliable Gaussian anchors."""

    cfg = config or SemiDenseAnchorConfig()
    filtered = _filtered_landmarks(landmarks, cfg)
    feature_dim = int(landmarks.feature_dim)
    rows: list[dict[str, object]] = []
    if bool(cfg.include_sparse):
        for idx in range(len(filtered)):
            rows.append(
                {
                    "xyz": filtered.xyz[idx],
                    "feature": filtered.features[idx],
                    "source_type": "sfm",
                    "source_track_id": int(filtered.track_ids[idx]),
                    "source_gaussian_index": -1,
                    "support_count": int(filtered.observation_counts[idx]),
                    "mean_distance": 0.0,
                    "feature_variance": float(filtered.mean_variances[idx]),
                    "observation_count": int(filtered.observation_counts[idx]),
                    "visibility_count": len(filtered.observation_image_ids[idx]),
                    "quality": 1.0,
                    "opacity": 1.0,
                    "scale": 0.0,
                    "observation_image_ids": tuple(filtered.observation_image_ids[idx]),
                }
            )
    if len(filtered) == 0 or gaussians.xyz.shape[0] == 0:
        return _rows_to_anchor_map(rows, feature_dim, cfg)

    candidate_mask = gaussians.opacity >= float(cfg.min_opacity)
    if cfg.max_gaussian_scale is not None:
        candidate_mask &= gaussians.scale <= float(cfg.max_gaussian_scale)
    candidate_indices = np.flatnonzero(candidate_mask)
    if candidate_indices.size == 0:
        return _rows_to_anchor_map(rows, feature_dim, cfg)

    tree = cKDTree(filtered.xyz)
    k = min(int(cfg.k_neighbors), len(filtered))
    distances, indices = tree.query(gaussians.xyz[candidate_indices], k=k, distance_upper_bound=float(cfg.max_distance))
    distances = np.asarray(distances)
    indices = np.asarray(indices)
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    valid = np.isfinite(distances) & (indices >= 0) & (indices < len(filtered))
    support_counts = valid.sum(axis=1)
    keep_local_rows = np.flatnonzero(support_counts >= int(cfg.min_support))
    for local_row in keep_local_rows:
        row_valid = valid[local_row]
        landmark_indices = indices[local_row, row_valid].astype(np.int64)
        row_distances = distances[local_row, row_valid].astype(np.float32)
        weights = 1.0 / np.maximum(row_distances, 1e-6)
        weights = weights / max(float(np.sum(weights)), 1e-12)
        support_features = filtered.features[landmark_indices].astype(np.float32)
        feature = np.sum(support_features * weights[:, None], axis=0).astype(np.float32)
        if bool(cfg.l2_normalize_features):
            feature, _valid_norm = normalize_rows(feature.reshape(1, -1))
            feature = feature.reshape(-1)
        feature_variance = float(np.mean(np.var(support_features, axis=0))) if support_features.shape[0] > 1 else float(
            filtered.mean_variances[int(landmark_indices[0])]
        )
        nearest = int(landmark_indices[int(np.argmin(row_distances))])
        support_image_ids = tuple(
            sorted(
                {
                    str(image_id)
                    for landmark_idx in landmark_indices.tolist()
                    for image_id in filtered.observation_image_ids[int(landmark_idx)]
                }
            )
        )
        source_row = int(candidate_indices[int(local_row)])
        quality = _quality_from_components(
            np.asarray([support_counts[local_row]], dtype=np.int64),
            np.asarray([float(np.mean(row_distances))], dtype=np.float32),
            np.asarray([feature_variance], dtype=np.float32),
            np.asarray([gaussians.opacity[source_row]], dtype=np.float32),
            cfg,
        )[0]
        rows.append(
            {
                "xyz": gaussians.xyz[source_row],
                "feature": feature,
                "source_type": "gaussian_near_sfm",
                "source_track_id": int(filtered.track_ids[nearest]),
                "source_gaussian_index": int(gaussians.gaussian_indices[source_row]),
                "support_count": int(support_counts[local_row]),
                "mean_distance": float(np.mean(row_distances)),
                "feature_variance": feature_variance,
                "observation_count": int(support_counts[local_row]),
                "visibility_count": int(np.max(filtered.observation_counts[landmark_indices])),
                "quality": float(quality),
                "opacity": float(gaussians.opacity[source_row]),
                "scale": float(gaussians.scale[source_row]),
                "observation_image_ids": support_image_ids,
            }
        )
    return _rows_to_anchor_map(rows, feature_dim, cfg)


def _rows_to_anchor_map(rows: list[dict[str, object]], feature_dim: int, config: SemiDenseAnchorConfig) -> SemiDenseAnchorMap:
    if not rows:
        return SemiDenseAnchorMap(
            anchor_ids=np.zeros((0,), dtype=np.int64),
            xyz=np.zeros((0, 3), dtype=np.float64),
            features=np.zeros((0, feature_dim), dtype=np.float32),
            source_types=np.zeros((0,), dtype=str),
            source_track_ids=np.zeros((0,), dtype=np.int64),
            source_gaussian_indices=np.zeros((0,), dtype=np.int64),
            support_counts=np.zeros((0,), dtype=np.int64),
            mean_distances=np.zeros((0,), dtype=np.float32),
            feature_variances=np.zeros((0,), dtype=np.float32),
            observation_counts=np.zeros((0,), dtype=np.int64),
            visibility_counts=np.zeros((0,), dtype=np.int64),
            quality_scores=np.zeros((0,), dtype=np.float32),
            opacity=np.zeros((0,), dtype=np.float32),
            scale=np.zeros((0,), dtype=np.float32),
            observation_image_ids=(),
            metadata={"config": config.to_dict()},
        )
    features = np.stack([np.asarray(row["feature"], dtype=np.float32).reshape(-1) for row in rows], axis=0)
    if bool(config.l2_normalize_features):
        features, _valid = normalize_rows(features)
    return SemiDenseAnchorMap(
        anchor_ids=np.arange(len(rows), dtype=np.int64),
        xyz=np.stack([np.asarray(row["xyz"], dtype=np.float64).reshape(3) for row in rows], axis=0),
        features=features.astype(np.float32, copy=False),
        source_types=np.asarray([str(row["source_type"]) for row in rows], dtype=str),
        source_track_ids=np.asarray([int(row["source_track_id"]) for row in rows], dtype=np.int64),
        source_gaussian_indices=np.asarray([int(row["source_gaussian_index"]) for row in rows], dtype=np.int64),
        support_counts=np.asarray([int(row["support_count"]) for row in rows], dtype=np.int64),
        mean_distances=np.asarray([float(row["mean_distance"]) for row in rows], dtype=np.float32),
        feature_variances=np.asarray([float(row["feature_variance"]) for row in rows], dtype=np.float32),
        observation_counts=np.asarray([int(row["observation_count"]) for row in rows], dtype=np.int64),
        visibility_counts=np.asarray([int(row["visibility_count"]) for row in rows], dtype=np.int64),
        quality_scores=np.asarray([float(row["quality"]) for row in rows], dtype=np.float32),
        opacity=np.asarray([float(row["opacity"]) for row in rows], dtype=np.float32),
        scale=np.asarray([float(row["scale"]) for row in rows], dtype=np.float32),
        observation_image_ids=tuple(tuple(str(item) for item in row.get("observation_image_ids", ())) for row in rows),
        metadata={"config": config.to_dict()},
    )


def semidense_anchor_map_stats(
    anchor_map: SemiDenseAnchorMap,
    sparse_landmark_count: int,
    source_gaussian_count: int,
) -> dict[str, float | int]:
    source_types = np.asarray(anchor_map.source_types, dtype=str)
    gaussian_count = int(np.sum(source_types == "gaussian_near_sfm"))
    sfm_count = int(np.sum(source_types == "sfm"))
    return {
        "anchor_count": int(len(anchor_map)),
        "sfm_count": sfm_count,
        "gaussian_near_sfm_count": gaussian_count,
        "sparse_landmark_count": int(sparse_landmark_count),
        "source_gaussian_count": int(source_gaussian_count),
        "expansion_ratio_vs_sparse": float(len(anchor_map) / max(int(sparse_landmark_count), 1)),
        "gaussian_keep_fraction": float(gaussian_count / max(int(source_gaussian_count), 1)),
        "feature_dim": int(anchor_map.feature_dim),
        "mean_quality": 0.0 if len(anchor_map) == 0 else float(np.mean(anchor_map.quality_scores)),
        "mean_feature_variance": 0.0 if len(anchor_map) == 0 else float(np.mean(anchor_map.feature_variances)),
        "mean_support_count": 0.0 if len(anchor_map) == 0 else float(np.mean(anchor_map.support_counts)),
        "mean_distance": 0.0 if len(anchor_map) == 0 else float(np.mean(anchor_map.mean_distances)),
        "mean_opacity": 0.0 if len(anchor_map) == 0 else float(np.mean(anchor_map.opacity)),
    }


def filter_semidense_by_source_visibility(
    anchor_map: SemiDenseAnchorMap,
    visibility_index,
    reference_images,
) -> tuple[LandmarkMapIndex, dict[str, float | int]]:
    """Filter semi-dense anchors by visibility of their supporting SfM tracks."""

    full_visible = visibility_index.visible_tracks(reference_images)
    if not full_visible:
        return anchor_map.to_landmark_index().subset([]), {
            "full_visible_tracks": 0,
            "bank_visible_tracks": 0,
            "bank_visibility_coverage": 0.0,
        }
    source_track_ids = np.asarray(anchor_map.source_track_ids, dtype=np.int64)
    mask = np.asarray([int(track_id) in full_visible for track_id in source_track_ids], dtype=bool)
    subset = anchor_map.to_landmark_index().subset(mask)
    return subset, {
        "full_visible_tracks": int(len(full_visible)),
        "bank_visible_tracks": int(len(subset)),
        "bank_visibility_coverage": float(len(subset) / max(len(full_visible), 1)),
    }


def _normalize_colors(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    channels = []
    for idx in range(array.shape[1]):
        channel = array[:, idx]
        low, high = np.percentile(channel, [1.0, 99.0])
        if high - low < 1e-12:
            channels.append(np.full(channel.shape, 127.0, dtype=np.float64))
        else:
            channels.append(np.clip((channel - low) / (high - low), 0.0, 1.0) * 255.0)
    color = np.stack(channels, axis=1)
    while color.shape[1] < 3:
        color = np.concatenate([color, np.zeros((color.shape[0], 1), dtype=np.float64)], axis=1)
    return np.asarray(np.round(color[:, :3]), dtype=np.uint8)


def _pca_colors(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.shape[0] < 2:
        return np.full((values.shape[0], 3), 127, dtype=np.uint8)
    centered = values - np.mean(values, axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    components = centered @ vt[: min(3, vt.shape[0])].T
    return _normalize_colors(components)


def _source_colors(source_types: np.ndarray, quality: np.ndarray) -> np.ndarray:
    colors = np.zeros((source_types.shape[0], 3), dtype=np.uint8)
    quality_u8 = np.asarray(np.clip(quality, 0.0, 1.0) * 255.0, dtype=np.uint8)
    for idx, source in enumerate(np.asarray(source_types, dtype=str).tolist()):
        if source == "sfm":
            colors[idx] = np.asarray([30, 120, 255], dtype=np.uint8)
        else:
            colors[idx] = np.asarray([255, int(quality_u8[idx]), 20], dtype=np.uint8)
    return colors


def _sample_indices(count: int, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or count <= max_points:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=int(max_points), replace=False)).astype(np.int64)


def _write_ascii_ply(
    path: Path,
    xyz: np.ndarray,
    colors: np.ndarray,
    anchor_ids: np.ndarray,
    source_track_ids: np.ndarray | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    has_source = source_track_ids is not None
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {xyz.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property int anchor_id",
    ]
    if has_source:
        lines.append("property int source_track_id")
    lines.append("end_header")
    for row, (point, color, anchor_id) in enumerate(zip(xyz, colors, anchor_ids)):
        item = (
            f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
            f"{int(color[0])} {int(color[1])} {int(color[2])} {int(anchor_id)}"
        )
        if has_source:
            item += f" {int(source_track_ids[row])}"
        lines.append(item)
    path.write_text("\n".join(lines) + "\n")


def _write_topdown_png(path: Path, sparse_xyz: np.ndarray, semidense_xyz: np.ndarray, semidense_colors: np.ndarray) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=160)
    axes[0].scatter(sparse_xyz[:, 0], sparse_xyz[:, 2], s=0.3, c=np.asarray([[0.12, 0.46, 1.0]]), linewidths=0)
    axes[0].set_title("Sparse SfM VFM Anchors")
    axes[1].scatter(
        semidense_xyz[:, 0],
        semidense_xyz[:, 2],
        s=0.25,
        c=np.asarray(semidense_colors, dtype=np.float32) / 255.0,
        linewidths=0,
    )
    axes[1].set_title("Semi-dense Reliable Anchors")
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def _project_camera_view(xyz: np.ndarray, pose_w2c: np.ndarray, camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool), np.zeros((0,), dtype=np.float64)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    camera_points = (pose[:3, :3] @ points.T + pose[:3, 3:4]).T
    depth = camera_points[:, 2]
    visible = depth > 1e-8
    safe_depth = np.where(visible, depth, 1.0)
    x = camera_points[:, 0] / safe_depth
    y = camera_points[:, 1] / safe_depth
    if int(camera.model_id) == 0:
        f, cx, cy = camera.params[:3]
        u = float(f) * x + float(cx)
        v = float(f) * y + float(cy)
    elif int(camera.model_id) == 1:
        fx, fy, cx, cy = camera.params[:4]
        u = float(fx) * x + float(cx)
        v = float(fy) * y + float(cy)
    elif int(camera.model_id) == 2:
        f, cx, cy, k = camera.params[:4]
        radial = 1.0 + float(k) * (x * x + y * y)
        u = float(f) * x * radial + float(cx)
        v = float(f) * y * radial + float(cy)
    else:
        raise ValueError(f"unsupported camera model id for camera-view visualization: {camera.model_id}")
    xy = np.stack([u, v], axis=1).astype(np.float64)
    visible &= (xy[:, 0] >= 0.0) & (xy[:, 0] <= float(camera.width - 1))
    visible &= (xy[:, 1] >= 0.0) & (xy[:, 1] <= float(camera.height - 1))
    return xy, visible, depth


def _camera_canvas(image_rgb: np.ndarray | None, width: int, height: int) -> np.ndarray:
    if image_rgb is None:
        return np.full((height, width, 3), 24, dtype=np.uint8)
    image = np.asarray(image_rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must have shape (H, W, 3)")
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for camera-view visualization") from exc
    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    return np.clip(image.astype(np.float32) * 0.55 + 18.0, 0.0, 255.0).astype(np.uint8)


def _draw_projected_points(
    canvas: np.ndarray,
    xy: np.ndarray,
    visible: np.ndarray,
    depth: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    seed: int,
    radius: int,
) -> int:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for camera-view visualization") from exc
    indices = np.flatnonzero(np.asarray(visible, dtype=bool))
    if max_points > 0 and indices.size > max_points:
        rng = np.random.default_rng(int(seed))
        indices = np.sort(rng.choice(indices, size=int(max_points), replace=False)).astype(np.int64)
    if indices.size == 0:
        return 0
    order = indices[np.argsort(depth[indices])[::-1]]
    for idx in order.tolist():
        point = xy[int(idx)]
        color = np.asarray(colors[int(idx)], dtype=np.uint8).reshape(3)
        cv2.circle(
            canvas,
            (int(round(point[0])), int(round(point[1]))),
            int(radius),
            (int(color[0]), int(color[1]), int(color[2])),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    return int(indices.size)


def write_semidense_anchor_camera_view_visualization(
    path: Path,
    sparse_index: LandmarkMapIndex,
    semidense_map: SemiDenseAnchorMap,
    pose_w2c: np.ndarray,
    camera,
    image_rgb: np.ndarray | None = None,
    max_points: int = 50_000,
    seed: int = 0,
    point_radius: int = 2,
    max_output_width: int = 1920,
) -> Path:
    """Render sparse and semi-dense anchors from one camera viewpoint."""

    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for camera-view visualization") from exc
    width = int(camera.width)
    height = int(camera.height)
    sparse_canvas = _camera_canvas(image_rgb, width, height)
    semidense_canvas = _camera_canvas(image_rgb, width, height)

    sparse_xy, sparse_visible, sparse_depth = _project_camera_view(sparse_index.xyz, pose_w2c, camera)
    sparse_colors = np.tile(np.asarray([[30, 120, 255]], dtype=np.uint8), (len(sparse_index), 1))
    sparse_count = _draw_projected_points(
        sparse_canvas,
        sparse_xy,
        sparse_visible,
        sparse_depth,
        sparse_colors,
        int(max_points),
        int(seed),
        int(point_radius),
    )

    semidense_xy, semidense_visible, semidense_depth = _project_camera_view(semidense_map.xyz, pose_w2c, camera)
    semidense_colors = _source_colors(semidense_map.source_types, semidense_map.quality_scores)
    semidense_count = _draw_projected_points(
        semidense_canvas,
        semidense_xy,
        semidense_visible,
        semidense_depth,
        semidense_colors,
        int(max_points),
        int(seed),
        int(point_radius),
    )

    cv2.putText(
        sparse_canvas,
        f"Sparse SfM anchors: {sparse_count}",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        semidense_canvas,
        f"Semi-dense anchors: {semidense_count}",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    combined = np.concatenate([sparse_canvas, semidense_canvas], axis=1)
    if int(max_output_width) > 0 and combined.shape[1] > int(max_output_width):
        scale = float(max_output_width) / float(combined.shape[1])
        combined = cv2.resize(
            combined,
            (int(round(combined.shape[1] * scale)), int(round(combined.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError(f"failed to write camera-view visualization: {output}")
    return output


def write_semidense_anchor_visualizations(
    output_dir: Path,
    sparse_index: LandmarkMapIndex,
    semidense_map: SemiDenseAnchorMap,
    max_points: int = 200_000,
    seed: int = 0,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sparse_idx = _sample_indices(len(sparse_index), int(max_points), int(seed))
    semi_idx = _sample_indices(len(semidense_map), int(max_points), int(seed))
    sparse_colors = _pca_colors(sparse_index.features[sparse_idx])
    semi_pca_colors = _pca_colors(semidense_map.features[semi_idx])
    semi_source_colors = _source_colors(semidense_map.source_types[semi_idx], semidense_map.quality_scores[semi_idx])
    sparse_pca_ply = output / "sparse_sfm_pca_color.ply"
    semidense_pca_ply = output / "semidense_pca_color.ply"
    semidense_source_ply = output / "semidense_source_quality_color.ply"
    topdown_png = output / "sparse_vs_semidense_topdown.png"
    _write_ascii_ply(
        sparse_pca_ply,
        sparse_index.xyz[sparse_idx],
        sparse_colors,
        sparse_index.track_ids[sparse_idx],
        sparse_index.track_ids[sparse_idx],
    )
    _write_ascii_ply(
        semidense_pca_ply,
        semidense_map.xyz[semi_idx],
        semi_pca_colors,
        semidense_map.anchor_ids[semi_idx],
        semidense_map.source_track_ids[semi_idx],
    )
    _write_ascii_ply(
        semidense_source_ply,
        semidense_map.xyz[semi_idx],
        semi_source_colors,
        semidense_map.anchor_ids[semi_idx],
        semidense_map.source_track_ids[semi_idx],
    )
    _write_topdown_png(
        topdown_png,
        sparse_index.xyz[sparse_idx],
        semidense_map.xyz[semi_idx],
        semi_source_colors,
    )
    return {
        "sparse_pca_ply": sparse_pca_ply,
        "semidense_pca_ply": semidense_pca_ply,
        "semidense_source_ply": semidense_source_ply,
        "topdown_png": topdown_png,
    }
