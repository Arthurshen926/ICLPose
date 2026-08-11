"""Landmark-associated Gaussian VFM feature fields."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, normalize_rows


@dataclass(frozen=True)
class GaussianVFMFieldConfig:
    max_distance: float = 0.05
    k_neighbors: int = 4
    min_support: int = 1
    max_landmark_variance: float | None = None
    min_landmark_observations: int = 2
    l2_normalize_features: bool = True

    def __post_init__(self) -> None:
        if self.max_distance <= 0.0:
            raise ValueError("max_distance must be positive")
        if self.k_neighbors <= 0:
            raise ValueError("k_neighbors must be positive")
        if self.min_support <= 0:
            raise ValueError("min_support must be positive")
        if self.min_landmark_observations <= 0:
            raise ValueError("min_landmark_observations must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_distance": float(self.max_distance),
            "k_neighbors": int(self.k_neighbors),
            "min_support": int(self.min_support),
            "max_landmark_variance": None
            if self.max_landmark_variance is None
            else float(self.max_landmark_variance),
            "min_landmark_observations": int(self.min_landmark_observations),
            "l2_normalize_features": bool(self.l2_normalize_features),
        }


@dataclass(frozen=True)
class GaussianVFMRenderConfig:
    width: int
    height: int
    radius_px: float = 2.0
    depth_epsilon: float = 0.02
    l2_normalize_pixels: bool = True

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("render width and height must be positive")
        if self.radius_px <= 0.0:
            raise ValueError("radius_px must be positive")
        if self.depth_epsilon < 0.0:
            raise ValueError("depth_epsilon must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "radius_px": float(self.radius_px),
            "depth_epsilon": float(self.depth_epsilon),
            "l2_normalize_pixels": bool(self.l2_normalize_pixels),
        }


@dataclass(frozen=True)
class GaussianVFMRayContributionConfig:
    radius_px: float = 1.0
    depth_epsilon: float = 0.02
    min_samples: int = 2
    opacity_threshold: float = 0.0
    l2_normalize_observations: bool = True
    l2_normalize_features: bool = True

    def __post_init__(self) -> None:
        if self.radius_px <= 0.0:
            raise ValueError("radius_px must be positive")
        if self.depth_epsilon < 0.0:
            raise ValueError("depth_epsilon must be non-negative")
        if self.min_samples <= 0:
            raise ValueError("min_samples must be positive")
        if self.opacity_threshold < 0.0:
            raise ValueError("opacity_threshold must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "radius_px": float(self.radius_px),
            "depth_epsilon": float(self.depth_epsilon),
            "min_samples": int(self.min_samples),
            "opacity_threshold": float(self.opacity_threshold),
            "l2_normalize_observations": bool(self.l2_normalize_observations),
            "l2_normalize_features": bool(self.l2_normalize_features),
        }


@dataclass(frozen=True)
class GaussianVFMFeatureView:
    image_id: str
    feature_map: np.ndarray
    pose_w2c: np.ndarray
    camera: ColmapCamera

    def __post_init__(self) -> None:
        feature_map = np.asarray(self.feature_map, dtype=np.float32)
        if feature_map.ndim != 3:
            raise ValueError("feature_map must have shape (C, H, W)")
        object.__setattr__(self, "feature_map", feature_map)
        object.__setattr__(self, "pose_w2c", np.asarray(self.pose_w2c, dtype=np.float64).reshape(4, 4))


@dataclass(frozen=True)
class GaussianVFMSource:
    xyz: np.ndarray
    opacity: np.ndarray
    scale: np.ndarray
    gaussian_indices: np.ndarray
    scale_xyz: np.ndarray | None = None
    rotation: np.ndarray | None = None
    normal: np.ndarray | None = None

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz, dtype=np.float64)
        opacity = np.asarray(self.opacity, dtype=np.float32).reshape(-1)
        scale = np.asarray(self.scale, dtype=np.float32).reshape(-1)
        gaussian_indices = np.asarray(self.gaussian_indices, dtype=np.int64).reshape(-1)
        scale_xyz = None if self.scale_xyz is None else np.asarray(self.scale_xyz, dtype=np.float32)
        rotation = None if self.rotation is None else np.asarray(self.rotation, dtype=np.float32)
        normal = None if self.normal is None else np.asarray(self.normal, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (N, 3)")
        if opacity.shape != (xyz.shape[0],):
            raise ValueError("opacity must have shape (N,)")
        if scale.shape != (xyz.shape[0],):
            raise ValueError("scale must have shape (N,)")
        if gaussian_indices.shape != (xyz.shape[0],):
            raise ValueError("gaussian_indices must have shape (N,)")
        if scale_xyz is None:
            scale_xyz = np.repeat(scale[:, None], 3, axis=1)
        if scale_xyz.shape != (xyz.shape[0], 3):
            raise ValueError("scale_xyz must have shape (N, 3)")
        if rotation is not None:
            if rotation.shape != (xyz.shape[0], 4):
                raise ValueError("rotation must have shape (N, 4)")
            rotation = _normalize_quaternions(rotation)
        if normal is not None:
            if normal.shape != (xyz.shape[0], 3):
                raise ValueError("normal must have shape (N, 3)")
            normal = _normalize_vectors(normal)
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "opacity", opacity)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "gaussian_indices", gaussian_indices)
        object.__setattr__(self, "scale_xyz", scale_xyz)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "normal", normal)


@dataclass(frozen=True)
class GaussianRGBSource:
    xyz: np.ndarray
    rgb: np.ndarray
    opacity: np.ndarray
    scale: np.ndarray
    gaussian_indices: np.ndarray

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz, dtype=np.float64)
        rgb = np.asarray(self.rgb, dtype=np.float32)
        opacity = np.asarray(self.opacity, dtype=np.float32).reshape(-1)
        scale = np.asarray(self.scale, dtype=np.float32).reshape(-1)
        gaussian_indices = np.asarray(self.gaussian_indices, dtype=np.int64).reshape(-1)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (N, 3)")
        if rgb.shape != (xyz.shape[0], 3):
            raise ValueError("rgb must have shape (N, 3)")
        for name, value in (("opacity", opacity), ("scale", scale), ("gaussian_indices", gaussian_indices)):
            if value.shape != (xyz.shape[0],):
                raise ValueError(f"{name} must have shape (N,)")
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "rgb", np.clip(rgb, 0.0, 1.0))
        object.__setattr__(self, "opacity", opacity)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "gaussian_indices", gaussian_indices)


@dataclass(frozen=True)
class GaussianVFMField:
    xyz: np.ndarray
    features: np.ndarray
    opacity: np.ndarray
    scale: np.ndarray
    gaussian_indices: np.ndarray
    nearest_track_ids: np.ndarray
    support_counts: np.ndarray
    mean_distances: np.ndarray
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz, dtype=np.float64)
        features = np.asarray(self.features, dtype=np.float32)
        opacity = np.asarray(self.opacity, dtype=np.float32).reshape(-1)
        scale = np.asarray(self.scale, dtype=np.float32).reshape(-1)
        gaussian_indices = np.asarray(self.gaussian_indices, dtype=np.int64).reshape(-1)
        nearest_track_ids = np.asarray(self.nearest_track_ids, dtype=np.int64).reshape(-1)
        support_counts = np.asarray(self.support_counts, dtype=np.int64).reshape(-1)
        mean_distances = np.asarray(self.mean_distances, dtype=np.float32).reshape(-1)
        row_count = xyz.shape[0]
        if xyz.shape != (row_count, 3):
            raise ValueError("xyz must have shape (N, 3)")
        if features.ndim != 2 or features.shape[0] != row_count:
            raise ValueError("features must have shape (N, C)")
        for name, value in (
            ("opacity", opacity),
            ("scale", scale),
            ("gaussian_indices", gaussian_indices),
            ("nearest_track_ids", nearest_track_ids),
            ("support_counts", support_counts),
            ("mean_distances", mean_distances),
        ):
            if value.shape != (row_count,):
                raise ValueError(f"{name} must have shape (N,)")
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "opacity", opacity)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "gaussian_indices", gaussian_indices)
        object.__setattr__(self, "nearest_track_ids", nearest_track_ids)
        object.__setattr__(self, "support_counts", support_counts)
        object.__setattr__(self, "mean_distances", mean_distances)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1]) if self.features.ndim == 2 else 0

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def save_npz(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            xyz=self.xyz.astype(np.float32, copy=False),
            features=self.features.astype(np.float32, copy=False),
            opacity=self.opacity.astype(np.float32, copy=False),
            scale=self.scale.astype(np.float32, copy=False),
            gaussian_indices=self.gaussian_indices.astype(np.int64, copy=False),
            nearest_track_ids=self.nearest_track_ids.astype(np.int64, copy=False),
            support_counts=self.support_counts.astype(np.int64, copy=False),
            mean_distances=self.mean_distances.astype(np.float32, copy=False),
            metadata_json=np.asarray(json.dumps(dict(self.metadata or {}), sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "GaussianVFMField":
        with np.load(Path(path), allow_pickle=True) as data:
            metadata = {}
            if "metadata_json" in data:
                metadata = json.loads(str(data["metadata_json"].tolist()))
            return cls(
                xyz=np.asarray(data["xyz"], dtype=np.float64),
                features=np.asarray(data["features"], dtype=np.float32),
                opacity=np.asarray(data["opacity"], dtype=np.float32),
                scale=np.asarray(data["scale"], dtype=np.float32),
                gaussian_indices=np.asarray(data["gaussian_indices"], dtype=np.int64),
                nearest_track_ids=np.asarray(data["nearest_track_ids"], dtype=np.int64),
                support_counts=np.asarray(data["support_counts"], dtype=np.int64),
                mean_distances=np.asarray(data["mean_distances"], dtype=np.float32),
                metadata=metadata,
            )


@dataclass(frozen=True)
class GaussianVFMRenderResult:
    feature_map: np.ndarray
    xyz_map: np.ndarray
    visibility_mask: np.ndarray
    depth: np.ndarray
    weight_sum: np.ndarray
    dominant_gaussian_index: np.ndarray

    def save_npz(self, path: Path, metadata: Mapping[str, object] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            feature_map=self.feature_map.astype(np.float32, copy=False),
            xyz_map=self.xyz_map.astype(np.float32, copy=False),
            visibility_mask=self.visibility_mask.astype(bool, copy=False),
            depth=self.depth.astype(np.float32, copy=False),
            weight_sum=self.weight_sum.astype(np.float32, copy=False),
            dominant_gaussian_index=self.dominant_gaussian_index.astype(np.int64, copy=False),
            metadata_json=np.asarray(json.dumps(dict(metadata or {}), sort_keys=True)),
        )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float32), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _normalize_vectors(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, float(eps))


def _normalize_quaternions(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    quaternions = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    normalized = quaternions / np.maximum(norms, float(eps))
    zero_rows = np.squeeze(norms <= float(eps), axis=1)
    if np.any(zero_rows):
        normalized[zero_rows] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return normalized.astype(np.float32, copy=False)


def _quaternion_rotation_matrices(quaternions: np.ndarray) -> np.ndarray:
    q = _normalize_quaternions(quaternions)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    matrices = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    matrices[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrices[:, 0, 1] = 2.0 * (x * y - z * w)
    matrices[:, 0, 2] = 2.0 * (x * z + y * w)
    matrices[:, 1, 0] = 2.0 * (x * y + z * w)
    matrices[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrices[:, 1, 2] = 2.0 * (y * z - x * w)
    matrices[:, 2, 0] = 2.0 * (x * z - y * w)
    matrices[:, 2, 1] = 2.0 * (y * z + x * w)
    matrices[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrices


def _smallest_axis_normals(scale_xyz: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    axis_indices = np.argmin(np.asarray(scale_xyz, dtype=np.float32), axis=1)
    matrices = _quaternion_rotation_matrices(rotation)
    rows = np.arange(matrices.shape[0], dtype=np.int64)
    normals = matrices[rows, :, axis_indices]
    return _normalize_vectors(normals.astype(np.float32, copy=False))


def _normal_from_ply_fields(vertex, limit: int) -> np.ndarray | None:
    names = vertex.data.dtype.names or ()
    if not {"nx", "ny", "nz"}.issubset(set(names)):
        return None
    normal = np.stack(
        [
            np.asarray(vertex["nx"], dtype=np.float32)[:limit],
            np.asarray(vertex["ny"], dtype=np.float32)[:limit],
            np.asarray(vertex["nz"], dtype=np.float32)[:limit],
        ],
        axis=1,
    )
    if not np.any(np.linalg.norm(normal, axis=1) > 1e-8):
        return None
    return _normalize_vectors(normal)


def load_gaussian_vfm_source_from_ply(path: Path, max_gaussians: int = 0) -> GaussianVFMSource:
    """Load Gaussian centers and render weights from a 3DGS/2DGS PLY file."""

    vertex = PlyData.read(Path(path)).elements[0]
    row_count = int(vertex.count)
    limit = row_count if max_gaussians <= 0 else min(int(max_gaussians), row_count)
    # Cleaned derivative PLYs retain the row identity of the original 2DGS in
    # ``source_index``.  Treating those rows as a new 0..N index silently
    # breaks primitive lineage and prevents recovery of the original oriented
    # Gaussian geometry.
    if "source_index" in vertex.data.dtype.names:
        indices = np.asarray(vertex["source_index"], dtype=np.int64)[:limit]
    else:
        indices = np.arange(limit, dtype=np.int64)
    xyz = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float64)[:limit],
            np.asarray(vertex["y"], dtype=np.float64)[:limit],
            np.asarray(vertex["z"], dtype=np.float64)[:limit],
        ],
        axis=1,
    )
    if "opacity" in vertex.data.dtype.names:
        opacity = _sigmoid(np.asarray(vertex["opacity"], dtype=np.float32)[:limit])
    else:
        opacity = np.ones((limit,), dtype=np.float32)
    scale_names = sorted(
        [name for name in vertex.data.dtype.names if name.startswith("scale_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    scale_xyz = None
    if scale_names:
        scales = np.stack([np.asarray(vertex[name], dtype=np.float32)[:limit] for name in scale_names], axis=1)
        exp_scales = np.exp(scales).astype(np.float32, copy=False)
        scale = np.mean(exp_scales, axis=1)
        if exp_scales.shape[1] == 3:
            scale_xyz = exp_scales
        elif exp_scales.shape[1] == 2:
            scale_xyz = np.concatenate(
                [exp_scales, np.min(exp_scales, axis=1, keepdims=True)],
                axis=1,
            ).astype(np.float32, copy=False)
    else:
        scale = np.ones((limit,), dtype=np.float32)
    rotation_names = sorted(
        [name for name in vertex.data.dtype.names if name.startswith("rot_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    rotation = None
    normal = _normal_from_ply_fields(vertex, limit)
    if len(rotation_names) == 4:
        rotation = np.stack([np.asarray(vertex[name], dtype=np.float32)[:limit] for name in rotation_names], axis=1)
        rotation = _normalize_quaternions(rotation)
        if scale_xyz is not None and len(scale_names) == 3:
            normal = _smallest_axis_normals(scale_xyz, rotation)
        elif scale_xyz is not None and len(scale_names) == 2:
            normal = _normalize_vectors(_quaternion_rotation_matrices(rotation)[:, :, 2])
    return GaussianVFMSource(
        xyz=xyz,
        opacity=opacity,
        scale=scale,
        gaussian_indices=indices,
        scale_xyz=scale_xyz,
        rotation=rotation,
        normal=normal,
    )


def _dc_sh_to_rgb(dc: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(dc, dtype=np.float32) * 0.28209479177387814 + 0.5, 0.0, 1.0)


def load_gaussian_rgb_source_from_ply(path: Path, max_gaussians: int = 0) -> GaussianRGBSource:
    """Load Gaussian centers and approximate RGB colors from a 3DGS/2DGS PLY."""

    vertex = PlyData.read(Path(path)).elements[0]
    row_count = int(vertex.count)
    limit = row_count if max_gaussians <= 0 else min(int(max_gaussians), row_count)
    indices = np.arange(limit, dtype=np.int64)
    xyz = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float64)[:limit],
            np.asarray(vertex["y"], dtype=np.float64)[:limit],
            np.asarray(vertex["z"], dtype=np.float64)[:limit],
        ],
        axis=1,
    )
    names = vertex.data.dtype.names or ()
    if {"red", "green", "blue"}.issubset(set(names)):
        rgb = np.stack(
            [
                np.asarray(vertex["red"], dtype=np.float32)[:limit],
                np.asarray(vertex["green"], dtype=np.float32)[:limit],
                np.asarray(vertex["blue"], dtype=np.float32)[:limit],
            ],
            axis=1,
        )
        rgb = rgb / 255.0 if float(np.nanmax(rgb)) > 1.5 else rgb
    elif {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(set(names)):
        dc = np.stack(
            [
                np.asarray(vertex["f_dc_0"], dtype=np.float32)[:limit],
                np.asarray(vertex["f_dc_1"], dtype=np.float32)[:limit],
                np.asarray(vertex["f_dc_2"], dtype=np.float32)[:limit],
            ],
            axis=1,
        )
        rgb = _dc_sh_to_rgb(dc)
    else:
        rgb = np.ones((limit, 3), dtype=np.float32) * 0.5
    if "opacity" in names:
        opacity = _sigmoid(np.asarray(vertex["opacity"], dtype=np.float32)[:limit])
    else:
        opacity = np.ones((limit,), dtype=np.float32)
    scale_names = sorted(
        [name for name in names if name.startswith("scale_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    if scale_names:
        scales = np.stack([np.asarray(vertex[name], dtype=np.float32)[:limit] for name in scale_names], axis=1)
        scale = np.mean(np.exp(scales), axis=1)
    else:
        scale = np.ones((limit,), dtype=np.float32)
    return GaussianRGBSource(xyz=xyz, rgb=rgb, opacity=opacity, scale=scale, gaussian_indices=indices)


def export_gaussian_vfm_field_to_ply(
    source_ply: Path,
    field: GaussianVFMField,
    output_ply: Path,
    zero_unassigned: bool = True,
) -> None:
    """Write a Gaussian PLY with appended `loc_*` VFM feature attributes.

    The output keeps every source Gaussian row by default. Feature-bearing rows
    receive their associated VFM feature; unassigned rows receive zeros so
    renderers that expect one feature vector per Gaussian can load the file.
    """

    source_ply = Path(source_ply)
    output_ply = Path(output_ply)
    vertex = PlyData.read(source_ply).elements[0]
    source = np.asarray(vertex.data)
    row_count = int(source.shape[0])
    feature_dim = int(field.feature_dim)
    loc = np.zeros((row_count, feature_dim), dtype=np.float32)
    if not zero_unassigned:
        loc[:] = np.nan
    for row, gaussian_index in enumerate(field.gaussian_indices):
        idx = int(gaussian_index)
        if 0 <= idx < row_count:
            loc[idx] = field.features[row].astype(np.float32, copy=False)

    source_names = list(source.dtype.names or ())
    source_dtype = [(name, source.dtype.fields[name][0]) for name in source_names]
    loc_names = [f"loc_{idx}" for idx in range(feature_dim)]
    dtype = source_dtype + [(name, "f4") for name in loc_names]
    output = np.empty((row_count,), dtype=dtype)
    for name in source_names:
        output[name] = source[name]
    for idx, name in enumerate(loc_names):
        output[name] = loc[:, idx]
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(output, "vertex")]).write(output_ply)


def associate_landmarks_to_gaussians(
    source: GaussianVFMSource,
    landmarks: LandmarkMapIndex,
    config: GaussianVFMFieldConfig | None = None,
) -> GaussianVFMField:
    """Assign aggregated 3D VFM landmark features to nearby reliable Gaussians."""

    cfg = config or GaussianVFMFieldConfig()
    if len(landmarks) == 0 or source.xyz.shape[0] == 0:
        return GaussianVFMField(
            xyz=np.zeros((0, 3), dtype=np.float64),
            features=np.zeros((0, landmarks.feature_dim), dtype=np.float32),
            opacity=np.zeros((0,), dtype=np.float32),
            scale=np.zeros((0,), dtype=np.float32),
            gaussian_indices=np.zeros((0,), dtype=np.int64),
            nearest_track_ids=np.zeros((0,), dtype=np.int64),
            support_counts=np.zeros((0,), dtype=np.int64),
            mean_distances=np.zeros((0,), dtype=np.float32),
            metadata={"config": cfg.to_dict()},
        )
    landmark_mask = landmarks.observation_counts >= int(cfg.min_landmark_observations)
    if cfg.max_landmark_variance is not None:
        landmark_mask = landmark_mask & (landmarks.mean_variances <= float(cfg.max_landmark_variance))
    filtered = landmarks.subset(landmark_mask)
    if len(filtered) == 0:
        return associate_landmarks_to_gaussians(source, LandmarkMapIndex(
            track_ids=np.zeros((0,), dtype=np.int64),
            xyz=np.zeros((0, 3), dtype=np.float64),
            features=np.zeros((0, landmarks.feature_dim), dtype=np.float32),
            mean_variances=np.zeros((0,), dtype=np.float32),
            observation_counts=np.zeros((0,), dtype=np.int64),
            observation_image_ids=(),
        ), cfg)

    tree = cKDTree(filtered.xyz)
    neighbor_count = min(int(cfg.k_neighbors), len(filtered))
    distances, indices = tree.query(source.xyz, k=neighbor_count, distance_upper_bound=float(cfg.max_distance))
    distances = np.asarray(distances)
    indices = np.asarray(indices)
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    valid = np.isfinite(distances) & (indices >= 0) & (indices < len(filtered))
    support_counts = valid.sum(axis=1)
    keep = support_counts >= int(cfg.min_support)
    if not np.any(keep):
        return GaussianVFMField(
            xyz=np.zeros((0, 3), dtype=np.float64),
            features=np.zeros((0, filtered.feature_dim), dtype=np.float32),
            opacity=np.zeros((0,), dtype=np.float32),
            scale=np.zeros((0,), dtype=np.float32),
            gaussian_indices=np.zeros((0,), dtype=np.int64),
            nearest_track_ids=np.zeros((0,), dtype=np.int64),
            support_counts=np.zeros((0,), dtype=np.int64),
            mean_distances=np.zeros((0,), dtype=np.float32),
            metadata={"config": cfg.to_dict()},
        )

    field_features = []
    nearest_track_ids = []
    mean_distances = []
    for row in np.flatnonzero(keep):
        row_valid = valid[row]
        row_indices = indices[row, row_valid].astype(np.int64)
        row_distances = distances[row, row_valid].astype(np.float32)
        weights = 1.0 / np.maximum(row_distances, 1e-6)
        weights = weights / np.maximum(float(weights.sum()), 1e-12)
        feature = np.sum(filtered.features[row_indices] * weights[:, None], axis=0)
        field_features.append(feature.astype(np.float32))
        nearest = row_indices[int(np.argmin(row_distances))]
        nearest_track_ids.append(int(filtered.track_ids[nearest]))
        mean_distances.append(float(np.mean(row_distances)))
    features = np.stack(field_features, axis=0).astype(np.float32)
    if cfg.l2_normalize_features:
        features, _valid_norm = normalize_rows(features)
    return GaussianVFMField(
        xyz=source.xyz[keep],
        features=features,
        opacity=source.opacity[keep],
        scale=source.scale[keep],
        gaussian_indices=source.gaussian_indices[keep],
        nearest_track_ids=np.asarray(nearest_track_ids, dtype=np.int64),
        support_counts=support_counts[keep].astype(np.int64),
        mean_distances=np.asarray(mean_distances, dtype=np.float32),
        metadata={"config": cfg.to_dict()},
    )


def _project_gaussians_to_image(
    xyz: np.ndarray,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    xyz_h = np.concatenate([np.asarray(xyz, dtype=np.float64), np.ones((xyz.shape[0], 1), dtype=np.float64)], axis=1)
    cam_xyz = (pose @ xyz_h.T).T[:, :3]
    k_matrix = _intrinsic_matrix(camera, width, height)
    uvw = (k_matrix @ cam_xyz.T).T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-12)
    return uv, cam_xyz[:, 2]


def aggregate_ray_contributed_gaussian_vfm_features(
    source: GaussianVFMSource,
    views: list[GaussianVFMFeatureView],
    config: GaussianVFMRayContributionConfig | None = None,
) -> GaussianVFMField:
    """Aggregate dense VFM token features onto dominant visible Gaussians.

    This first version works on the feature-map grid. It assigns each feature
    token to the closest/frontmost Gaussian splat within `radius_px`, then
    averages token features per Gaussian.
    """

    cfg = config or GaussianVFMRayContributionConfig()
    if source.xyz.shape[0] == 0 or not views:
        feature_dim = int(views[0].feature_map.shape[0]) if views else 0
        return GaussianVFMField(
            xyz=np.zeros((0, 3), dtype=np.float64),
            features=np.zeros((0, feature_dim), dtype=np.float32),
            opacity=np.zeros((0,), dtype=np.float32),
            scale=np.zeros((0,), dtype=np.float32),
            gaussian_indices=np.zeros((0,), dtype=np.int64),
            nearest_track_ids=np.zeros((0,), dtype=np.int64),
            support_counts=np.zeros((0,), dtype=np.int64),
            mean_distances=np.zeros((0,), dtype=np.float32),
            metadata={"ray_contribution_config": cfg.to_dict()},
        )
    feature_dim = int(views[0].feature_map.shape[0])
    sums = np.zeros((source.xyz.shape[0], feature_dim), dtype=np.float64)
    counts = np.zeros((source.xyz.shape[0],), dtype=np.int64)
    distance_sums = np.zeros((source.xyz.shape[0],), dtype=np.float64)
    radius_sq = float(cfg.radius_px) * float(cfg.radius_px)
    for view in views:
        if int(view.feature_map.shape[0]) != feature_dim:
            raise ValueError("all feature views must have the same channel dimension")
        _channels, height, width = view.feature_map.shape
        uv, depths = _project_gaussians_to_image(source.xyz, view.pose_w2c, view.camera, width, height)
        order = np.argsort(depths, kind="mergesort")
        owner = np.full((height, width), -1, dtype=np.int64)
        owner_depth = np.full((height, width), np.inf, dtype=np.float64)
        owner_dist = np.zeros((height, width), dtype=np.float32)
        for idx in order:
            z = float(depths[int(idx)])
            if z <= 1e-6 or float(source.opacity[int(idx)]) < float(cfg.opacity_threshold):
                continue
            u, v = float(uv[int(idx), 0]), float(uv[int(idx), 1])
            radius = float(cfg.radius_px)
            if u < -radius or v < -radius or u >= width + radius or v >= height + radius:
                continue
            x0 = max(0, int(np.floor(u - radius)))
            x1 = min(width - 1, int(np.ceil(u + radius)))
            y0 = max(0, int(np.floor(v - radius)))
            y1 = min(height - 1, int(np.ceil(v + radius)))
            for y in range(y0, y1 + 1):
                dy = float(y) - v
                for x in range(x0, x1 + 1):
                    dx = float(x) - u
                    dist_sq = dx * dx + dy * dy
                    if dist_sq > radius_sq:
                        continue
                    if z > float(owner_depth[y, x]) + float(cfg.depth_epsilon):
                        continue
                    if z + float(cfg.depth_epsilon) < float(owner_depth[y, x]) or dist_sq < float(owner_dist[y, x]):
                        owner[y, x] = int(idx)
                        owner_depth[y, x] = z
                        owner_dist[y, x] = np.float32(dist_sq)
        for y in range(height):
            for x in range(width):
                idx = int(owner[y, x])
                if idx < 0:
                    continue
                feature = view.feature_map[:, y, x].astype(np.float32, copy=False)
                if cfg.l2_normalize_observations:
                    norm = max(float(np.linalg.norm(feature)), 1e-6)
                    feature = feature / norm
                sums[idx] += feature.astype(np.float64)
                counts[idx] += 1
                distance_sums[idx] += float(np.sqrt(owner_dist[y, x]))
    keep = counts >= int(cfg.min_samples)
    if not np.any(keep):
        return GaussianVFMField(
            xyz=np.zeros((0, 3), dtype=np.float64),
            features=np.zeros((0, feature_dim), dtype=np.float32),
            opacity=np.zeros((0,), dtype=np.float32),
            scale=np.zeros((0,), dtype=np.float32),
            gaussian_indices=np.zeros((0,), dtype=np.int64),
            nearest_track_ids=np.zeros((0,), dtype=np.int64),
            support_counts=np.zeros((0,), dtype=np.int64),
            mean_distances=np.zeros((0,), dtype=np.float32),
            metadata={"ray_contribution_config": cfg.to_dict()},
        )
    features = (sums[keep] / np.maximum(counts[keep, None], 1)).astype(np.float32)
    if cfg.l2_normalize_features:
        features, _valid = normalize_rows(features)
    return GaussianVFMField(
        xyz=source.xyz[keep],
        features=features,
        opacity=source.opacity[keep],
        scale=source.scale[keep],
        gaussian_indices=source.gaussian_indices[keep],
        nearest_track_ids=np.full((int(np.sum(keep)),), -1, dtype=np.int64),
        support_counts=counts[keep].astype(np.int64),
        mean_distances=(distance_sums[keep] / np.maximum(counts[keep], 1)).astype(np.float32),
        metadata={"ray_contribution_config": cfg.to_dict()},
    )


def merge_gaussian_vfm_fields(
    primary: GaussianVFMField,
    fallback: GaussianVFMField,
    l2_normalize_features: bool = False,
) -> GaussianVFMField:
    """Merge two Gaussian VFM fields, keeping primary features on conflicts."""

    if primary.feature_dim != fallback.feature_dim:
        raise ValueError("primary and fallback feature dimensions must match")
    rows: dict[int, tuple[np.ndarray, np.ndarray, float, float, int, int, float]] = {}
    for idx in range(len(primary)):
        gaussian_index = int(primary.gaussian_indices[idx])
        rows[gaussian_index] = (
            primary.xyz[idx],
            primary.features[idx],
            float(primary.opacity[idx]),
            float(primary.scale[idx]),
            int(primary.nearest_track_ids[idx]),
            int(primary.support_counts[idx]),
            float(primary.mean_distances[idx]),
        )
    fallback_added = 0
    for idx in range(len(fallback)):
        gaussian_index = int(fallback.gaussian_indices[idx])
        if gaussian_index in rows:
            continue
        rows[gaussian_index] = (
            fallback.xyz[idx],
            fallback.features[idx],
            float(fallback.opacity[idx]),
            float(fallback.scale[idx]),
            int(fallback.nearest_track_ids[idx]),
            int(fallback.support_counts[idx]),
            float(fallback.mean_distances[idx]),
        )
        fallback_added += 1
    if not rows:
        return GaussianVFMField(
            xyz=np.zeros((0, 3), dtype=np.float64),
            features=np.zeros((0, primary.feature_dim), dtype=np.float32),
            opacity=np.zeros((0,), dtype=np.float32),
            scale=np.zeros((0,), dtype=np.float32),
            gaussian_indices=np.zeros((0,), dtype=np.int64),
            nearest_track_ids=np.zeros((0,), dtype=np.int64),
            support_counts=np.zeros((0,), dtype=np.int64),
            mean_distances=np.zeros((0,), dtype=np.float32),
            metadata={
                "merge_strategy": "primary_preferred",
                "primary_count": int(len(primary)),
                "fallback_count": int(len(fallback)),
                "fallback_added_count": 0,
            },
        )
    ordered = sorted(rows)
    xyz = np.stack([rows[idx][0] for idx in ordered], axis=0).astype(np.float64)
    features = np.stack([rows[idx][1] for idx in ordered], axis=0).astype(np.float32)
    if l2_normalize_features:
        features, _valid = normalize_rows(features)
    return GaussianVFMField(
        xyz=xyz,
        features=features,
        opacity=np.asarray([rows[idx][2] for idx in ordered], dtype=np.float32),
        scale=np.asarray([rows[idx][3] for idx in ordered], dtype=np.float32),
        gaussian_indices=np.asarray(ordered, dtype=np.int64),
        nearest_track_ids=np.asarray([rows[idx][4] for idx in ordered], dtype=np.int64),
        support_counts=np.asarray([rows[idx][5] for idx in ordered], dtype=np.int64),
        mean_distances=np.asarray([rows[idx][6] for idx in ordered], dtype=np.float32),
        metadata={
            "merge_strategy": "primary_preferred",
            "primary_count": int(len(primary)),
            "fallback_count": int(len(fallback)),
            "fallback_added_count": int(fallback_added),
            "primary_metadata": dict(primary.metadata or {}),
            "fallback_metadata": dict(fallback.metadata or {}),
        },
    )


def _intrinsic_matrix(camera: ColmapCamera, width: int, height: int) -> np.ndarray:
    params = tuple(float(value) for value in camera.params)
    if camera.model_id == 1 and len(params) >= 4:
        fx, fy, cx, cy = params[:4]
    elif camera.model_id in {0, 2, 8} and len(params) >= 3:
        fx = fy = params[0]
        cx, cy = params[1:3]
    else:
        fx = fy = params[0] if params else float(max(width, height))
        cx, cy = float(width) * 0.5, float(height) * 0.5
    sx = float(width) / max(float(camera.width), 1.0)
    sy = float(height) / max(float(camera.height), 1.0)
    return np.asarray([[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]], dtype=np.float64)


def render_gaussian_vfm_feature_map(
    field: GaussianVFMField,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    config: GaussianVFMRenderConfig,
) -> GaussianVFMRenderResult:
    """Render a dense VFM feature map from feature-bearing Gaussians.

    This is a deterministic soft z-buffer splat renderer for verification and
    feature-map export. It intentionally does not replace a full 3DGS rasterizer.
    """

    feature_dim = field.feature_dim
    feature_sum = np.zeros((feature_dim, config.height, config.width), dtype=np.float32)
    xyz_sum = np.zeros((3, config.height, config.width), dtype=np.float32)
    weight_sum = np.zeros((config.height, config.width), dtype=np.float32)
    depth = np.full((config.height, config.width), np.inf, dtype=np.float32)
    dominant = np.full((config.height, config.width), -1, dtype=np.int64)
    if len(field) == 0:
        return GaussianVFMRenderResult(
            feature_map=feature_sum,
            xyz_map=np.zeros((config.height, config.width, 3), dtype=np.float32),
            visibility_mask=np.zeros((config.height, config.width), dtype=bool),
            depth=depth,
            weight_sum=weight_sum,
            dominant_gaussian_index=dominant,
        )
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    xyz_h = np.concatenate([field.xyz, np.ones((len(field), 1), dtype=np.float64)], axis=1)
    cam_xyz = (pose @ xyz_h.T).T[:, :3]
    in_front = cam_xyz[:, 2] > 1e-6
    if not np.any(in_front):
        return GaussianVFMRenderResult(
            feature_map=feature_sum,
            xyz_map=np.zeros((config.height, config.width, 3), dtype=np.float32),
            visibility_mask=np.zeros((config.height, config.width), dtype=bool),
            depth=depth,
            weight_sum=weight_sum,
            dominant_gaussian_index=dominant,
        )
    k_matrix = _intrinsic_matrix(camera, config.width, config.height)
    uvw = (k_matrix @ cam_xyz.T).T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-12)
    order = np.argsort(cam_xyz[:, 2], kind="mergesort")
    radius = float(config.radius_px)
    radius_sq = radius * radius
    for idx in order:
        if not in_front[int(idx)]:
            continue
        u, v = float(uv[int(idx), 0]), float(uv[int(idx), 1])
        if u < -radius or v < -radius or u >= config.width + radius or v >= config.height + radius:
            continue
        x0 = max(0, int(np.floor(u - radius)))
        x1 = min(config.width - 1, int(np.ceil(u + radius)))
        y0 = max(0, int(np.floor(v - radius)))
        y1 = min(config.height - 1, int(np.ceil(v + radius)))
        if x1 < x0 or y1 < y0:
            continue
        z = float(cam_xyz[int(idx), 2])
        for y in range(y0, y1 + 1):
            dy = float(y) - v
            for x in range(x0, x1 + 1):
                dx = float(x) - u
                dist_sq = dx * dx + dy * dy
                if dist_sq > radius_sq:
                    continue
                if z > float(depth[y, x]) + float(config.depth_epsilon):
                    continue
                if z + float(config.depth_epsilon) < float(depth[y, x]):
                    feature_sum[:, y, x] = 0.0
                    xyz_sum[:, y, x] = 0.0
                    weight_sum[y, x] = 0.0
                    depth[y, x] = np.float32(z)
                    dominant[y, x] = int(field.gaussian_indices[int(idx)])
                alpha = float(field.opacity[int(idx)])
                spatial = np.exp(-0.5 * dist_sq / max(radius_sq * 0.25, 1e-6))
                weight = np.float32(alpha * spatial)
                if weight <= 0.0:
                    continue
                feature_sum[:, y, x] += field.features[int(idx)] * weight
                xyz_sum[:, y, x] += field.xyz[int(idx)].astype(np.float32) * weight
                weight_sum[y, x] += weight
                if dominant[y, x] < 0:
                    dominant[y, x] = int(field.gaussian_indices[int(idx)])
    visible = weight_sum > 1e-8
    feature_map = np.zeros_like(feature_sum)
    xyz_map_chw = np.zeros_like(xyz_sum)
    feature_map[:, visible] = feature_sum[:, visible] / np.maximum(weight_sum[visible][None, :], 1e-8)
    xyz_map_chw[:, visible] = xyz_sum[:, visible] / np.maximum(weight_sum[visible][None, :], 1e-8)
    if config.l2_normalize_pixels and np.any(visible):
        pixels = feature_map[:, visible].T
        pixels, _valid = normalize_rows(pixels)
        feature_map[:, visible] = pixels.T
    depth[~visible] = 0.0
    return GaussianVFMRenderResult(
        feature_map=feature_map,
        xyz_map=np.transpose(xyz_map_chw, (1, 2, 0)).astype(np.float32, copy=False),
        visibility_mask=visible,
        depth=depth,
        weight_sum=weight_sum,
        dominant_gaussian_index=dominant,
    )


def render_gaussian_vfm_feature_map_gsplat(
    field: GaussianVFMField,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    config: GaussianVFMRenderConfig,
    device: str = "cuda",
    channel_chunk: int = 32,
) -> GaussianVFMRenderResult:
    """Render dense VFM features with gsplat alpha compositing."""

    try:
        import torch
        from gsplat import rasterization
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("gsplat and torch are required for gsplat rendering") from exc
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        return render_gaussian_vfm_feature_map(field, pose_w2c, camera, config)
    if torch_device.type != "cuda":
        return render_gaussian_vfm_feature_map(field, pose_w2c, camera, config)
    feature_dim = int(field.feature_dim)
    if len(field) == 0:
        zeros = np.zeros((feature_dim, config.height, config.width), dtype=np.float32)
        return GaussianVFMRenderResult(
            feature_map=zeros,
            xyz_map=np.zeros((config.height, config.width, 3), dtype=np.float32),
            visibility_mask=np.zeros((config.height, config.width), dtype=bool),
            depth=np.zeros((config.height, config.width), dtype=np.float32),
            weight_sum=np.zeros((config.height, config.width), dtype=np.float32),
            dominant_gaussian_index=np.full((config.height, config.width), -1, dtype=np.int64),
        )
    means = torch.as_tensor(field.xyz, dtype=torch.float32, device=torch_device)
    quats = torch.zeros((len(field), 4), dtype=torch.float32, device=torch_device)
    quats[:, 0] = 1.0
    scales_1d = torch.as_tensor(field.scale, dtype=torch.float32, device=torch_device).clamp(min=1e-4)
    scales = torch.stack([scales_1d, scales_1d, scales_1d], dim=1)
    opacities = torch.as_tensor(field.opacity, dtype=torch.float32, device=torch_device).reshape(-1).clamp(0.0, 1.0)
    colors = torch.as_tensor(field.features, dtype=torch.float32, device=torch_device)
    pose = torch.as_tensor(np.asarray(pose_w2c, dtype=np.float32).reshape(4, 4), dtype=torch.float32, device=torch_device)
    k_matrix = torch.as_tensor(
        _intrinsic_matrix(camera, config.width, config.height),
        dtype=torch.float32,
        device=torch_device,
    )
    colors_rendered, alphas, _info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=pose[None],
        Ks=k_matrix[None],
        width=int(config.width),
        height=int(config.height),
        packed=False,
        sh_degree=None,
        near_plane=0.01,
        far_plane=1e10,
        render_mode="RGB",
        channel_chunk=int(channel_chunk),
    )
    xyz_rendered, _xyz_alphas, _xyz_info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=means,
        viewmats=pose[None],
        Ks=k_matrix[None],
        width=int(config.width),
        height=int(config.height),
        packed=False,
        sh_degree=None,
        near_plane=0.01,
        far_plane=1e10,
        render_mode="RGB",
        channel_chunk=3,
    )
    feature_map = colors_rendered[0].permute(2, 0, 1).detach().cpu().numpy().astype(np.float32, copy=False)
    alpha_map = alphas[0]
    if alpha_map.ndim == 3:
        alpha_map = alpha_map[..., 0]
    weight_sum = alpha_map.detach().cpu().numpy().astype(np.float32, copy=False)
    visible = weight_sum > 1e-6
    xyz_map = xyz_rendered[0].detach().cpu().numpy().astype(np.float32, copy=False)
    if np.any(visible):
        xyz_map[visible] = xyz_map[visible] / np.maximum(weight_sum[visible, None], 1e-8)
    xyz_map[~visible] = 0.0
    depth = np.zeros((config.height, config.width), dtype=np.float32)
    if np.any(visible):
        pose_np = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
        visible_xyz = xyz_map[visible].astype(np.float64)
        visible_xyz_h = np.concatenate([visible_xyz, np.ones((visible_xyz.shape[0], 1), dtype=np.float64)], axis=1)
        depth[visible] = (pose_np @ visible_xyz_h.T).T[:, 2].astype(np.float32)
    if config.l2_normalize_pixels and np.any(visible):
        pixels = feature_map[:, visible].T
        pixels, _valid = normalize_rows(pixels)
        feature_map[:, visible] = pixels.T
    return GaussianVFMRenderResult(
        feature_map=feature_map,
        xyz_map=xyz_map,
        visibility_mask=visible,
        depth=depth,
        weight_sum=weight_sum,
        dominant_gaussian_index=np.full((config.height, config.width), -1, dtype=np.int64),
    )


def render_gaussian_rgb_image_gsplat(
    source: GaussianRGBSource,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    config: GaussianVFMRenderConfig,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Render an approximate RGB image from Gaussian DC colors."""

    try:
        import torch
        from gsplat import rasterization
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("gsplat and torch are required for RGB Gaussian rendering") from exc
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        return render_gaussian_rgb_image_soft(source, pose_w2c, camera, config)
    means = torch.as_tensor(source.xyz, dtype=torch.float32, device=torch_device)
    quats = torch.zeros((source.xyz.shape[0], 4), dtype=torch.float32, device=torch_device)
    quats[:, 0] = 1.0
    scales_1d = torch.as_tensor(source.scale, dtype=torch.float32, device=torch_device).clamp(min=1e-4)
    scales = torch.stack([scales_1d, scales_1d, scales_1d], dim=1)
    opacities = torch.as_tensor(source.opacity, dtype=torch.float32, device=torch_device).reshape(-1).clamp(0.0, 1.0)
    colors = torch.as_tensor(source.rgb, dtype=torch.float32, device=torch_device).clamp(0.0, 1.0)
    pose = torch.as_tensor(np.asarray(pose_w2c, dtype=np.float32).reshape(4, 4), dtype=torch.float32, device=torch_device)
    k_matrix = torch.as_tensor(
        _intrinsic_matrix(camera, config.width, config.height),
        dtype=torch.float32,
        device=torch_device,
    )
    rendered, alphas, _info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=pose[None],
        Ks=k_matrix[None],
        width=int(config.width),
        height=int(config.height),
        packed=False,
        sh_degree=None,
        near_plane=0.01,
        far_plane=1e10,
        render_mode="RGB",
        channel_chunk=3,
    )
    rgb = rendered[0].detach().cpu().numpy().astype(np.float32, copy=False)
    alpha = alphas[0]
    if alpha.ndim == 3:
        alpha = alpha[..., 0]
    return np.clip(rgb, 0.0, 1.0), alpha.detach().cpu().numpy().astype(np.float32, copy=False)


def render_gaussian_rgb_image_soft(
    source: GaussianRGBSource,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    config: GaussianVFMRenderConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic soft z-buffer RGB splat renderer for diagnostics."""

    rgb_sum = np.zeros((3, config.height, config.width), dtype=np.float32)
    weight_sum = np.zeros((config.height, config.width), dtype=np.float32)
    depth = np.full((config.height, config.width), np.inf, dtype=np.float32)
    if source.xyz.shape[0] == 0:
        return np.zeros((config.height, config.width, 3), dtype=np.float32), weight_sum
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    xyz_h = np.concatenate([source.xyz, np.ones((source.xyz.shape[0], 1), dtype=np.float64)], axis=1)
    cam_xyz = (pose @ xyz_h.T).T[:, :3]
    in_front = cam_xyz[:, 2] > 1e-6
    k_matrix = _intrinsic_matrix(camera, config.width, config.height)
    uvw = (k_matrix @ cam_xyz.T).T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-12)
    order = np.argsort(cam_xyz[:, 2], kind="mergesort")
    radius = float(config.radius_px)
    radius_sq = radius * radius
    for idx in order:
        if not in_front[int(idx)]:
            continue
        u, v = float(uv[int(idx), 0]), float(uv[int(idx), 1])
        if u < -radius or v < -radius or u >= config.width + radius or v >= config.height + radius:
            continue
        x0 = max(0, int(np.floor(u - radius)))
        x1 = min(config.width - 1, int(np.ceil(u + radius)))
        y0 = max(0, int(np.floor(v - radius)))
        y1 = min(config.height - 1, int(np.ceil(v + radius)))
        z = float(cam_xyz[int(idx), 2])
        for y in range(y0, y1 + 1):
            dy = float(y) - v
            for x in range(x0, x1 + 1):
                dx = float(x) - u
                dist_sq = dx * dx + dy * dy
                if dist_sq > radius_sq:
                    continue
                if z > float(depth[y, x]) + float(config.depth_epsilon):
                    continue
                if z + float(config.depth_epsilon) < float(depth[y, x]):
                    rgb_sum[:, y, x] = 0.0
                    weight_sum[y, x] = 0.0
                    depth[y, x] = np.float32(z)
                spatial = np.exp(-0.5 * dist_sq / max(radius_sq * 0.25, 1e-6))
                weight = np.float32(float(source.opacity[int(idx)]) * spatial)
                if weight <= 0.0:
                    continue
                rgb_sum[:, y, x] += source.rgb[int(idx)] * weight
                weight_sum[y, x] += weight
    visible = weight_sum > 1e-8
    rgb = np.zeros_like(rgb_sum)
    rgb[:, visible] = rgb_sum[:, visible] / np.maximum(weight_sum[visible][None, :], 1e-8)
    return np.transpose(np.clip(rgb, 0.0, 1.0), (1, 2, 0)).astype(np.float32, copy=False), weight_sum
