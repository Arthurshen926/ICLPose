"""Canonical dense feature atlases for 2DGS surface maplets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from feature_extract.vfm.gaussian_vfm_field import GaussianVFMSource
from feature_extract.vfm.surface_maplet_bank import VfmSurfaceMapletBank
from feature_extract.vfm.vfm_2dgs_mapping import _surface_tangent_axes_and_scales


def _normalize_features(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-8)


@dataclass(frozen=True)
class MapletFeatureAtlasBank:
    """Dense canonical charts; mapping observations never move chart geometry."""

    maplet_ids: np.ndarray
    centers: np.ndarray
    frames: np.ndarray
    extents: np.ndarray
    xyz: np.ndarray
    primitive_ids: np.ndarray
    features: np.ndarray
    variance: np.ndarray
    support_count: np.ndarray
    valid_mask: np.ndarray
    metadata: Mapping[str, object] | None = None
    mode_features: np.ndarray | None = None
    mode_weights: np.ndarray | None = None
    mode_view_directions: np.ndarray | None = None
    mode_view_covariance: np.ndarray | None = None
    mode_variance: np.ndarray | None = None
    mode_valid_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        ids = np.asarray(self.maplet_ids, dtype=np.int64).reshape(-1)
        count = int(ids.size)
        if np.unique(ids).size != count:
            raise ValueError("maplet_ids must be unique")
        xyz = np.asarray(self.xyz, dtype=np.float32)
        if xyz.ndim != 4 or xyz.shape[0] != count or xyz.shape[-1] != 3:
            raise ValueError("xyz must have shape (N,H,W,3)")
        height, width = int(xyz.shape[1]), int(xyz.shape[2])
        features = np.asarray(self.features, dtype=np.float32)
        if (
            features.ndim != 4
            or features.shape[0] != count
            or features.shape[2:] != (height, width)
        ):
            raise ValueError("features must have shape (N,C,H,W)")
        for name, shape, dtype in (
            ("centers", (count, 3), np.float32),
            ("frames", (count, 3, 3), np.float32),
            ("extents", (count, 3), np.float32),
            ("primitive_ids", (count, height, width), np.int64),
            ("variance", (count, height, width), np.float32),
            ("support_count", (count, height, width), np.int32),
            ("valid_mask", (count, height, width), np.bool_),
        ):
            value = np.asarray(getattr(self, name), dtype=dtype)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            object.__setattr__(self, name, value)
        valid = np.asarray(self.valid_mask, dtype=bool)
        if np.any(valid & (np.asarray(self.primitive_ids) < 0)):
            raise ValueError("valid atlas cells require a primitive ID")
        normalized = np.zeros_like(features, dtype=np.float32)
        flat = features.transpose(0, 2, 3, 1).reshape(-1, features.shape[1])
        normalized_flat = _normalize_features(flat)
        normalized[:] = normalized_flat.reshape(
            count, height, width, features.shape[1]
        ).transpose(0, 3, 1, 2)
        normalized *= valid[:, None]
        metadata = dict(self.metadata or {})
        forbidden = (
            "stores_mapping_rgb",
            "stores_mapping_image_paths",
            "uses_sfm_points",
            "uses_sfm_tracks",
            "uses_alike_descriptors",
            "uses_radio_intermediate",
            "uses_point_correspondence_pnp",
        )
        if any(bool(metadata.get(key, False)) for key in forbidden):
            raise ValueError("atlas violates the V6 map-only contract")
        object.__setattr__(self, "maplet_ids", ids)
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "features", normalized)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "metadata", metadata)
        mode_values = (
            self.mode_features,
            self.mode_weights,
            self.mode_view_directions,
            self.mode_view_covariance,
            self.mode_variance,
            self.mode_valid_mask,
        )
        if any(value is not None for value in mode_values):
            if not all(value is not None for value in mode_values):
                raise ValueError("all view-conditioned mode arrays are required")
            mode_features = np.asarray(self.mode_features, dtype=np.float32)
            if (
                mode_features.ndim != 5
                or mode_features.shape[0] != count
                or mode_features.shape[2] != features.shape[1]
                or mode_features.shape[3:] != (height, width)
            ):
                raise ValueError("mode_features must have shape (N,K,C,H,W)")
            modes = int(mode_features.shape[1])
            expected = (count, modes, height, width)
            mode_weights = np.asarray(self.mode_weights, dtype=np.float32)
            mode_directions = np.asarray(
                self.mode_view_directions, dtype=np.float32
            )
            mode_covariance = np.asarray(
                self.mode_view_covariance, dtype=np.float32
            )
            mode_variance = np.asarray(self.mode_variance, dtype=np.float32)
            mode_valid = np.asarray(self.mode_valid_mask, dtype=bool)
            if mode_weights.shape != expected:
                raise ValueError("mode_weights shape differs")
            if mode_directions.shape != (*expected, 3):
                raise ValueError("mode_view_directions shape differs")
            if mode_covariance.shape != (*expected, 3, 3):
                raise ValueError("mode_view_covariance shape differs")
            if mode_variance.shape != expected or mode_valid.shape != expected:
                raise ValueError("mode variance/valid shape differs")
            normalized_modes = _normalize_features(
                mode_features.transpose(0, 1, 3, 4, 2).reshape(
                    -1, features.shape[1]
                )
            ).reshape(count, modes, height, width, features.shape[1])
            normalized_modes = normalized_modes.transpose(0, 1, 4, 2, 3)
            normalized_modes *= mode_valid[:, :, None]
            mode_weights = np.where(mode_valid, mode_weights, 0.0)
            mode_weights /= np.maximum(
                np.sum(mode_weights, axis=1, keepdims=True), 1e-8
            )
            object.__setattr__(self, "mode_features", normalized_modes)
            object.__setattr__(self, "mode_weights", mode_weights)
            object.__setattr__(self, "mode_view_directions", mode_directions)
            object.__setattr__(self, "mode_view_covariance", mode_covariance)
            object.__setattr__(self, "mode_variance", mode_variance)
            object.__setattr__(self, "mode_valid_mask", mode_valid)

    @property
    def height(self) -> int:
        return int(self.xyz.shape[1])

    @property
    def width(self) -> int:
        return int(self.xyz.shape[2])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def appearance_mode_count(self) -> int:
        return 1 if self.mode_features is None else int(self.mode_features.shape[1])

    def __len__(self) -> int:
        return int(self.maplet_ids.size)

    def save_npz(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            maplet_ids=self.maplet_ids,
            centers=self.centers.astype(np.float32),
            frames=self.frames.astype(np.float32),
            extents=self.extents.astype(np.float32),
            xyz=self.xyz.astype(np.float32),
            primitive_ids=self.primitive_ids.astype(np.int64),
            features=self.features.astype(np.float16),
            variance=self.variance.astype(np.float16),
            support_count=self.support_count.astype(np.int32),
            valid_mask=self.valid_mask.astype(np.uint8),
            metadata_json=np.asarray(json.dumps(dict(self.metadata), sort_keys=True)),
        )
        if self.mode_features is not None:
            payload.update(
                mode_features=self.mode_features.astype(np.float16),
                mode_weights=self.mode_weights.astype(np.float16),
                mode_view_directions=self.mode_view_directions.astype(np.float16),
                mode_view_covariance=self.mode_view_covariance.astype(np.float16),
                mode_variance=self.mode_variance.astype(np.float16),
                mode_valid_mask=self.mode_valid_mask.astype(np.uint8),
            )
        np.savez_compressed(Path(path), **payload)

    @classmethod
    def load_npz(cls, path: Path) -> "MapletFeatureAtlasBank":
        with np.load(Path(path), allow_pickle=False) as data:
            optional = {}
            if "mode_features" in data.files:
                optional = {
                    "mode_features": data["mode_features"],
                    "mode_weights": data["mode_weights"],
                    "mode_view_directions": data["mode_view_directions"],
                    "mode_view_covariance": data["mode_view_covariance"],
                    "mode_variance": data["mode_variance"],
                    "mode_valid_mask": np.asarray(
                        data["mode_valid_mask"], dtype=bool
                    ),
                }
            return cls(
                maplet_ids=data["maplet_ids"],
                centers=data["centers"],
                frames=data["frames"],
                extents=data["extents"],
                xyz=data["xyz"],
                primitive_ids=data["primitive_ids"],
                features=data["features"],
                variance=data["variance"],
                support_count=data["support_count"],
                valid_mask=np.asarray(data["valid_mask"], dtype=bool),
                metadata=json.loads(str(data["metadata_json"].item())),
                **optional,
            )


def canonical_maplet_geometry(
    source: GaussianVFMSource,
    maplets: VfmSurfaceMapletBank,
    *,
    resolution: int = 32,
    maximum_disk_sigma: float = 3.0,
    minimum_normal_cosine: float = 0.10,
    maximum_chart_depth_ratio: float = 1.5,
    allowed_source_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Create observation-independent chart geometry from supporting 2DGS disks.

    Every atlas cell launches an orthographic ray along the maplet normal and
    intersects only the maplet's declared supporting disks.  The selected
    point is therefore a deterministic function of 2DGS geometry and atlas
    coordinates; mapping observations can change support/variance but never
    move it.
    """

    size = int(resolution)
    if size < 2:
        raise ValueError("resolution must be at least two")
    count = len(maplets)
    xyz = np.zeros((count, size, size, 3), dtype=np.float32)
    primitive_ids = np.full((count, size, size), -1, dtype=np.int64)
    valid_mask = np.zeros((count, size, size), dtype=bool)
    source_count = int(source.xyz.shape[0])
    allowed = None
    if allowed_source_indices is not None:
        allowed = np.zeros((source_count,), dtype=bool)
        allowed_rows = np.asarray(allowed_source_indices, dtype=np.int64)
        allowed_rows = allowed_rows[
            (allowed_rows >= 0) & (allowed_rows < source_count)
        ]
        allowed[allowed_rows] = True
    normals_all = np.asarray(source.normal, dtype=np.float32)
    if normals_all.shape != (source_count, 3):
        raise ValueError("complete 2DGS source must provide disk normals")
    u_grid = (np.arange(size, dtype=np.float64) + 0.5) / size * 2.0 - 1.0
    v_grid = (np.arange(size, dtype=np.float64) + 0.5) / size * 2.0 - 1.0
    vv, uu = np.meshgrid(v_grid, u_grid, indexing="ij")
    local_unit = np.stack([uu.reshape(-1), vv.reshape(-1)], axis=1)
    rejected_nonplanar = 0
    covered_cells = 0
    for row in range(count):
        begin = int(maplets.support_offsets[row])
        end = int(maplets.support_offsets[row + 1])
        support = np.unique(
            np.asarray(maplets.support_element_ids[begin:end], dtype=np.int64)
        )
        support = support[(support >= 0) & (support < source_count)]
        if allowed is not None:
            support = support[allowed[support]]
        if support.size == 0:
            continue
        frame = np.asarray(maplets.tangent_frames[row], dtype=np.float64)
        center = np.asarray(maplets.centers[row], dtype=np.float64)
        extent = np.maximum(
            np.asarray(maplets.extents[row], dtype=np.float64), 1e-4
        )
        if extent[2] / max(extent[0], extent[1]) > float(
            maximum_chart_depth_ratio
        ):
            rejected_nonplanar += 1
            continue
        bases = (
            center[None]
            + local_unit[:, 0:1] * extent[0] * frame[0][None]
            + local_unit[:, 1:2] * extent[1] * frame[1][None]
        )
        disk_centers = np.asarray(source.xyz, dtype=np.float64)[support]
        disk_normals = normals_all[support].astype(np.float64)
        tangent1, tangent2, scale1, scale2 = _surface_tangent_axes_and_scales(
            source, support, disk_normals.astype(np.float32)
        )
        denominator = disk_normals @ frame[2]
        compatible = np.abs(denominator) >= float(minimum_normal_cosine)
        safe_denominator = np.where(
            np.abs(denominator) > 1e-8,
            denominator,
            np.where(denominator < 0.0, -1e-8, 1e-8),
        )
        ray_t = (
            np.sum(
                disk_normals[None]
                * (disk_centers[None] - bases[:, None]),
                axis=2,
            )
            / safe_denominator[None]
        )
        points = bases[:, None] + ray_t[..., None] * frame[2][None, None]
        relative = points - disk_centers[None]
        disk_u = np.sum(relative * tangent1[None], axis=2) / np.maximum(
            scale1[None], 1e-6
        )
        disk_v = np.sum(relative * tangent2[None], axis=2) / np.maximum(
            scale2[None], 1e-6
        )
        radius2 = disk_u * disk_u + disk_v * disk_v
        valid = (
            compatible[None]
            & np.isfinite(points).all(axis=2)
            & (radius2 <= float(maximum_disk_sigma) ** 2)
            & (np.abs(ray_t) <= float(maximum_chart_depth_ratio) * extent[2])
        )
        score = (
            np.abs(ray_t) / max(extent[2], 1e-4)
            + 0.05 * radius2
            - 0.02
            * np.log(
                np.clip(
                    np.asarray(source.opacity, dtype=np.float32)[support][None],
                    1e-4,
                    1.0,
                )
            )
        )
        score[~valid] = np.inf
        selected = np.argmin(score, axis=1)
        selected_valid = np.isfinite(score[np.arange(score.shape[0]), selected])
        flat_xyz = xyz[row].reshape(-1, 3)
        flat_ids = primitive_ids[row].reshape(-1)
        flat_mask = valid_mask[row].reshape(-1)
        flat_xyz[selected_valid] = points[
            np.flatnonzero(selected_valid), selected[selected_valid]
        ].astype(np.float32)
        flat_ids[selected_valid] = support[selected[selected_valid]]
        flat_mask[selected_valid] = True
        covered_cells += int(np.sum(selected_valid))
    audit = {
        "maplet_count": count,
        "resolution": size,
        "valid_cell_count": covered_cells,
        "valid_cell_fraction": float(covered_cells / max(count * size * size, 1)),
        "rejected_nonplanar_maplet_count": rejected_nonplanar,
        "geometry_source": "canonical_maplet_grid_to_declared_2dgs_disk_intersection",
        "observation_dependent_geometry": False,
    }
    return xyz, primitive_ids, valid_mask, audit


def empty_atlas_bank_from_geometry(
    maplets: VfmSurfaceMapletBank,
    xyz: np.ndarray,
    primitive_ids: np.ndarray,
    valid_mask: np.ndarray,
    *,
    feature_dim: int,
    metadata: Mapping[str, object] | None = None,
) -> MapletFeatureAtlasBank:
    count, height, width = valid_mask.shape
    return MapletFeatureAtlasBank(
        maplet_ids=maplets.maplet_ids,
        centers=maplets.centers,
        frames=maplets.tangent_frames,
        extents=maplets.extents,
        xyz=xyz,
        primitive_ids=primitive_ids,
        features=np.zeros((count, int(feature_dim), height, width), dtype=np.float32),
        variance=np.ones((count, height, width), dtype=np.float32),
        support_count=np.zeros((count, height, width), dtype=np.int32),
        valid_mask=valid_mask,
        metadata=dict(metadata or {}),
    )
