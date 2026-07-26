"""Track-free, two-level VFM/2DGS localization map.

The mapping code in :mod:`feature_extract.vfm.vfm_2dgs_mapping` associates
RADIO tokens with 2DGS surface support.  Its fused rows are useful VFM-scale
regions, but a region centroid is not a sufficiently stable metric point for
PnP.  This module makes the missing distinction explicit:

* :class:`VfmSurfaceMapletBank` is the RADIO-final retrieval representation.
* :class:`StableSurfaceAnchorMap` contains fixed 2DGS surface elements used by
  the local feature matcher and PnP.

No SfM point, observation track, or RADIO intermediate feature is accepted by
this module.  Mapping camera poses are still required to turn persistent 2DGS
surface elements into real-image support observations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

from feature_extract.vfm.gaussian_vfm_field import GaussianVFMFeatureView, _intrinsic_matrix
from feature_extract.vfm.vfm_2dgs_mapping import (
    SurfaceElementMap,
    Vfm2DgsAnchorMap,
    Vfm2DgsObservationBank,
)


def _normalize_rows(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must have shape (N, C)")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, float(eps))


def _normalize_vector(value: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    return vector / max(float(np.linalg.norm(vector)), float(eps))


def _validate_offsets(name: str, offsets: np.ndarray, row_count: int, value_count: int) -> np.ndarray:
    value = np.asarray(offsets, dtype=np.int64).reshape(-1)
    if value.shape != (row_count + 1,):
        raise ValueError(f"{name} must have shape (N + 1,)")
    if int(value[0]) != 0 or int(value[-1]) != int(value_count) or np.any(np.diff(value) < 0):
        raise ValueError(f"{name} is not a valid monotonic ragged offset array")
    return value


@dataclass(frozen=True)
class RadioFinalRegionConfig:
    """Spatial context encoded solely from RADIO final features."""

    pool_sizes: tuple[int, ...] = (1, 3, 5, 9)
    pool_weights: tuple[float, ...] = (0.40, 0.30, 0.20, 0.10)
    global_context_weight: float = 0.0

    def __post_init__(self) -> None:
        if not self.pool_sizes:
            raise ValueError("at least one RADIO-final pooling scale is required")
        if len(self.pool_sizes) != len(self.pool_weights):
            raise ValueError("pool_sizes and pool_weights must have equal length")
        if any(int(size) <= 0 or int(size) % 2 == 0 for size in self.pool_sizes):
            raise ValueError("RADIO-final pool sizes must be positive odd integers")
        if any(float(weight) < 0.0 for weight in self.pool_weights):
            raise ValueError("RADIO-final pool weights must be non-negative")
        if float(sum(self.pool_weights)) <= 0.0:
            raise ValueError("at least one RADIO-final pool weight must be positive")
        if float(self.global_context_weight) < 0.0:
            raise ValueError("global_context_weight must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_source": "radio_final",
            "pool_sizes": [int(value) for value in self.pool_sizes],
            "pool_weights": [float(value) for value in self.pool_weights],
            "global_context_weight": float(self.global_context_weight),
        }


def encode_radio_final_regions(
    feature_map: np.ndarray,
    token_xy: np.ndarray,
    config: RadioFinalRegionConfig = RadioFinalRegionConfig(),
) -> np.ndarray:
    """Encode multi-scale regions without using RADIO intermediate activations.

    Each spatial mean is normalized before scale fusion.  Consequently a large
    crop contributes context rather than simply dominating through its feature
    magnitude.  The returned descriptor dimension remains the RADIO-final
    channel dimension, so an existing full-map mapper may be applied before
    this operation or to every pooled map consistently.
    """

    fmap = np.asarray(feature_map, dtype=np.float32)
    if fmap.ndim == 4 and int(fmap.shape[0]) == 1:
        fmap = fmap[0]
    if fmap.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    points = np.asarray(token_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("token_xy must have shape (N, 2)")
    channels, height, width = fmap.shape
    output = np.zeros((points.shape[0], channels), dtype=np.float32)
    total_weight = float(sum(config.pool_weights)) + float(config.global_context_weight)
    global_descriptor = None
    if float(config.global_context_weight) > 0.0:
        global_descriptor = _normalize_rows(np.mean(fmap, axis=(1, 2), keepdims=False)[None])[0]
    for row, point in enumerate(points):
        x = int(np.clip(np.rint(float(point[0])), 0, width - 1))
        y = int(np.clip(np.rint(float(point[1])), 0, height - 1))
        descriptor = np.zeros((channels,), dtype=np.float32)
        for size, weight in zip(config.pool_sizes, config.pool_weights):
            if float(weight) <= 0.0:
                continue
            radius = int(size) // 2
            x0, x1 = max(0, x - radius), min(width, x + radius + 1)
            y0, y1 = max(0, y - radius), min(height, y + radius + 1)
            pooled = np.mean(fmap[:, y0:y1, x0:x1], axis=(1, 2))
            pooled = _normalize_vector(pooled).astype(np.float32, copy=False)
            descriptor += float(weight) * pooled
        if global_descriptor is not None:
            descriptor += float(config.global_context_weight) * global_descriptor
        output[row] = descriptor / max(total_weight, 1e-8)
    return _normalize_rows(output)


@dataclass(frozen=True)
class TwoDGSPrimitiveQuality:
    """Optional reconstruction-quality fields indexed by source primitive."""

    geometry_confidence: np.ndarray
    primitive_class: np.ndarray
    protected_flag: np.ndarray
    block_id: np.ndarray

    def __post_init__(self) -> None:
        geometry = np.asarray(self.geometry_confidence, dtype=np.float32).reshape(-1)
        count = int(geometry.shape[0])
        object.__setattr__(self, "geometry_confidence", geometry)
        for name, dtype in (
            ("primitive_class", np.int32),
            ("protected_flag", np.int8),
            ("block_id", np.int32),
        ):
            value = np.asarray(getattr(self, name), dtype=dtype).reshape(-1)
            if value.shape != (count,):
                raise ValueError(f"{name} must have the same length as geometry_confidence")
            object.__setattr__(self, name, value)

    def __len__(self) -> int:
        return int(self.geometry_confidence.shape[0])

    @classmethod
    def neutral(cls, count: int) -> "TwoDGSPrimitiveQuality":
        return cls(
            geometry_confidence=np.ones((int(count),), dtype=np.float32),
            primitive_class=np.zeros((int(count),), dtype=np.int32),
            protected_flag=np.zeros((int(count),), dtype=np.int8),
            block_id=np.full((int(count),), -1, dtype=np.int32),
        )


def load_2dgs_primitive_quality(path: Path) -> TwoDGSPrimitiveQuality:
    """Read optional quality fields without interpreting a Gaussian as an ID."""

    vertex = PlyData.read(Path(path), mmap=True).elements[0]
    names = set(vertex.data.dtype.names or ())
    count = int(vertex.count)

    def field(name: str, default: float, dtype) -> np.ndarray:
        if name not in names:
            return np.full((count,), default, dtype=dtype)
        return np.asarray(vertex[name], dtype=dtype)

    return TwoDGSPrimitiveQuality(
        geometry_confidence=field("geometry_confidence", 1.0, np.float32),
        primitive_class=field("primitive_class", 0, np.int32),
        protected_flag=field("protected_flag", 0, np.int8),
        block_id=field("block_id", -1, np.int32),
    )


@dataclass(frozen=True)
class SurfaceMapletBuildConfig:
    """Quality and coverage policy for metric anchors inside VFM maplets."""

    assignment_neighbor_count: int = 8
    assignment_max_center_distance: float = 1.0
    assignment_min_surface_iou: float = 0.05
    min_anchor_views: int = 2
    min_anchors_per_maplet: int = 4
    max_anchors_per_maplet: int = 64
    min_anchor_opacity: float = 0.05
    min_geometry_confidence: float = 0.5
    allowed_primitive_classes: tuple[int, ...] | None = (0,)
    min_anchor_normal_cosine: float = 0.5
    min_anchor_separation: float = 0.02
    max_anchor_scale: float = 0.0
    observation_min_element_weight: float = 0.01
    normal_variance_scale: float = 0.02

    def __post_init__(self) -> None:
        if int(self.assignment_neighbor_count) <= 0:
            raise ValueError("assignment_neighbor_count must be positive")
        if float(self.assignment_max_center_distance) <= 0.0:
            raise ValueError("assignment_max_center_distance must be positive")
        if not 0.0 <= float(self.assignment_min_surface_iou) <= 1.0:
            raise ValueError("assignment_min_surface_iou must be in [0, 1]")
        if int(self.min_anchor_views) <= 0:
            raise ValueError("min_anchor_views must be positive")
        if not 0 <= int(self.min_anchors_per_maplet) <= int(self.max_anchors_per_maplet):
            raise ValueError("anchor count bounds are invalid")
        if not 0.0 <= float(self.min_anchor_opacity) <= 1.0:
            raise ValueError("min_anchor_opacity must be in [0, 1]")
        if float(self.min_geometry_confidence) < 0.0:
            raise ValueError("min_geometry_confidence must be non-negative")
        if not -1.0 <= float(self.min_anchor_normal_cosine) <= 1.0:
            raise ValueError("min_anchor_normal_cosine must be in [-1, 1]")
        if float(self.min_anchor_separation) < 0.0:
            raise ValueError("min_anchor_separation must be non-negative")
        if float(self.max_anchor_scale) < 0.0:
            raise ValueError("max_anchor_scale must be non-negative")
        if float(self.observation_min_element_weight) < 0.0:
            raise ValueError("observation_min_element_weight must be non-negative")
        if float(self.normal_variance_scale) <= 0.0:
            raise ValueError("normal_variance_scale must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "assignment_neighbor_count": int(self.assignment_neighbor_count),
            "assignment_max_center_distance": float(self.assignment_max_center_distance),
            "assignment_min_surface_iou": float(self.assignment_min_surface_iou),
            "min_anchor_views": int(self.min_anchor_views),
            "min_anchors_per_maplet": int(self.min_anchors_per_maplet),
            "max_anchors_per_maplet": int(self.max_anchors_per_maplet),
            "min_anchor_opacity": float(self.min_anchor_opacity),
            "min_geometry_confidence": float(self.min_geometry_confidence),
            "allowed_primitive_classes": (
                None
                if self.allowed_primitive_classes is None
                else [int(value) for value in self.allowed_primitive_classes]
            ),
            "min_anchor_normal_cosine": float(self.min_anchor_normal_cosine),
            "min_anchor_separation": float(self.min_anchor_separation),
            "max_anchor_scale": float(self.max_anchor_scale),
            "observation_min_element_weight": float(self.observation_min_element_weight),
            "normal_variance_scale": float(self.normal_variance_scale),
        }


@dataclass(frozen=True)
class StableSurfaceAnchorMap:
    """Persistent 2DGS surface identities and their real-view projections."""

    anchor_ids: np.ndarray
    owner_maplet_ids: np.ndarray
    surface_element_ids: np.ndarray
    parent_primitive_indices: np.ndarray
    xyz: np.ndarray
    normals: np.ndarray
    tangent_covariances: np.ndarray
    support_radii: np.ndarray
    quality_scores: np.ndarray
    geometry_confidence: np.ndarray
    opacity: np.ndarray
    observation_offsets: np.ndarray
    observation_image_ids: tuple[str, ...]
    observation_xy: np.ndarray
    observation_depth: np.ndarray
    observation_weights: np.ndarray
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        anchor_ids = np.asarray(self.anchor_ids, dtype=np.int64).reshape(-1)
        count = int(anchor_ids.shape[0])
        if len(np.unique(anchor_ids)) != count:
            raise ValueError("stable surface anchor IDs must be unique")
        object.__setattr__(self, "anchor_ids", anchor_ids)
        for name in ("owner_maplet_ids", "surface_element_ids", "parent_primitive_indices"):
            value = np.asarray(getattr(self, name), dtype=np.int64).reshape(-1)
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape (N,)")
            object.__setattr__(self, name, value)
        for name, shape in (
            ("xyz", (count, 3)),
            ("normals", (count, 3)),
            ("tangent_covariances", (count, 3, 3)),
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64 if name == "xyz" else np.float32)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "normals", _normalize_rows(np.asarray(self.normals, dtype=np.float32)))
        for name in ("support_radii", "quality_scores", "geometry_confidence", "opacity"):
            value = np.asarray(getattr(self, name), dtype=np.float32).reshape(-1)
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape (N,)")
            object.__setattr__(self, name, value)
        image_ids = tuple(str(value) for value in self.observation_image_ids)
        observation_count = len(image_ids)
        object.__setattr__(self, "observation_image_ids", image_ids)
        object.__setattr__(
            self,
            "observation_offsets",
            _validate_offsets("observation_offsets", self.observation_offsets, count, observation_count),
        )
        xy = np.asarray(self.observation_xy, dtype=np.float32)
        if xy.shape != (observation_count, 2):
            raise ValueError("observation_xy must have shape (M, 2)")
        object.__setattr__(self, "observation_xy", xy)
        for name in ("observation_depth", "observation_weights"):
            value = np.asarray(getattr(self, name), dtype=np.float32).reshape(-1)
            if value.shape != (observation_count,):
                raise ValueError(f"{name} must have shape (M,)")
            object.__setattr__(self, name, value)
        metadata = dict(self.metadata or {})
        if bool(metadata.get("uses_sfm_points", False)):
            raise ValueError("SfM points are forbidden in the stable surface-anchor map")
        if bool(metadata.get("uses_sfm_tracks", False)):
            raise ValueError("SfM tracks are forbidden in the stable surface-anchor map")
        object.__setattr__(self, "metadata", metadata)

    def __len__(self) -> int:
        return int(self.anchor_ids.shape[0])

    def row_by_id(self) -> dict[int, int]:
        return {int(value): int(row) for row, value in enumerate(self.anchor_ids.tolist())}

    def save_npz(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            anchor_ids=self.anchor_ids,
            owner_maplet_ids=self.owner_maplet_ids,
            surface_element_ids=self.surface_element_ids,
            parent_primitive_indices=self.parent_primitive_indices,
            xyz=self.xyz.astype(np.float32),
            normals=self.normals,
            tangent_covariances=self.tangent_covariances,
            support_radii=self.support_radii,
            quality_scores=self.quality_scores,
            geometry_confidence=self.geometry_confidence,
            opacity=self.opacity,
            observation_offsets=self.observation_offsets,
            observation_xy=self.observation_xy,
            observation_depth=self.observation_depth,
            observation_weights=self.observation_weights,
            observation_image_ids_json=np.asarray(json.dumps(list(self.observation_image_ids))),
            metadata_json=np.asarray(json.dumps(dict(self.metadata or {}), sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "StableSurfaceAnchorMap":
        with np.load(Path(path), allow_pickle=True) as data:
            return cls(
                anchor_ids=np.asarray(data["anchor_ids"], dtype=np.int64),
                owner_maplet_ids=np.asarray(data["owner_maplet_ids"], dtype=np.int64),
                surface_element_ids=np.asarray(data["surface_element_ids"], dtype=np.int64),
                parent_primitive_indices=np.asarray(data["parent_primitive_indices"], dtype=np.int64),
                xyz=np.asarray(data["xyz"], dtype=np.float64),
                normals=np.asarray(data["normals"], dtype=np.float32),
                tangent_covariances=np.asarray(data["tangent_covariances"], dtype=np.float32),
                support_radii=np.asarray(data["support_radii"], dtype=np.float32),
                quality_scores=np.asarray(data["quality_scores"], dtype=np.float32),
                geometry_confidence=np.asarray(data["geometry_confidence"], dtype=np.float32),
                opacity=np.asarray(data["opacity"], dtype=np.float32),
                observation_offsets=np.asarray(data["observation_offsets"], dtype=np.int64),
                observation_image_ids=tuple(
                    json.loads(str(np.asarray(data["observation_image_ids_json"]).item()))
                ),
                observation_xy=np.asarray(data["observation_xy"], dtype=np.float32),
                observation_depth=np.asarray(data["observation_depth"], dtype=np.float32),
                observation_weights=np.asarray(data["observation_weights"], dtype=np.float32),
                metadata=(
                    json.loads(str(np.asarray(data["metadata_json"]).item()))
                    if "metadata_json" in data
                    else {}
                ),
            )


@dataclass(frozen=True)
class VfmSurfaceMapletBank:
    """RADIO-final surface regions with multi-view context and metric anchors."""

    maplet_ids: np.ndarray
    centers: np.ndarray
    normals: np.ndarray
    tangent_frames: np.ndarray
    extents: np.ndarray
    descriptors: np.ndarray
    quality_scores: np.ndarray
    descriptor_variances: np.ndarray
    anchor_offsets: np.ndarray
    anchor_ids: np.ndarray
    support_offsets: np.ndarray
    support_element_ids: np.ndarray
    view_offsets: np.ndarray
    view_image_ids: tuple[str, ...]
    view_token_xy: np.ndarray
    view_grid_sizes: np.ndarray
    view_descriptors: np.ndarray
    view_quality_scores: np.ndarray
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        maplet_ids = np.asarray(self.maplet_ids, dtype=np.int64).reshape(-1)
        count = int(maplet_ids.shape[0])
        if len(np.unique(maplet_ids)) != count:
            raise ValueError("surface maplet IDs must be unique")
        object.__setattr__(self, "maplet_ids", maplet_ids)
        descriptors = np.asarray(self.descriptors, dtype=np.float32)
        if descriptors.ndim != 2 or descriptors.shape[0] != count:
            raise ValueError("descriptors must have shape (N, C)")
        object.__setattr__(self, "descriptors", _normalize_rows(descriptors))
        feature_dim = int(descriptors.shape[1])
        for name, shape in (
            ("centers", (count, 3)),
            ("normals", (count, 3)),
            ("tangent_frames", (count, 3, 3)),
            ("extents", (count, 3)),
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64 if name == "centers" else np.float32)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "normals", _normalize_rows(np.asarray(self.normals, dtype=np.float32)))
        for name in ("quality_scores", "descriptor_variances"):
            value = np.asarray(getattr(self, name), dtype=np.float32).reshape(-1)
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape (N,)")
            object.__setattr__(self, name, value)
        anchor_ids = np.asarray(self.anchor_ids, dtype=np.int64).reshape(-1)
        support_ids = np.asarray(self.support_element_ids, dtype=np.int64).reshape(-1)
        object.__setattr__(self, "anchor_ids", anchor_ids)
        object.__setattr__(self, "support_element_ids", support_ids)
        object.__setattr__(
            self,
            "anchor_offsets",
            _validate_offsets("anchor_offsets", self.anchor_offsets, count, len(anchor_ids)),
        )
        object.__setattr__(
            self,
            "support_offsets",
            _validate_offsets("support_offsets", self.support_offsets, count, len(support_ids)),
        )
        view_ids = tuple(str(value) for value in self.view_image_ids)
        view_count = len(view_ids)
        object.__setattr__(self, "view_image_ids", view_ids)
        object.__setattr__(
            self,
            "view_offsets",
            _validate_offsets("view_offsets", self.view_offsets, count, view_count),
        )
        view_xy = np.asarray(self.view_token_xy, dtype=np.float32)
        view_sizes = np.asarray(self.view_grid_sizes, dtype=np.int32)
        view_descriptors = np.asarray(self.view_descriptors, dtype=np.float32)
        if view_xy.shape != (view_count, 2):
            raise ValueError("view_token_xy must have shape (V, 2)")
        if view_sizes.shape != (view_count, 2):
            raise ValueError("view_grid_sizes must have shape (V, 2)")
        if view_descriptors.shape != (view_count, feature_dim):
            raise ValueError("view_descriptors must have shape (V, C)")
        object.__setattr__(self, "view_token_xy", view_xy)
        object.__setattr__(self, "view_grid_sizes", view_sizes)
        object.__setattr__(self, "view_descriptors", _normalize_rows(view_descriptors))
        view_quality = np.asarray(self.view_quality_scores, dtype=np.float32).reshape(-1)
        if view_quality.shape != (view_count,):
            raise ValueError("view_quality_scores must have shape (V,)")
        object.__setattr__(self, "view_quality_scores", view_quality)
        metadata = dict(self.metadata or {})
        if metadata.get("vfm_layer", "radio_final") != "radio_final":
            raise ValueError("production surface maplets only accept RADIO final features")
        if bool(metadata.get("uses_radio_intermediate", False)):
            raise ValueError("RADIO intermediate is forbidden in the production surface map")
        if bool(metadata.get("uses_sfm_tracks", False)):
            raise ValueError("SfM tracks are forbidden in the production surface map")
        if bool(metadata.get("uses_sfm_points", False)):
            raise ValueError("SfM points are forbidden in the production surface map")
        object.__setattr__(self, "metadata", metadata)

    def __len__(self) -> int:
        return int(self.maplet_ids.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.descriptors.shape[1])

    def save_npz(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            maplet_ids=self.maplet_ids,
            centers=self.centers.astype(np.float32),
            normals=self.normals,
            tangent_frames=self.tangent_frames,
            extents=self.extents,
            descriptors=self.descriptors,
            quality_scores=self.quality_scores,
            descriptor_variances=self.descriptor_variances,
            anchor_offsets=self.anchor_offsets,
            anchor_ids=self.anchor_ids,
            support_offsets=self.support_offsets,
            support_element_ids=self.support_element_ids,
            view_offsets=self.view_offsets,
            view_token_xy=self.view_token_xy,
            view_grid_sizes=self.view_grid_sizes,
            view_descriptors=self.view_descriptors,
            view_quality_scores=self.view_quality_scores,
            view_image_ids_json=np.asarray(json.dumps(list(self.view_image_ids))),
            metadata_json=np.asarray(json.dumps(dict(self.metadata or {}), sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "VfmSurfaceMapletBank":
        with np.load(Path(path), allow_pickle=True) as data:
            return cls(
                maplet_ids=np.asarray(data["maplet_ids"], dtype=np.int64),
                centers=np.asarray(data["centers"], dtype=np.float64),
                normals=np.asarray(data["normals"], dtype=np.float32),
                tangent_frames=np.asarray(data["tangent_frames"], dtype=np.float32),
                extents=np.asarray(data["extents"], dtype=np.float32),
                descriptors=np.asarray(data["descriptors"], dtype=np.float32),
                quality_scores=np.asarray(data["quality_scores"], dtype=np.float32),
                descriptor_variances=np.asarray(data["descriptor_variances"], dtype=np.float32),
                anchor_offsets=np.asarray(data["anchor_offsets"], dtype=np.int64),
                anchor_ids=np.asarray(data["anchor_ids"], dtype=np.int64),
                support_offsets=np.asarray(data["support_offsets"], dtype=np.int64),
                support_element_ids=np.asarray(data["support_element_ids"], dtype=np.int64),
                view_offsets=np.asarray(data["view_offsets"], dtype=np.int64),
                view_image_ids=tuple(json.loads(str(np.asarray(data["view_image_ids_json"]).item()))),
                view_token_xy=np.asarray(data["view_token_xy"], dtype=np.float32),
                view_grid_sizes=np.asarray(data["view_grid_sizes"], dtype=np.int32),
                view_descriptors=np.asarray(data["view_descriptors"], dtype=np.float32),
                view_quality_scores=np.asarray(data["view_quality_scores"], dtype=np.float32),
                metadata=(
                    json.loads(str(np.asarray(data["metadata_json"]).item()))
                    if "metadata_json" in data
                    else {}
                ),
            )


def _ragged_row(offsets: np.ndarray, values: np.ndarray, row: int) -> np.ndarray:
    return values[int(offsets[row]) : int(offsets[row + 1])]


def _weighted_support_iou(
    left_ids: np.ndarray,
    left_weights: np.ndarray,
    right_ids: np.ndarray,
    right_weights: np.ndarray,
) -> float:
    left_ids_arr = np.asarray(left_ids, dtype=np.int64).reshape(-1)
    right_ids_arr = np.asarray(right_ids, dtype=np.int64).reshape(-1)
    left_weights_arr = np.asarray(left_weights, dtype=np.float32).reshape(-1)
    right_weights_arr = np.asarray(right_weights, dtype=np.float32).reshape(-1)
    if (
        left_ids_arr.shape != left_weights_arr.shape
        or right_ids_arr.shape != right_weights_arr.shape
    ):
        raise ValueError("weighted support IDs and weights must have matching shapes")
    left_sorted = left_ids_arr.size < 2 or bool(np.all(np.diff(left_ids_arr) > 0))
    right_sorted = right_ids_arr.size < 2 or bool(np.all(np.diff(right_ids_arr) > 0))
    if left_sorted and right_sorted:
        _shared, left_rows, right_rows = np.intersect1d(
            left_ids_arr,
            right_ids_arr,
            assume_unique=True,
            return_indices=True,
        )
        intersection = float(
            np.minimum(
                left_weights_arr[left_rows],
                right_weights_arr[right_rows],
            ).sum(dtype=np.float64)
        )
        union = (
            float(left_weights_arr.sum(dtype=np.float64))
            + float(right_weights_arr.sum(dtype=np.float64))
            - intersection
        )
        return float(intersection / max(union, 1e-12))
    left = {
        int(key): float(value)
        for key, value in zip(left_ids_arr.tolist(), left_weights_arr.tolist())
    }
    right = {
        int(key): float(value)
        for key, value in zip(right_ids_arr.tolist(), right_weights_arr.tolist())
    }
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    intersection = sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    union = sum(max(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    return float(intersection / max(union, 1e-12))


def _assign_observations_to_maplets(
    maplets: Vfm2DgsAnchorMap,
    observations: Vfm2DgsObservationBank,
    config: SurfaceMapletBuildConfig,
) -> np.ndarray:
    assignment = np.full((len(observations),), -1, dtype=np.int64)
    if len(maplets) == 0 or len(observations) == 0:
        return assignment
    tree = cKDTree(maplets.centers)
    neighbor_count = min(int(config.assignment_neighbor_count), len(maplets))
    distances, neighbors = tree.query(
        observations.centers,
        k=neighbor_count,
        distance_upper_bound=float(config.assignment_max_center_distance),
    )
    if neighbor_count == 1:
        distances = np.asarray(distances)[:, None]
        neighbors = np.asarray(neighbors)[:, None]
    for obs_row in range(len(observations)):
        obs_ids = _ragged_row(observations.support_offsets, observations.element_ids, obs_row)
        obs_weights = _ragged_row(observations.support_offsets, observations.element_weights, obs_row)
        best_row = -1
        best_score = -np.inf
        for distance, candidate in zip(distances[obs_row], neighbors[obs_row]):
            candidate = int(candidate)
            if not np.isfinite(distance) or not 0 <= candidate < len(maplets):
                continue
            support_ids = _ragged_row(maplets.support_offsets, maplets.support_element_ids, candidate)
            support_weights = _ragged_row(maplets.support_offsets, maplets.support_weights, candidate)
            iou = _weighted_support_iou(obs_ids, obs_weights, support_ids, support_weights)
            normal_cosine = float(np.dot(observations.normals[obs_row], maplets.normals[candidate]))
            distance_score = np.exp(-float(distance) / max(float(config.assignment_max_center_distance), 1e-8))
            score = 2.0 * iou + 0.25 * normal_cosine + 0.25 * distance_score
            if iou >= float(config.assignment_min_surface_iou) and score > best_score:
                best_score = score
                best_row = candidate
        assignment[obs_row] = int(best_row)
    return assignment


def _maplet_frame(normal: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    normal = _normalize_vector(normal)
    covariance = np.asarray(covariance, dtype=np.float64).reshape(3, 3)
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    tangent = vectors[:, int(np.argmax(values))]
    tangent = tangent - normal * float(np.dot(tangent, normal))
    if float(np.linalg.norm(tangent)) <= 1e-8:
        reference = np.asarray([1.0, 0.0, 0.0]) if abs(float(normal[0])) < 0.8 else np.asarray([0.0, 1.0, 0.0])
        tangent = np.cross(normal, reference)
    tangent = _normalize_vector(tangent)
    bitangent = _normalize_vector(np.cross(normal, tangent))
    tangent = _normalize_vector(np.cross(bitangent, normal))
    return np.stack([tangent, bitangent, normal], axis=0).astype(np.float32)


def _maplet_extent(
    center: np.ndarray,
    frame: np.ndarray,
    surface_elements: SurfaceElementMap,
    support_ids: np.ndarray,
    row_by_id: Mapping[int, int] | None = None,
) -> np.ndarray:
    element_row_by_id = (
        {
            int(value): int(row)
            for row, value in enumerate(surface_elements.element_ids.tolist())
        }
        if row_by_id is None
        else row_by_id
    )
    rows = [
        element_row_by_id[int(value)]
        for value in support_ids.tolist()
        if int(value) in element_row_by_id
    ]
    if not rows:
        return np.zeros((3,), dtype=np.float32)
    offsets = surface_elements.centers[np.asarray(rows, dtype=np.int64)] - np.asarray(center, dtype=np.float64)
    local = offsets @ np.asarray(frame, dtype=np.float64).T
    return np.max(np.abs(local), axis=0).astype(np.float32)


def _element_view_evidence(
    assignments: np.ndarray,
    observations: Vfm2DgsObservationBank,
    config: SurfaceMapletBuildConfig,
) -> dict[int, dict[int, dict[str, float]]]:
    evidence: dict[int, dict[int, dict[str, float]]] = {}
    for obs_row, maplet_row in enumerate(assignments.tolist()):
        if int(maplet_row) < 0:
            continue
        start, end = int(observations.support_offsets[obs_row]), int(observations.support_offsets[obs_row + 1])
        quality = float(observations.quality_scores[obs_row])
        image_id = str(observations.image_ids[obs_row])
        maplet_evidence = evidence.setdefault(int(maplet_row), {})
        for element_id, weight in zip(
            observations.element_ids[start:end].tolist(),
            observations.element_weights[start:end].tolist(),
        ):
            if float(weight) < float(config.observation_min_element_weight):
                continue
            view_weights = maplet_evidence.setdefault(int(element_id), {})
            view_weights[image_id] = max(view_weights.get(image_id, 0.0), float(weight) * quality)
    return evidence


def _select_spatial_anchors(
    rows: np.ndarray,
    centers: np.ndarray,
    quality: np.ndarray,
    max_count: int,
    min_separation: float,
) -> np.ndarray:
    candidate_rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if candidate_rows.size <= int(max_count) and float(min_separation) <= 0.0:
        order = np.argsort(-quality[candidate_rows], kind="mergesort")
        return candidate_rows[order]
    selected: list[int] = []
    remaining = set(int(value) for value in candidate_rows.tolist())
    while remaining and len(selected) < int(max_count):
        if not selected:
            choice = min(remaining, key=lambda row: (-float(quality[row]), row))
        else:
            remaining_rows = np.asarray(sorted(remaining), dtype=np.int64)
            distances = np.linalg.norm(
                centers[remaining_rows, None, :] - centers[np.asarray(selected), :][None, :, :],
                axis=2,
            )
            minimum_distance = np.min(distances, axis=1)
            valid = minimum_distance >= float(min_separation)
            if not np.any(valid):
                break
            score = minimum_distance * np.sqrt(np.maximum(quality[remaining_rows], 1e-12))
            score[~valid] = -np.inf
            choice = int(remaining_rows[int(np.argmax(score))])
        selected.append(int(choice))
        remaining.remove(int(choice))
    return np.asarray(selected, dtype=np.int64)


def _project_anchor(
    xyz: np.ndarray,
    pose_w2c: np.ndarray,
    camera,
) -> tuple[np.ndarray, float] | None:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    camera_xyz = pose[:3, :3] @ np.asarray(xyz, dtype=np.float64) + pose[:3, 3]
    depth = float(camera_xyz[2])
    if depth <= 1e-6:
        return None
    matrix = _intrinsic_matrix(camera, int(camera.width), int(camera.height))
    pixel = matrix @ camera_xyz
    xy = pixel[:2] / pixel[2]
    if not (0.0 <= float(xy[0]) < float(camera.width) and 0.0 <= float(xy[1]) < float(camera.height)):
        return None
    return xy.astype(np.float32), depth


def build_track_free_surface_map(
    surface_elements: SurfaceElementMap,
    region_map: Vfm2DgsAnchorMap,
    observation_bank: Vfm2DgsObservationBank,
    radio_final_feature_maps: Mapping[str, np.ndarray],
    pose_w2c_by_image: Mapping[str, np.ndarray],
    camera_by_image: Mapping[str, object],
    primitive_quality: TwoDGSPrimitiveQuality | None = None,
    build_config: SurfaceMapletBuildConfig = SurfaceMapletBuildConfig(),
    region_config: RadioFinalRegionConfig = RadioFinalRegionConfig(),
    descriptor_space_metadata: Mapping[str, object] | None = None,
) -> tuple[VfmSurfaceMapletBank, StableSurfaceAnchorMap, dict[str, object]]:
    """Construct the two-level production map from 2DGS/VFM observations."""

    if observation_bank.features.ndim != 2:
        raise ValueError("observation bank must contain RADIO-final descriptors")
    feature_dim = int(observation_bank.features.shape[1])
    for image_id, feature_map in radio_final_feature_maps.items():
        fmap = np.asarray(feature_map)
        if fmap.ndim == 4 and int(fmap.shape[0]) == 1:
            fmap = fmap[0]
        if fmap.ndim != 3 or int(fmap.shape[0]) != feature_dim:
            raise ValueError(f"RADIO-final feature map for {image_id} has an incompatible shape")
    max_parent = int(np.max(surface_elements.parent_gaussian_indices, initial=-1))
    quality_fields = primitive_quality or TwoDGSPrimitiveQuality.neutral(max_parent + 1)
    if len(quality_fields) <= max_parent:
        raise ValueError("primitive quality table does not cover every 2DGS parent primitive")

    assignments = _assign_observations_to_maplets(region_map, observation_bank, build_config)
    evidence = _element_view_evidence(assignments, observation_bank, build_config)
    element_row_by_id = {
        int(value): int(row) for row, value in enumerate(surface_elements.element_ids.tolist())
    }

    contextual_descriptors = np.asarray(observation_bank.features, dtype=np.float32).copy()
    grid_size_by_image: dict[str, tuple[int, int]] = {}
    rows_by_image: dict[str, list[int]] = {}
    for row, image_id in enumerate(observation_bank.image_ids):
        rows_by_image.setdefault(str(image_id), []).append(int(row))
    for image_id, rows in rows_by_image.items():
        if image_id not in radio_final_feature_maps:
            continue
        fmap = np.asarray(radio_final_feature_maps[image_id], dtype=np.float32)
        if fmap.ndim == 4:
            fmap = fmap[0]
        grid_size_by_image[image_id] = (int(fmap.shape[2]), int(fmap.shape[1]))
        contextual_descriptors[np.asarray(rows, dtype=np.int64)] = encode_radio_final_regions(
            fmap,
            observation_bank.token_xy[np.asarray(rows, dtype=np.int64)],
            region_config,
        )
    contextual_descriptors = _normalize_rows(contextual_descriptors)

    selected_by_maplet: dict[int, np.ndarray] = {}
    candidate_quality_by_pair: dict[tuple[int, int], float] = {}
    for maplet_row in range(len(region_map)):
        support_ids = _ragged_row(region_map.support_offsets, region_map.support_element_ids, maplet_row)
        support_weights = _ragged_row(region_map.support_offsets, region_map.support_weights, maplet_row)
        support_weight_by_id = {
            int(element_id): float(weight)
            for element_id, weight in zip(support_ids.tolist(), support_weights.tolist())
        }
        candidate_rows: list[int] = []
        element_quality = np.zeros((len(surface_elements),), dtype=np.float32)
        for element_id in support_ids.tolist():
            element_id = int(element_id)
            if element_id not in element_row_by_id:
                continue
            row = element_row_by_id[element_id]
            parent = int(surface_elements.parent_gaussian_indices[row])
            view_weights = evidence.get(maplet_row, {}).get(element_id, {})
            if len(view_weights) < int(build_config.min_anchor_views):
                continue
            geometry = float(quality_fields.geometry_confidence[parent])
            primitive_class = int(quality_fields.primitive_class[parent])
            if geometry < float(build_config.min_geometry_confidence):
                continue
            if (
                build_config.allowed_primitive_classes is not None
                and primitive_class not in set(int(value) for value in build_config.allowed_primitive_classes)
            ):
                continue
            opacity = float(surface_elements.opacity[row])
            if opacity < float(build_config.min_anchor_opacity):
                continue
            radius = float(max(surface_elements.scale1[row], surface_elements.scale2[row]))
            if float(build_config.max_anchor_scale) > 0.0 and radius > float(build_config.max_anchor_scale):
                continue
            normal_cosine = float(np.dot(surface_elements.normals[row], region_map.normals[maplet_row]))
            if normal_cosine < float(build_config.min_anchor_normal_cosine):
                continue
            view_score = float(np.mean(list(view_weights.values())))
            score = (
                np.sqrt(max(support_weight_by_id.get(element_id, 0.0), 1e-8))
                * max(opacity, 1e-4)
                * max(geometry, 1e-4)
                * max(normal_cosine, 0.0)
                * np.sqrt(max(view_score, 1e-8))
                * np.log1p(len(view_weights))
            )
            candidate_rows.append(row)
            element_quality[row] = float(score)
            candidate_quality_by_pair[(maplet_row, element_id)] = float(score)
        chosen_rows = _select_spatial_anchors(
            np.asarray(candidate_rows, dtype=np.int64),
            surface_elements.centers,
            element_quality,
            max_count=int(build_config.max_anchors_per_maplet),
            min_separation=float(build_config.min_anchor_separation),
        )
        if chosen_rows.size >= int(build_config.min_anchors_per_maplet):
            selected_by_maplet[maplet_row] = surface_elements.element_ids[chosen_rows]

    kept_maplet_rows = np.asarray(sorted(selected_by_maplet), dtype=np.int64)
    selected_elements = sorted(
        {
            int(element_id)
            for values in selected_by_maplet.values()
            for element_id in np.asarray(values, dtype=np.int64).tolist()
        }
    )
    selected_set_by_maplet = {
        int(row): set(np.asarray(values, dtype=np.int64).tolist())
        for row, values in selected_by_maplet.items()
    }
    owners_by_element: dict[int, list[int]] = {}
    for maplet_row, element_set in selected_set_by_maplet.items():
        for element_id in element_set:
            owners_by_element.setdefault(int(element_id), []).append(
                int(maplet_row)
            )
    owner_by_element: dict[int, int] = {}
    for element_id in selected_elements:
        owners = owners_by_element[int(element_id)]
        owner_by_element[element_id] = max(
            owners,
            key=lambda row: (candidate_quality_by_pair.get((row, element_id), 0.0), -row),
        )

    anchor_ids: list[int] = []
    owner_ids: list[int] = []
    element_ids: list[int] = []
    parent_ids: list[int] = []
    anchor_xyz: list[np.ndarray] = []
    anchor_normals: list[np.ndarray] = []
    anchor_covariances: list[np.ndarray] = []
    anchor_radii: list[float] = []
    anchor_quality: list[float] = []
    anchor_geometry: list[float] = []
    anchor_opacity: list[float] = []
    observation_offsets = [0]
    observation_image_ids: list[str] = []
    observation_xy: list[np.ndarray] = []
    observation_depth: list[float] = []
    observation_weights: list[float] = []
    for element_id in selected_elements:
        row = element_row_by_id[element_id]
        owner_row = int(owner_by_element[element_id])
        parent = int(surface_elements.parent_gaussian_indices[row])
        combined_views: dict[str, float] = {}
        element_owners = owners_by_element[int(element_id)]
        for maplet_row in element_owners:
            for image_id, weight in evidence.get(maplet_row, {}).get(element_id, {}).items():
                combined_views[image_id] = max(combined_views.get(image_id, 0.0), float(weight))
        valid_observations = []
        for image_id, weight in sorted(combined_views.items()):
            if image_id not in pose_w2c_by_image or image_id not in camera_by_image:
                continue
            projected = _project_anchor(
                surface_elements.centers[row],
                pose_w2c_by_image[image_id],
                camera_by_image[image_id],
            )
            if projected is None:
                continue
            xy, depth = projected
            camera_center = -np.asarray(pose_w2c_by_image[image_id])[:3, :3].T @ np.asarray(
                pose_w2c_by_image[image_id]
            )[:3, 3]
            viewing = _normalize_vector(camera_center - surface_elements.centers[row])
            frontalness = abs(float(np.dot(surface_elements.normals[row], viewing)))
            valid_observations.append((image_id, xy, depth, float(weight) * max(frontalness, 0.05)))
        if len(valid_observations) < int(build_config.min_anchor_views):
            continue
        tangent1 = surface_elements.tangent1[row].astype(np.float64)
        tangent2 = surface_elements.tangent2[row].astype(np.float64)
        normal = surface_elements.normals[row].astype(np.float64)
        covariance = (
            float(surface_elements.scale1[row]) ** 2 * np.outer(tangent1, tangent1)
            + float(surface_elements.scale2[row]) ** 2 * np.outer(tangent2, tangent2)
            + float(build_config.normal_variance_scale) ** 2 * np.outer(normal, normal)
        )
        anchor_ids.append(int(element_id))
        owner_ids.append(int(region_map.anchor_ids[owner_row]))
        element_ids.append(int(element_id))
        parent_ids.append(parent)
        anchor_xyz.append(surface_elements.centers[row])
        anchor_normals.append(surface_elements.normals[row])
        anchor_covariances.append(covariance.astype(np.float32))
        anchor_radii.append(float(max(surface_elements.scale1[row], surface_elements.scale2[row])))
        anchor_quality.append(
            max(
                candidate_quality_by_pair.get((maplet_row, element_id), 0.0)
                for maplet_row in element_owners
            )
        )
        anchor_geometry.append(float(quality_fields.geometry_confidence[parent]))
        anchor_opacity.append(float(surface_elements.opacity[row]))
        for image_id, xy, depth, weight in valid_observations:
            observation_image_ids.append(image_id)
            observation_xy.append(xy)
            observation_depth.append(depth)
            observation_weights.append(weight)
        observation_offsets.append(len(observation_image_ids))

    stable_anchors = StableSurfaceAnchorMap(
        anchor_ids=np.asarray(anchor_ids, dtype=np.int64),
        owner_maplet_ids=np.asarray(owner_ids, dtype=np.int64),
        surface_element_ids=np.asarray(element_ids, dtype=np.int64),
        parent_primitive_indices=np.asarray(parent_ids, dtype=np.int64),
        xyz=np.asarray(anchor_xyz, dtype=np.float64).reshape(-1, 3),
        normals=np.asarray(anchor_normals, dtype=np.float32).reshape(-1, 3),
        tangent_covariances=np.asarray(anchor_covariances, dtype=np.float32).reshape(-1, 3, 3),
        support_radii=np.asarray(anchor_radii, dtype=np.float32),
        quality_scores=np.asarray(anchor_quality, dtype=np.float32),
        geometry_confidence=np.asarray(anchor_geometry, dtype=np.float32),
        opacity=np.asarray(anchor_opacity, dtype=np.float32),
        observation_offsets=np.asarray(observation_offsets, dtype=np.int64),
        observation_image_ids=tuple(observation_image_ids),
        observation_xy=np.asarray(observation_xy, dtype=np.float32).reshape(-1, 2),
        observation_depth=np.asarray(observation_depth, dtype=np.float32),
        observation_weights=np.asarray(observation_weights, dtype=np.float32),
        metadata={
            "representation": "stable_2dgs_surface_anchor",
            "identity": "surface_element_id",
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "build_config": build_config.to_dict(),
        },
    )
    valid_anchor_ids = set(stable_anchors.anchor_ids.tolist())

    maplet_ids: list[int] = []
    maplet_centers: list[np.ndarray] = []
    maplet_normals: list[np.ndarray] = []
    maplet_frames: list[np.ndarray] = []
    maplet_extents: list[np.ndarray] = []
    maplet_descriptors: list[np.ndarray] = []
    maplet_quality: list[float] = []
    maplet_variances: list[float] = []
    maplet_anchor_offsets = [0]
    maplet_anchor_ids: list[int] = []
    maplet_support_offsets = [0]
    maplet_support_ids: list[int] = []
    view_offsets = [0]
    view_image_ids: list[str] = []
    view_token_xy: list[np.ndarray] = []
    view_grid_sizes: list[tuple[int, int]] = []
    view_descriptors: list[np.ndarray] = []
    view_quality_scores: list[float] = []
    for maplet_row in kept_maplet_rows.tolist():
        member_anchor_ids = [
            int(value)
            for value in selected_by_maplet[int(maplet_row)].tolist()
            if int(value) in valid_anchor_ids
        ]
        if len(member_anchor_ids) < int(build_config.min_anchors_per_maplet):
            continue
        obs_rows = np.flatnonzero(assignments == int(maplet_row))
        per_view: dict[str, list[int]] = {}
        for obs_row in obs_rows.tolist():
            per_view.setdefault(str(observation_bank.image_ids[obs_row]), []).append(int(obs_row))
        aggregated_view_descriptors: list[np.ndarray] = []
        aggregated_view_weights: list[float] = []
        pending_views = []
        for image_id, rows in sorted(per_view.items()):
            weights = np.maximum(observation_bank.descriptor_weights[np.asarray(rows)], 1e-6)
            descriptors = contextual_descriptors[np.asarray(rows)]
            descriptor = np.average(descriptors, axis=0, weights=weights)
            descriptor = _normalize_vector(descriptor).astype(np.float32)
            xy = np.average(observation_bank.token_xy[np.asarray(rows)], axis=0, weights=weights)
            grid_size = grid_size_by_image.get(image_id, (0, 0))
            quality = float(np.mean(observation_bank.quality_scores[np.asarray(rows)]))
            pending_views.append((image_id, xy, grid_size, descriptor, quality))
            aggregated_view_descriptors.append(descriptor)
            aggregated_view_weights.append(max(quality, 1e-6))
        if aggregated_view_descriptors:
            view_matrix = np.stack(aggregated_view_descriptors, axis=0)
            descriptor = np.average(view_matrix, axis=0, weights=np.asarray(aggregated_view_weights))
            descriptor = _normalize_vector(descriptor).astype(np.float32)
            variance = float(np.mean(1.0 - np.clip(view_matrix @ descriptor, -1.0, 1.0)))
        else:
            descriptor = _normalize_vector(region_map.features[maplet_row]).astype(np.float32)
            variance = float(region_map.feature_variances[maplet_row])
        support_ids = _ragged_row(region_map.support_offsets, region_map.support_element_ids, maplet_row)
        frame = _maplet_frame(region_map.normals[maplet_row], region_map.covariances[maplet_row])
        maplet_ids.append(int(region_map.anchor_ids[maplet_row]))
        maplet_centers.append(region_map.centers[maplet_row])
        maplet_normals.append(region_map.normals[maplet_row])
        maplet_frames.append(frame)
        maplet_extents.append(
            _maplet_extent(
                region_map.centers[maplet_row],
                frame,
                surface_elements,
                support_ids,
                row_by_id=element_row_by_id,
            )
        )
        maplet_descriptors.append(descriptor)
        maplet_quality.append(float(region_map.quality_scores[maplet_row]))
        maplet_variances.append(variance)
        maplet_anchor_ids.extend(member_anchor_ids)
        maplet_anchor_offsets.append(len(maplet_anchor_ids))
        maplet_support_ids.extend(int(value) for value in support_ids.tolist())
        maplet_support_offsets.append(len(maplet_support_ids))
        for image_id, xy, grid_size, view_descriptor, quality in pending_views:
            view_image_ids.append(image_id)
            view_token_xy.append(np.asarray(xy, dtype=np.float32))
            view_grid_sizes.append((int(grid_size[0]), int(grid_size[1])))
            view_descriptors.append(view_descriptor)
            view_quality_scores.append(quality)
        view_offsets.append(len(view_image_ids))

    maplet_bank = VfmSurfaceMapletBank(
        maplet_ids=np.asarray(maplet_ids, dtype=np.int64),
        centers=np.asarray(maplet_centers, dtype=np.float64).reshape(-1, 3),
        normals=np.asarray(maplet_normals, dtype=np.float32).reshape(-1, 3),
        tangent_frames=np.asarray(maplet_frames, dtype=np.float32).reshape(-1, 3, 3),
        extents=np.asarray(maplet_extents, dtype=np.float32).reshape(-1, 3),
        descriptors=np.asarray(maplet_descriptors, dtype=np.float32).reshape(-1, feature_dim),
        quality_scores=np.asarray(maplet_quality, dtype=np.float32),
        descriptor_variances=np.asarray(maplet_variances, dtype=np.float32),
        anchor_offsets=np.asarray(maplet_anchor_offsets, dtype=np.int64),
        anchor_ids=np.asarray(maplet_anchor_ids, dtype=np.int64),
        support_offsets=np.asarray(maplet_support_offsets, dtype=np.int64),
        support_element_ids=np.asarray(maplet_support_ids, dtype=np.int64),
        view_offsets=np.asarray(view_offsets, dtype=np.int64),
        view_image_ids=tuple(view_image_ids),
        view_token_xy=np.asarray(view_token_xy, dtype=np.float32).reshape(-1, 2),
        view_grid_sizes=np.asarray(view_grid_sizes, dtype=np.int32).reshape(-1, 2),
        view_descriptors=np.asarray(view_descriptors, dtype=np.float32).reshape(-1, feature_dim),
        view_quality_scores=np.asarray(view_quality_scores, dtype=np.float32),
        metadata={
            "representation": "vfm_aligned_2dgs_surface_maplet",
            "vfm_layer": "radio_final",
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "region_config": region_config.to_dict(),
            "build_config": build_config.to_dict(),
            "descriptor_space": dict(descriptor_space_metadata or {}),
        },
    )
    anchors_per_maplet = np.diff(maplet_bank.anchor_offsets)
    summary = {
        "representation": "track_free_vfm_2dgs_surface_map",
        "surface_element_count": int(len(surface_elements)),
        "input_region_count": int(len(region_map)),
        "observation_count": int(len(observation_bank)),
        "assigned_observation_count": int(np.sum(assignments >= 0)),
        "maplet_count": int(len(maplet_bank)),
        "stable_anchor_count": int(len(stable_anchors)),
        "stable_anchor_observation_count": int(len(stable_anchors.observation_image_ids)),
        "anchors_per_maplet": {
            "min": int(np.min(anchors_per_maplet)) if anchors_per_maplet.size else 0,
            "median": float(np.median(anchors_per_maplet)) if anchors_per_maplet.size else 0.0,
            "mean": float(np.mean(anchors_per_maplet)) if anchors_per_maplet.size else 0.0,
            "max": int(np.max(anchors_per_maplet)) if anchors_per_maplet.size else 0,
        },
        "radio_intermediate_used": False,
        "sfm_points_used": False,
        "sfm_tracks_used": False,
    }
    return maplet_bank, stable_anchors, summary
