"""Feature-only continuous 2DGS surface field for VFM pose alignment.

This is deliberately not an anchor map.  A row is a retained 2DGS surfel and
stores one robust RADIO-final feature estimate plus metric geometry and
uncertainty.  It has no image identity, keypoint identity, descriptor list, or
2D-to-3D correspondence identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np


_FORBIDDEN_TRUE_METADATA = (
    "uses_mapping_rgb_at_inference",
    "stores_mapping_rgb",
    "stores_mapping_image_paths",
    "uses_pairwise_image_matching",
    "uses_alike_descriptors",
    "uses_radio_intermediate",
    "uses_sfm_points",
    "uses_sfm_tracks",
    "uses_stable_anchor_identity",
)


def _normalize_rows(value: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), eps)


@dataclass(frozen=True)
class SurfaceFeatureField:
    """Sparse feature field attached directly to clean 2DGS surfels."""

    source_indices: np.ndarray
    centers: np.ndarray
    normals: np.ndarray
    tangent1: np.ndarray
    tangent2: np.ndarray
    scale1: np.ndarray
    scale2: np.ndarray
    opacity: np.ndarray
    features: np.ndarray
    uncertainty: np.ndarray
    confidence: np.ndarray
    support_weight: np.ndarray
    support_count: np.ndarray
    owner_maplet_ids: np.ndarray
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        source_indices = np.asarray(self.source_indices, dtype=np.int64).reshape(-1)
        count = int(source_indices.size)
        if np.unique(source_indices).size != count or np.any(source_indices < 0):
            raise ValueError("source_indices must be unique and non-negative")
        object.__setattr__(self, "source_indices", source_indices)
        for name in ("centers", "normals", "tangent1", "tangent2"):
            value = np.asarray(
                getattr(self, name),
                dtype=np.float64 if name == "centers" else np.float32,
            )
            if value.shape != (count, 3):
                raise ValueError(f"{name} must have shape (N, 3)")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "normals", _normalize_rows(self.normals))
        object.__setattr__(self, "tangent1", _normalize_rows(self.tangent1))
        object.__setattr__(self, "tangent2", _normalize_rows(self.tangent2))
        for name in (
            "scale1",
            "scale2",
            "opacity",
            "uncertainty",
            "confidence",
            "support_weight",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float32).reshape(-1)
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape (N,)")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        support_count = np.asarray(self.support_count, dtype=np.int32).reshape(-1)
        owner_maplet_ids = np.asarray(self.owner_maplet_ids, dtype=np.int64).reshape(-1)
        if support_count.shape != (count,) or owner_maplet_ids.shape != (count,):
            raise ValueError("support_count and owner_maplet_ids must have shape (N,)")
        object.__setattr__(self, "support_count", support_count)
        object.__setattr__(self, "owner_maplet_ids", owner_maplet_ids)
        features = np.asarray(self.features, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != count:
            raise ValueError("features must have shape (N, C)")
        object.__setattr__(self, "features", _normalize_rows(features))
        metadata = dict(self.metadata or {})
        if metadata.get("artifact_type") != "radio_final_2dgs_surface_feature_field":
            raise ValueError("invalid surface feature field artifact type")
        if metadata.get("vfm_layer") != "radio_final":
            raise ValueError("surface feature field must use RADIO final")
        for name in _FORBIDDEN_TRUE_METADATA:
            if bool(metadata.get(name, False)):
                raise ValueError(f"surface feature field violates contract: {name}")
        object.__setattr__(self, "metadata", metadata)

    def __len__(self) -> int:
        return int(self.source_indices.size)

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    def save_npz(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            Path(path),
            source_indices=self.source_indices,
            centers=self.centers.astype(np.float32),
            normals=self.normals.astype(np.float16),
            tangent1=self.tangent1.astype(np.float16),
            tangent2=self.tangent2.astype(np.float16),
            scale1=self.scale1.astype(np.float16),
            scale2=self.scale2.astype(np.float16),
            opacity=self.opacity.astype(np.float16),
            features=self.features.astype(np.float16),
            uncertainty=self.uncertainty.astype(np.float16),
            confidence=self.confidence.astype(np.float16),
            support_weight=self.support_weight.astype(np.float32),
            support_count=self.support_count.astype(np.int32),
            owner_maplet_ids=self.owner_maplet_ids.astype(np.int64),
            metadata_json=np.asarray(json.dumps(dict(self.metadata), sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "SurfaceFeatureField":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                source_indices=np.asarray(data["source_indices"], dtype=np.int64),
                centers=np.asarray(data["centers"], dtype=np.float64),
                normals=np.asarray(data["normals"], dtype=np.float32),
                tangent1=np.asarray(data["tangent1"], dtype=np.float32),
                tangent2=np.asarray(data["tangent2"], dtype=np.float32),
                scale1=np.asarray(data["scale1"], dtype=np.float32),
                scale2=np.asarray(data["scale2"], dtype=np.float32),
                opacity=np.asarray(data["opacity"], dtype=np.float32),
                features=np.asarray(data["features"], dtype=np.float32),
                uncertainty=np.asarray(data["uncertainty"], dtype=np.float32),
                confidence=np.asarray(data["confidence"], dtype=np.float32),
                support_weight=np.asarray(data["support_weight"], dtype=np.float32),
                support_count=np.asarray(data["support_count"], dtype=np.int32),
                owner_maplet_ids=np.asarray(data["owner_maplet_ids"], dtype=np.int64),
                metadata=json.loads(str(np.asarray(data["metadata_json"]).item())),
            )
