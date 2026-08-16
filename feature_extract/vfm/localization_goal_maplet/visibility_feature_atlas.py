"""Canonical RADIO surface-feature grids for visibility-chart reranking.

The grids are synthesized from the frozen canonical 3DGS surface field and
frozen contributor visibility.  They contain no mapping RGB or mapping-image
descriptor.  Their role is to add appearance/layout evidence to the physical
child co-visibility proposal, not to serve as final poses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .canonical_field import CanonicalSurfaceField
from .lineage import arrays_sha256
from .physical_map import GoalMapletPhysicalMap


SCHEMA = "goal_maplet_canonical_feature_visibility_pose_atlas_v1"
SCORE_SEMANTICS = "fixed_grid_canonical_radio_cosine_missing_floor_v1"


def pool_normalized_feature_grid(
    feature: np.ndarray,
    valid: np.ndarray,
    *,
    output_rows: int,
    output_cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool a normalized [C,H,W] field into fixed equal cells."""

    value = np.asarray(feature, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    if value.ndim != 3 or mask.shape != value.shape[1:] or np.any(~np.isfinite(value)):
        raise ValueError("invalid feature grid")
    channels, height, width = value.shape
    if height % int(output_rows) or width % int(output_cols):
        raise ValueError("output grid must divide the source feature grid")
    row_factor, col_factor = height // int(output_rows), width // int(output_cols)
    source = value.reshape(
        channels, int(output_rows), row_factor, int(output_cols), col_factor
    )
    source_mask = mask.reshape(
        int(output_rows), row_factor, int(output_cols), col_factor
    )
    weighted = source * source_mask[None]
    count = np.sum(source_mask, axis=(1, 3))
    pooled = np.sum(weighted, axis=(2, 4)) / np.maximum(count[None], 1.0)
    pooled_valid = count > 0
    norm = np.linalg.norm(pooled, axis=0)
    pooled[:, pooled_valid] /= np.maximum(norm[pooled_valid], 1e-8)
    pooled[:, ~pooled_valid] = 0.0
    return pooled.astype(np.float32), pooled_valid


@dataclass(frozen=True)
class CanonicalFeatureVisibilityPoseAtlas:
    poses_w2c: np.ndarray
    features: np.ndarray
    valid: np.ndarray
    physical_map_sha256: str
    canonical_field_sha256: str
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        pose = np.asarray(self.poses_w2c, dtype=np.float64)
        feature = np.asarray(self.features, dtype=np.float16)
        valid = np.asarray(self.valid, dtype=bool)
        if (
            pose.ndim != 3 or pose.shape[1:] != (4, 4)
            or feature.ndim != 4 or feature.shape[0] != pose.shape[0]
            or valid.shape != (feature.shape[0], feature.shape[2], feature.shape[3])
            or np.any(~np.isfinite(pose)) or np.any(~np.isfinite(feature))
        ):
            raise ValueError("invalid canonical feature visibility atlas")
        metadata = dict(self.metadata or {})
        if metadata.get("artifact_type", SCHEMA) != SCHEMA:
            raise ValueError("not a canonical feature visibility atlas")
        if any(bool(metadata.get(key, False)) for key in (
            "stores_mapping_rgb", "stores_mapping_image_descriptors",
            "uses_query_pose", "uses_query_ground_truth", "uses_pnp", "uses_alike",
        )):
            raise ValueError("canonical feature atlas violates its frozen-map contract")
        object.__setattr__(self, "poses_w2c", pose)
        object.__setattr__(self, "features", feature)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "metadata", metadata)

    @property
    def content_sha256(self) -> str:
        return arrays_sha256({
            "poses_w2c": self.poses_w2c,
            "features": self.features,
            "valid": self.valid.astype(np.uint8),
        })

    @property
    def view_count(self) -> int:
        return int(self.poses_w2c.shape[0])

    def save_npz(self, path: Path) -> None:
        metadata = {
            **dict(self.metadata or {}),
            "artifact_type": SCHEMA,
            "score_semantics": SCORE_SEMANTICS,
            "content_sha256": self.content_sha256,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            Path(path),
            poses_w2c=self.poses_w2c,
            features=self.features,
            valid=self.valid.astype(np.uint8),
            physical_map_sha256=np.asarray(self.physical_map_sha256),
            canonical_field_sha256=np.asarray(self.canonical_field_sha256),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "CanonicalFeatureVisibilityPoseAtlas":
        with np.load(Path(path), allow_pickle=False) as data:
            value = cls(
                poses_w2c=data["poses_w2c"],
                features=data["features"],
                valid=data["valid"],
                physical_map_sha256=str(data["physical_map_sha256"].item()),
                canonical_field_sha256=str(data["canonical_field_sha256"].item()),
                metadata=json.loads(str(data["metadata_json"].item())),
            )
        if value.metadata.get("score_semantics") != SCORE_SEMANTICS:
            raise ValueError("canonical feature atlas score semantics differ")
        if value.metadata.get("content_sha256") != value.content_sha256:
            raise ValueError("canonical feature atlas content hash mismatch")
        return value


def build_canonical_feature_visibility_pose_atlas(
    physical: GoalMapletPhysicalMap,
    field: CanonicalSurfaceField,
    contributor_paths: Sequence[Path],
    *,
    output_rows: int = 9,
    output_cols: int = 16,
    metadata: Mapping[str, object] | None = None,
) -> CanonicalFeatureVisibilityPoseAtlas:
    if field.physical_map_sha256 != physical.content_sha256:
        raise ValueError("canonical field and physical map differ")
    paths = tuple(Path(path) for path in contributor_paths)
    if not paths:
        raise ValueError("canonical feature atlas needs contributors")
    primitive_ids = np.asarray(physical.primitive_ids, dtype=np.int64)
    row_by_id = np.full((int(np.max(primitive_ids)) + 1,), -1, dtype=np.int32)
    row_by_id[primitive_ids] = np.arange(primitive_ids.size, dtype=np.int32)
    field_row_by_primitive = np.full((primitive_ids.size,), -1, dtype=np.int32)
    field_row_by_primitive[np.asarray(field.primitive_rows, dtype=np.int64)] = np.arange(
        field.primitive_rows.size, dtype=np.int32
    )
    codes = np.asarray(field.codes, dtype=np.float32)
    poses: list[np.ndarray] = []
    pooled_features: list[np.ndarray] = []
    pooled_valid: list[np.ndarray] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            ids = np.asarray(data["topk_ids"], dtype=np.int64)
            weights = np.asarray(data["topk_weights"], dtype=np.float32)
            pose = np.asarray(data["pose_w2c"], dtype=np.float64)
        height, width, slots = ids.shape
        if weights.shape != ids.shape or height % 36 or width % 64:
            raise ValueError(f"contributor grid cannot align to RADIO tokens: {path}")
        y = np.arange(36) * height // 36 + (height // 36) // 2
        x = np.arange(64) * width // 64 + (width // 64) // 2
        sampled_ids = ids[y[:, None], x[None], :]
        sampled_weights = weights[y[:, None], x[None], :]
        safe_id = np.clip(sampled_ids, 0, row_by_id.size - 1)
        primitive_row = row_by_id[safe_id]
        valid_id = (sampled_ids >= 0) & (sampled_ids < row_by_id.size) & (primitive_row >= 0)
        safe_primitive = np.maximum(primitive_row, 0)
        field_row = field_row_by_primitive[safe_primitive]
        valid_code = valid_id & (field_row >= 0) & (sampled_weights > 0.0)
        token_feature = np.zeros((36, 64, field.feature_dim), dtype=np.float32)
        token_mass = np.zeros((36, 64), dtype=np.float32)
        for slot in range(slots):
            local = valid_code[..., slot]
            if np.any(local):
                value = sampled_weights[..., slot] * local
                token_feature += value[..., None] * codes[np.maximum(field_row[..., slot], 0)]
                token_mass += value
        token_valid = token_mass > 0.0
        token_feature[token_valid] /= token_mass[token_valid, None]
        norm = np.linalg.norm(token_feature, axis=2)
        token_feature[token_valid] /= np.maximum(norm[token_valid, None], 1e-8)
        pooled, valid = pool_normalized_feature_grid(
            token_feature.transpose(2, 0, 1), token_valid,
            output_rows=int(output_rows), output_cols=int(output_cols),
        )
        poses.append(pose)
        pooled_features.append(pooled.astype(np.float16))
        pooled_valid.append(valid)
    return CanonicalFeatureVisibilityPoseAtlas(
        poses_w2c=np.asarray(poses, dtype=np.float64),
        features=np.stack(pooled_features),
        valid=np.stack(pooled_valid),
        physical_map_sha256=physical.content_sha256,
        canonical_field_sha256=field.content_sha256,
        metadata={
            "artifact_type": SCHEMA,
            "score_semantics": SCORE_SEMANTICS,
            "stores_mapping_rgb": False,
            "stores_mapping_image_descriptors": False,
            "uses_query_pose": False,
            "uses_query_ground_truth": False,
            "uses_pnp": False,
            "uses_alike": False,
            "feature_source": "frozen_canonical_3dgs_surface_field",
            "mapping_visibility_source": "frozen_3dgs_contributors",
            "source_contributor_count": len(paths),
            **dict(metadata or {}),
        },
    )


def score_canonical_feature_visibility_pose_atlas(
    atlas: CanonicalFeatureVisibilityPoseAtlas,
    query_feature: np.ndarray,
) -> np.ndarray:
    pooled, query_valid = pool_normalized_feature_grid(
        query_feature,
        np.ones(np.asarray(query_feature).shape[1:], dtype=bool),
        output_rows=atlas.features.shape[2], output_cols=atlas.features.shape[3],
    )
    grid_count = int(atlas.features.shape[2] * atlas.features.shape[3])
    atlas_feature = np.asarray(atlas.features, dtype=np.float32)
    cosine = np.einsum("vcrs,crs->vrs", atlas_feature, pooled, optimize=True)
    comparable = atlas.valid & query_valid[None]
    score = np.sum(np.where(comparable, cosine, -1.0), axis=(1, 2)) / float(grid_count)
    return np.clip(score, -1.0, 1.0).astype(np.float32)
