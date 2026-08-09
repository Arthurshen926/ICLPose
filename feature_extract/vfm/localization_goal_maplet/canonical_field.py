"""One canonical RADIO-final code per observed clean 2DGS primitive."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from feature_extract.vfm.localization.surface_feature_field import SurfaceFeatureField

from .lineage import arrays_sha256, validate_deployment_metadata
from .physical_map import GoalMapletPhysicalMap


SCHEMA = "goal_maplet_canonical_surface_field_v1"


def _normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    # Do not renormalize an already-normalized float32 field.  A second
    # division changes a handful of low bits and therefore breaks the content
    # hash across save/load even though the semantic value is unchanged.
    if np.all(np.abs(norm - 1.0) <= 1e-6):
        return array.copy()
    return array / np.maximum(norm, 1e-8)


@dataclass(frozen=True)
class CanonicalSurfaceField:
    primitive_rows: np.ndarray
    codes: np.ndarray
    confidence: np.ndarray
    uncertainty: np.ndarray
    physical_map_sha256: str
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        rows = np.asarray(self.primitive_rows, dtype=np.int64).reshape(-1)
        codes = np.asarray(self.codes, dtype=np.float32)
        confidence = np.asarray(self.confidence, dtype=np.float32).reshape(-1)
        uncertainty = np.asarray(self.uncertainty, dtype=np.float32).reshape(-1)
        if (
            np.unique(rows).size != rows.size
            or np.any(rows < 0)
            or codes.ndim != 2
            or codes.shape[0] != rows.size
            or confidence.shape != rows.shape
            or uncertainty.shape != rows.shape
        ):
            raise ValueError("invalid canonical surface field")
        object.__setattr__(self, "primitive_rows", rows)
        object.__setattr__(self, "codes", _normalize(codes))
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "uncertainty", uncertainty)
        metadata = dict(self.metadata or {})
        if metadata.get("artifact_type", SCHEMA) != SCHEMA:
            raise ValueError("not a Goal-Maplet canonical field")
        validate_deployment_metadata(metadata)
        if int(metadata.get("stored_downstream_embedding_count", 0)) != 0:
            raise ValueError("canonical field cannot store downstream embeddings")
        object.__setattr__(self, "metadata", metadata)
        declared_hash = str(metadata.get("content_sha256", ""))
        if declared_hash and declared_hash != self.content_sha256:
            raise ValueError(
                "canonical field content hash mismatch: "
                f"declared={declared_hash} actual={self.content_sha256}"
            )

    @property
    def feature_dim(self) -> int:
        return int(self.codes.shape[1])

    @property
    def content_sha256(self) -> str:
        return arrays_sha256({
            "primitive_rows": self.primitive_rows,
            "codes": self.codes,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
        })

    def save_npz(self, path: Path) -> None:
        metadata = {**dict(self.metadata or {}), "artifact_type": SCHEMA, "content_sha256": self.content_sha256}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            Path(path),
            primitive_rows=self.primitive_rows,
            # Preserve the canonical normalized code exactly; downstream
            # readouts must not depend on save/load quantization.
            codes=self.codes.astype(np.float32),
            confidence=self.confidence.astype(np.float32),
            uncertainty=self.uncertainty.astype(np.float32),
            physical_map_sha256=np.asarray(self.physical_map_sha256),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "CanonicalSurfaceField":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                primitive_rows=np.asarray(data["primitive_rows"], dtype=np.int64),
                codes=np.asarray(data["codes"], dtype=np.float32),
                confidence=np.asarray(data["confidence"], dtype=np.float32),
                uncertainty=np.asarray(data["uncertainty"], dtype=np.float32),
                physical_map_sha256=str(np.asarray(data["physical_map_sha256"]).item()),
                metadata=json.loads(str(np.asarray(data["metadata_json"]).item())),
            )


@dataclass(frozen=True)
class CanonicalFieldReadout:
    parent_descriptors: np.ndarray
    parent_coverage: np.ndarray
    child_descriptors: np.ndarray
    child_coverage: np.ndarray


class CanonicalFieldFusionAccumulator:
    """View-balanced exact-contributor fusion without storing observations."""

    def __init__(self, primitive_count: int, feature_dim: int) -> None:
        self.feature_sum = np.zeros((int(primitive_count), int(feature_dim)), dtype=np.float32)
        self.weight_sum = np.zeros((int(primitive_count),), dtype=np.float32)
        self.view_count = np.zeros((int(primitive_count),), dtype=np.int32)

    def add_view(
        self,
        primitive_rows: np.ndarray,
        descriptors: np.ndarray,
        contribution_mass: np.ndarray,
        observation_quality: np.ndarray | None = None,
    ) -> None:
        rows = np.asarray(primitive_rows, dtype=np.int64).reshape(-1)
        feature = _normalize(np.asarray(descriptors, dtype=np.float32))
        mass = np.asarray(contribution_mass, dtype=np.float32).reshape(-1)
        if feature.shape[0] != rows.size or mass.shape != rows.shape or np.unique(rows).size != rows.size:
            raise ValueError("one fusion view must contain unique aligned primitive rows")
        if np.any((rows < 0) | (rows >= self.feature_sum.shape[0])) or np.any(mass <= 0.0):
            raise ValueError("invalid canonical fusion observation")
        quality = (
            np.ones(rows.shape, dtype=np.float32)
            if observation_quality is None
            else np.asarray(observation_quality, dtype=np.float32).reshape(-1)
        )
        if quality.shape != rows.shape or np.any(~np.isfinite(quality)) or np.any(quality <= 0.0):
            raise ValueError("invalid canonical fusion observation quality")
        # Cap projected footprint dominance: one large close-up surfel is one
        # view observation, not hundreds of independent descriptors.  An
        # optional offline-teacher quality multiplies the view weight; it may
        # downweight an inconsistent observation but never creates another
        # deployed feature or removes physical support.
        weight = np.clip(np.sqrt(mass), 0.25, 4.0).astype(np.float32) * quality
        self.feature_sum[rows] += weight[:, None] * feature
        self.weight_sum[rows] += weight
        self.view_count[rows] += 1

    def finalize(
        self,
        physical_map: GoalMapletPhysicalMap,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> CanonicalSurfaceField:
        rows = np.flatnonzero(self.weight_sum > 0.0)
        resultant = np.linalg.norm(self.feature_sum[rows], axis=1)
        uncertainty = np.clip(1.0 - resultant / np.maximum(self.weight_sum[rows], 1e-8), 0.0, 1.0)
        confidence = (
            (1.0 - np.exp(-self.weight_sum[rows] / 3.0))
            * (1.0 - np.exp(-self.view_count[rows].astype(np.float32) / 2.0))
            * (1.0 - 0.5 * uncertainty)
        )
        return CanonicalSurfaceField(
            primitive_rows=rows,
            codes=self.feature_sum[rows],
            confidence=confidence.astype(np.float32),
            uncertainty=uncertainty.astype(np.float32),
            physical_map_sha256=physical_map.content_sha256,
            metadata={
                "artifact_type": SCHEMA,
                "representation": "one_exact_contributor_fused_canonical_vfm_code_per_observed_clean_2dgs_primitive",
                "vfm_layer": "radio_final",
                "stored_feature_type_count": 1,
                "stored_downstream_embedding_count": 0,
                "retrieval_and_local_heads": "regenerable_readouts_not_stored",
                "stores_mapping_rgb": False,
                "stores_mapping_image_paths": False,
                "stores_mapping_image_ids": False,
                "uses_alike_descriptors": False,
                "uses_radio_intermediate": False,
                "uses_sfm_points": False,
                "uses_sfm_tracks": False,
                "uses_point_correspondences": False,
                **dict(metadata or {}),
            },
        )


def build_canonical_surface_field(
    source: SurfaceFeatureField,
    physical_map: GoalMapletPhysicalMap,
    *,
    minimum_confidence: float = 0.0,
    metadata: Mapping[str, object] | None = None,
) -> CanonicalSurfaceField:
    row_by_id = {int(value): row for row, value in enumerate(physical_map.primitive_ids.tolist())}
    source_rows, primitive_rows = [], []
    for source_row, primitive_id in enumerate(source.source_indices.tolist()):
        row = row_by_id.get(int(primitive_id))
        if row is not None and float(source.confidence[source_row]) >= float(minimum_confidence):
            source_rows.append(source_row)
            primitive_rows.append(row)
    selected = np.asarray(source_rows, dtype=np.int64)
    return CanonicalSurfaceField(
        primitive_rows=np.asarray(primitive_rows, dtype=np.int64),
        codes=source.features[selected],
        confidence=source.confidence[selected],
        uncertainty=source.uncertainty[selected],
        physical_map_sha256=physical_map.content_sha256,
        metadata={
            "artifact_type": SCHEMA,
            "representation": "one_canonical_radio_final_code_per_observed_clean_2dgs_primitive",
            "vfm_layer": "radio_final",
            "stored_feature_type_count": 1,
            "stored_downstream_embedding_count": 0,
            "retrieval_and_local_heads": "regenerable_readouts_not_stored",
            "stores_mapping_rgb": False,
            "stores_mapping_image_paths": False,
            "stores_mapping_image_ids": False,
            "uses_alike_descriptors": False,
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_point_correspondences": False,
            **dict(metadata or {}),
        },
    )


def _aggregate_readout(
    offsets: np.ndarray,
    member_rows: np.ndarray,
    member_weights: np.ndarray,
    field: CanonicalSurfaceField,
    scene_primitive_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    field_row = np.full((scene_primitive_count,), -1, dtype=np.int64)
    field_row[field.primitive_rows] = np.arange(field.primitive_rows.size, dtype=np.int64)
    descriptors = np.zeros((offsets.size - 1, field.feature_dim), dtype=np.float32)
    coverage = np.zeros((offsets.size - 1,), dtype=np.float32)
    for row in range(offsets.size - 1):
        start, end = int(offsets[row]), int(offsets[row + 1])
        primitive = member_rows[start:end]
        selected = field_row[primitive]
        valid = selected >= 0
        if not np.any(valid):
            continue
        raw_weight = np.asarray(member_weights[start:end], dtype=np.float64)
        weight = raw_weight[valid] * np.maximum(field.confidence[selected[valid]], 1e-3)
        descriptor = np.sum(weight[:, None] * field.codes[selected[valid]], axis=0)
        descriptors[row] = descriptor / max(float(np.linalg.norm(descriptor)), 1e-8)
        coverage[row] = float(np.sum(raw_weight[valid]) / max(float(np.sum(raw_weight)), 1e-12))
    return descriptors, coverage


def readout_canonical_field(
    field: CanonicalSurfaceField,
    physical_map: GoalMapletPhysicalMap,
) -> CanonicalFieldReadout:
    if field.physical_map_sha256 != physical_map.content_sha256:
        raise ValueError("canonical field and physical map lineage differ")
    parent, parent_coverage = _aggregate_readout(
        physical_map.membership_offsets,
        physical_map.membership_primitive_rows,
        physical_map.membership_weights,
        field,
        physical_map.primitive_ids.size,
    )
    child, child_coverage = _aggregate_readout(
        physical_map.child_member_offsets,
        physical_map.child_member_primitive_rows,
        physical_map.child_member_weights,
        field,
        physical_map.primitive_ids.size,
    )
    return CanonicalFieldReadout(parent, parent_coverage, child, child_coverage)
