"""Compact low-rank view-conditioned residuals over a canonical VFM field.

The deployed map still owns exactly one canonical RADIO-final code ``mu_i``
per observed physical primitive.  This artifact stores a shared residual basis
and small per-primitive regression coefficients, so a pose can evaluate

    normalize(mu_i + B @ a_i(view_direction, projected_scale)).

It deliberately stores neither mapping images nor per-view descriptors.  A
primitive outside its observed view/scale chart falls back to ``mu_i`` rather
than extrapolating an unconstrained appearance code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .lineage import arrays_sha256, validate_deployment_metadata


SCHEMA = "goal_maplet_low_rank_view_conditioned_field_v1"
PREDICTOR_COUNT = 5  # intercept, centered local view xyz, centered log scale


def _unit_rows(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-8)


@dataclass(frozen=True)
class ViewConditionedPrimitiveField:
    """Low-rank conditional residuals aligned one-to-one with a canonical field."""

    primitive_rows: np.ndarray
    residual_basis: np.ndarray
    coefficients: np.ndarray
    observation_count: np.ndarray
    mean_local_direction: np.ndarray
    direction_concentration: np.ndarray
    minimum_direction_cosine: np.ndarray
    mean_log_projected_scale: np.ndarray
    minimum_log_projected_scale: np.ndarray
    maximum_log_projected_scale: np.ndarray
    physical_map_sha256: str
    canonical_field_sha256: str
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        rows = np.asarray(self.primitive_rows, dtype=np.int64).reshape(-1)
        basis = np.asarray(self.residual_basis, dtype=np.float32)
        coefficient = np.asarray(self.coefficients)
        count = np.asarray(self.observation_count, dtype=np.int32).reshape(-1)
        direction = np.asarray(self.mean_local_direction, dtype=np.float32)
        concentration = np.asarray(self.direction_concentration, dtype=np.float32).reshape(-1)
        minimum_cosine = np.asarray(self.minimum_direction_cosine, dtype=np.float32).reshape(-1)
        mean_scale = np.asarray(self.mean_log_projected_scale, dtype=np.float32).reshape(-1)
        minimum_scale = np.asarray(self.minimum_log_projected_scale, dtype=np.float32).reshape(-1)
        maximum_scale = np.asarray(self.maximum_log_projected_scale, dtype=np.float32).reshape(-1)
        primitive_count = rows.size
        if (
            np.unique(rows).size != primitive_count
            or np.any(rows < 0)
            or basis.ndim != 2
            or basis.shape[0] <= 0
            or coefficient.shape != (primitive_count, PREDICTOR_COUNT, basis.shape[0])
            or count.shape != rows.shape
            or direction.shape != (primitive_count, 3)
            or concentration.shape != rows.shape
            or minimum_cosine.shape != rows.shape
            or mean_scale.shape != rows.shape
            or minimum_scale.shape != rows.shape
            or maximum_scale.shape != rows.shape
        ):
            raise ValueError("invalid low-rank view-conditioned field arrays")
        for name, value in (
            ("residual_basis", basis),
            ("coefficients", coefficient),
            ("mean_local_direction", direction),
            ("direction_concentration", concentration),
            ("minimum_direction_cosine", minimum_cosine),
            ("mean_log_projected_scale", mean_scale),
            ("minimum_log_projected_scale", minimum_scale),
            ("maximum_log_projected_scale", maximum_scale),
        ):
            if not np.all(np.isfinite(value)):
                raise ValueError(f"non-finite {name} in view-conditioned field")
        if (
            np.any(count < 0)
            or np.any((concentration < -1e-5) | (concentration > 1.0 + 1e-5))
            or np.any((minimum_cosine < -1.0001) | (minimum_cosine > 1.0001))
            or np.any(minimum_scale > maximum_scale)
        ):
            raise ValueError("invalid view-conditioned validity chart")
        observed = count > 0
        if np.any(observed):
            direction_norm = np.linalg.norm(direction[observed], axis=1)
            if np.any(np.abs(direction_norm - 1.0) > 2e-3):
                raise ValueError("observed mean view directions must be unit length")
        gram = basis @ basis.T
        if not np.allclose(gram, np.eye(basis.shape[0]), atol=2e-3, rtol=2e-3):
            raise ValueError("view-conditioned residual basis must be row-orthonormal")
        object.__setattr__(self, "primitive_rows", rows)
        object.__setattr__(self, "residual_basis", basis)
        # Float16 is an intentional deployment representation, not an
        # incidental save-time quantization.  Preserve it for stable lineage.
        object.__setattr__(self, "coefficients", coefficient.astype(np.float16, copy=False))
        object.__setattr__(self, "observation_count", count)
        object.__setattr__(self, "mean_local_direction", direction)
        object.__setattr__(self, "direction_concentration", concentration)
        object.__setattr__(self, "minimum_direction_cosine", minimum_cosine)
        object.__setattr__(self, "mean_log_projected_scale", mean_scale)
        object.__setattr__(self, "minimum_log_projected_scale", minimum_scale)
        object.__setattr__(self, "maximum_log_projected_scale", maximum_scale)
        metadata = dict(self.metadata or {})
        if metadata.get("artifact_type", SCHEMA) != SCHEMA:
            raise ValueError("not a low-rank view-conditioned Goal-Maplet field")
        validate_deployment_metadata(metadata)
        object.__setattr__(self, "metadata", metadata)
        declared_hash = str(metadata.get("content_sha256", ""))
        if declared_hash and declared_hash != self.content_sha256:
            raise ValueError(
                "view-conditioned field content hash mismatch: "
                f"declared={declared_hash} actual={self.content_sha256}"
            )

    @property
    def rank(self) -> int:
        return int(self.residual_basis.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.residual_basis.shape[1])

    @property
    def minimum_views(self) -> int:
        return max(int(dict(self.metadata or {}).get("minimum_views", 3)), 1)

    @property
    def content_sha256(self) -> str:
        return arrays_sha256({
            "primitive_rows": self.primitive_rows,
            "residual_basis": self.residual_basis,
            "coefficients": self.coefficients,
            "observation_count": self.observation_count,
            "mean_local_direction": self.mean_local_direction,
            "direction_concentration": self.direction_concentration,
            "minimum_direction_cosine": self.minimum_direction_cosine,
            "mean_log_projected_scale": self.mean_log_projected_scale,
            "minimum_log_projected_scale": self.minimum_log_projected_scale,
            "maximum_log_projected_scale": self.maximum_log_projected_scale,
        })

    def validate_alignment(
        self,
        *,
        physical_map_sha256: str,
        canonical_field_sha256: str,
        canonical_primitive_rows: np.ndarray,
        canonical_feature_dim: int,
    ) -> None:
        if self.physical_map_sha256 != str(physical_map_sha256):
            raise ValueError("view-conditioned and physical map lineage differ")
        if self.canonical_field_sha256 != str(canonical_field_sha256):
            raise ValueError("view-conditioned and canonical field lineage differ")
        if not np.array_equal(
            self.primitive_rows, np.asarray(canonical_primitive_rows, dtype=np.int64),
        ):
            raise ValueError("view-conditioned primitive order differs from canonical field")
        if self.feature_dim != int(canonical_feature_dim):
            raise ValueError("view-conditioned feature dimension differs")

    def save_npz(self, path: Path) -> None:
        metadata = {
            **dict(self.metadata or {}),
            "artifact_type": SCHEMA,
            "content_sha256": self.content_sha256,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            Path(path),
            primitive_rows=self.primitive_rows,
            residual_basis=self.residual_basis.astype(np.float32),
            coefficients=self.coefficients.astype(np.float16),
            observation_count=self.observation_count.astype(np.int32),
            mean_local_direction=self.mean_local_direction.astype(np.float32),
            direction_concentration=self.direction_concentration.astype(np.float32),
            minimum_direction_cosine=self.minimum_direction_cosine.astype(np.float32),
            mean_log_projected_scale=self.mean_log_projected_scale.astype(np.float32),
            minimum_log_projected_scale=self.minimum_log_projected_scale.astype(np.float32),
            maximum_log_projected_scale=self.maximum_log_projected_scale.astype(np.float32),
            physical_map_sha256=np.asarray(self.physical_map_sha256),
            canonical_field_sha256=np.asarray(self.canonical_field_sha256),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "ViewConditionedPrimitiveField":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                primitive_rows=np.asarray(data["primitive_rows"], dtype=np.int64),
                residual_basis=np.asarray(data["residual_basis"], dtype=np.float32),
                coefficients=np.asarray(data["coefficients"], dtype=np.float16),
                observation_count=np.asarray(data["observation_count"], dtype=np.int32),
                mean_local_direction=np.asarray(data["mean_local_direction"], dtype=np.float32),
                direction_concentration=np.asarray(data["direction_concentration"], dtype=np.float32),
                minimum_direction_cosine=np.asarray(data["minimum_direction_cosine"], dtype=np.float32),
                mean_log_projected_scale=np.asarray(data["mean_log_projected_scale"], dtype=np.float32),
                minimum_log_projected_scale=np.asarray(data["minimum_log_projected_scale"], dtype=np.float32),
                maximum_log_projected_scale=np.asarray(data["maximum_log_projected_scale"], dtype=np.float32),
                physical_map_sha256=str(np.asarray(data["physical_map_sha256"]).item()),
                canonical_field_sha256=str(np.asarray(data["canonical_field_sha256"]).item()),
                metadata=json.loads(str(np.asarray(data["metadata_json"]).item())),
            )

    def condition_codes_numpy(
        self,
        canonical_codes: np.ndarray,
        field_indices: np.ndarray,
        local_view_directions: np.ndarray,
        log_projected_scales: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return conditioned codes and whether the residual chart was active."""

        canonical = np.asarray(canonical_codes, dtype=np.float32)
        index = np.asarray(field_indices, dtype=np.int64).reshape(-1)
        direction = _unit_rows(np.asarray(local_view_directions, dtype=np.float32).reshape(-1, 3))
        log_scale = np.asarray(log_projected_scales, dtype=np.float32).reshape(-1)
        if (
            canonical.shape != (self.primitive_rows.size, self.feature_dim)
            or direction.shape[0] != index.size
            or log_scale.shape != index.shape
            or np.any((index < 0) | (index >= self.primitive_rows.size))
        ):
            raise ValueError("view-conditioned code query differs from field")
        mean_direction = self.mean_local_direction[index]
        direction_center = mean_direction * self.direction_concentration[index, None]
        direction_cosine = np.sum(direction * mean_direction, axis=1)
        direction_margin = float(dict(self.metadata or {}).get("direction_cosine_margin", 0.05))
        scale_margin = float(dict(self.metadata or {}).get("log_scale_margin", 0.25))
        active = (
            (self.observation_count[index] >= self.minimum_views)
            & (direction_cosine >= self.minimum_direction_cosine[index] - direction_margin)
            & (log_scale >= self.minimum_log_projected_scale[index] - scale_margin)
            & (log_scale <= self.maximum_log_projected_scale[index] + scale_margin)
        )
        predictor = np.concatenate((
            np.ones((index.size, 1), dtype=np.float32),
            direction - direction_center,
            (log_scale - self.mean_log_projected_scale[index])[:, None],
        ), axis=1)
        latent = np.einsum(
            "np,npr->nr", predictor, self.coefficients[index].astype(np.float32),
        )
        residual = latent @ self.residual_basis
        residual[~active] = 0.0
        code = canonical[index] + residual
        return _unit_rows(code), active

