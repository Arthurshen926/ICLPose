"""Cross-fitted mapping-view appearance modes for physical child surfaces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .lineage import arrays_sha256
from .multimodal_parent_retrieval import AnonymousChildModeReadout


SCHEMA = "goal_maplet_observed_child_mode_readout_v1"
MODE_SEMANTICS = "deterministic_online_weighted_spherical_child_view_modes_v1"


@dataclass(frozen=True)
class ObservedChildModeArtifact:
    readout: AnonymousChildModeReadout
    physical_map_sha256: str
    canonical_field_sha256: str
    surface_mapper_file_sha256: str
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        metadata = dict(self.metadata)
        if metadata.get("artifact_type", SCHEMA) != SCHEMA:
            raise ValueError("not an observed child mode artifact")
        if metadata.get("mode_semantics") != MODE_SEMANTICS:
            raise ValueError("observed child mode semantics differ")
        for name, value in (
            ("physical_map_sha256", self.physical_map_sha256),
            ("canonical_field_sha256", self.canonical_field_sha256),
            ("surface_mapper_file_sha256", self.surface_mapper_file_sha256),
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"invalid {name}")
        if metadata.get("uses_query_pose") is not False or metadata.get(
            "uses_query_ground_truth"
        ) is not False:
            raise ValueError("observed child modes must be map-only")
        object.__setattr__(self, "metadata", metadata)

    @property
    def content_sha256(self) -> str:
        return arrays_sha256(
            {
                "descriptors": self.readout.descriptors,
                "weights": self.readout.weights,
                "child_coverage": self.readout.child_coverage,
            }
        )

    def save_npz(self, path: Path) -> None:
        metadata = {
            **dict(self.metadata),
            "artifact_type": SCHEMA,
            "mode_semantics": MODE_SEMANTICS,
            "content_sha256": self.content_sha256,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            descriptors=self.readout.descriptors,
            weights=self.readout.weights,
            child_coverage=self.readout.child_coverage,
            physical_map_sha256=np.asarray(self.physical_map_sha256),
            canonical_field_sha256=np.asarray(self.canonical_field_sha256),
            surface_mapper_file_sha256=np.asarray(self.surface_mapper_file_sha256),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "ObservedChildModeArtifact":
        with np.load(path, allow_pickle=False) as data:
            expected = {
                "descriptors", "weights", "child_coverage",
                "physical_map_sha256", "canonical_field_sha256",
                "surface_mapper_file_sha256", "metadata_json",
            }
            if set(data.files) != expected:
                raise ValueError("observed child mode NPZ members differ")
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
            result = cls(
                readout=AnonymousChildModeReadout(
                    descriptors=np.asarray(data["descriptors"], dtype=np.float32),
                    weights=np.asarray(data["weights"], dtype=np.float32),
                    child_coverage=np.asarray(data["child_coverage"], dtype=np.float32),
                ),
                physical_map_sha256=str(np.asarray(data["physical_map_sha256"]).item()),
                canonical_field_sha256=str(np.asarray(data["canonical_field_sha256"]).item()),
                surface_mapper_file_sha256=str(
                    np.asarray(data["surface_mapper_file_sha256"]).item()
                ),
                metadata=metadata,
            )
        if metadata.get("content_sha256") != result.content_sha256:
            raise ValueError("observed child mode content hash differs")
        return result


def update_online_child_modes(
    descriptors: np.ndarray,
    accumulated_weights: np.ndarray,
    mode_counts: np.ndarray,
    child_rows: np.ndarray,
    observations: np.ndarray,
    observation_weights: np.ndarray,
    *,
    minimum_angular_residual: float,
) -> None:
    """Deterministically update bounded weighted spherical modes in-place."""

    centers = np.asarray(descriptors, dtype=np.float32)
    mass = np.asarray(accumulated_weights, dtype=np.float64)
    counts = np.asarray(mode_counts, dtype=np.int32)
    child = np.asarray(child_rows, dtype=np.int64).reshape(-1)
    value = np.asarray(observations, dtype=np.float32)
    weight = np.asarray(observation_weights, dtype=np.float64).reshape(-1)
    if (
        centers.ndim != 3
        or mass.shape != centers.shape[:2]
        or counts.shape != (centers.shape[0],)
        or value.shape != (child.size, centers.shape[2])
        or weight.shape != child.shape
        or np.any((child < 0) | (child >= centers.shape[0]))
        or np.any(~np.isfinite(value))
        or np.any(~np.isfinite(weight))
        or np.any(weight <= 0.0)
        or not 0.0 <= float(minimum_angular_residual) <= 2.0
    ):
        raise ValueError("invalid online child mode update")
    value = value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-8)
    for row, descriptor, item_weight in zip(child.tolist(), value, weight.tolist()):
        count = int(counts[row])
        if count == 0:
            chosen = 0
            counts[row] = 1
        else:
            similarity = centers[row, :count] @ descriptor
            nearest = int(np.argmax(similarity))
            if (
                count < centers.shape[1]
                and 1.0 - float(similarity[nearest])
                >= float(minimum_angular_residual)
            ):
                chosen = count
                counts[row] += 1
            else:
                chosen = nearest
        old_mass = float(mass[row, chosen])
        combined = centers[row, chosen] * old_mass + descriptor * float(item_weight)
        centers[row, chosen] = combined / max(float(np.linalg.norm(combined)), 1e-8)
        mass[row, chosen] = old_mass + float(item_weight)
