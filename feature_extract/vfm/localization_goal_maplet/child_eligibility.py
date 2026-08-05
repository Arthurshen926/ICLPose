"""Task-specific child-surface eligibility derived from exact geometry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .audit import _component_count
from .canonical_field import CanonicalSurfaceField, readout_canonical_field
from .lineage import arrays_sha256
from .physical_map import DOUBLE_SIDED, GoalMapletPhysicalMap


SCHEMA = "goal_maplet_child_eligibility_v1"


@dataclass(frozen=True)
class ChildGeometryEligibility:
    component_count: np.ndarray
    normal_p90_degrees: np.ndarray
    relative_depth_span: np.ndarray
    feature_coverage: np.ndarray
    retrieval_qualified: np.ndarray
    proposal_qualified: np.ndarray
    refinement_qualified: np.ndarray
    physical_map_sha256: str
    canonical_field_sha256: str
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        count = np.asarray(self.component_count, dtype=np.int64).reshape(-1).size
        for name, dtype in (
            ("component_count", np.int64),
            ("normal_p90_degrees", np.float32),
            ("relative_depth_span", np.float32),
            ("feature_coverage", np.float32),
            ("retrieval_qualified", bool),
            ("proposal_qualified", bool),
            ("refinement_qualified", bool),
        ):
            value = np.asarray(getattr(self, name), dtype=dtype).reshape(-1)
            if value.shape != (count,):
                raise ValueError("child eligibility arrays differ")
            object.__setattr__(self, name, value)
        metadata = dict(self.metadata or {})
        if metadata.get("artifact_type", SCHEMA) != SCHEMA:
            raise ValueError("not a Goal-Maplet child eligibility artifact")
        object.__setattr__(self, "metadata", metadata)

    @property
    def content_sha256(self) -> str:
        return arrays_sha256({
            name: np.asarray(getattr(self, name))
            for name in (
                "component_count", "normal_p90_degrees", "relative_depth_span",
                "feature_coverage", "retrieval_qualified", "proposal_qualified",
                "refinement_qualified",
            )
        })

    def save_npz(self, path: Path) -> None:
        metadata = {**dict(self.metadata), "artifact_type": SCHEMA, "content_sha256": self.content_sha256}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            Path(path),
            component_count=self.component_count,
            normal_p90_degrees=self.normal_p90_degrees,
            relative_depth_span=self.relative_depth_span,
            feature_coverage=self.feature_coverage,
            retrieval_qualified=self.retrieval_qualified,
            proposal_qualified=self.proposal_qualified,
            refinement_qualified=self.refinement_qualified,
            physical_map_sha256=np.asarray(self.physical_map_sha256),
            canonical_field_sha256=np.asarray(self.canonical_field_sha256),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "ChildGeometryEligibility":
        with np.load(Path(path), allow_pickle=False) as data:
            result = cls(
                *[np.asarray(data[name]) for name in (
                    "component_count", "normal_p90_degrees", "relative_depth_span",
                    "feature_coverage", "retrieval_qualified", "proposal_qualified",
                    "refinement_qualified",
                )],
                str(np.asarray(data["physical_map_sha256"]).item()),
                str(np.asarray(data["canonical_field_sha256"]).item()),
                json.loads(str(np.asarray(data["metadata_json"]).item())),
            )
        if str(result.metadata.get("content_sha256", "")) != result.content_sha256:
            raise ValueError("child eligibility hash mismatch")
        return result


def build_child_geometry_eligibility(
    physical: GoalMapletPhysicalMap,
    field: CanonicalSurfaceField,
    *,
    proposal_normal_p90_degrees: float = 40.0,
    proposal_relative_depth_span: float = 0.40,
    refinement_normal_p90_degrees: float = 25.0,
    refinement_relative_depth_span: float = 0.25,
    refinement_minimum_feature_coverage: float = 0.75,
) -> ChildGeometryEligibility:
    readout = readout_canonical_field(field, physical)
    components, normal_p90, depth_span = [], [], []
    for child in range(physical.child_parent_rows.size):
        start = int(physical.child_member_offsets[child])
        end = int(physical.child_member_offsets[child + 1])
        primitive = physical.child_member_primitive_rows[start:end]
        points = physical.primitive_centers[primitive]
        radii = np.maximum(physical.primitive_scale1[primitive], physical.primitive_scale2[primitive])
        components.append(_component_count(points, radii))
        cosine = np.clip(physical.primitive_normals[primitive] @ physical.child_normals[child], -1.0, 1.0)
        cosine = np.where(physical.primitive_sidedness[primitive] == DOUBLE_SIDED, np.abs(cosine), cosine)
        normal_p90.append(float(np.percentile(np.degrees(np.arccos(cosine)), 90.0)))
        local = (points - physical.child_centers[child]) @ physical.child_frames[child].T
        depth_span.append(float(np.ptp(local[:, 2]) / max(2.0 * np.max(physical.child_extents[child, :2]), 1e-6)))
    components = np.asarray(components, dtype=np.int64)
    normal_p90 = np.asarray(normal_p90, dtype=np.float32)
    depth_span = np.asarray(depth_span, dtype=np.float32)
    coverage = np.asarray(readout.child_coverage, dtype=np.float32)
    retrieval = coverage > 0.0
    proposal = retrieval & (normal_p90 <= float(proposal_normal_p90_degrees)) & (
        depth_span <= float(proposal_relative_depth_span)
    )
    refinement = proposal & (components == 1) & (
        normal_p90 <= float(refinement_normal_p90_degrees)
    ) & (depth_span <= float(refinement_relative_depth_span)) & (
        coverage >= float(refinement_minimum_feature_coverage)
    )
    return ChildGeometryEligibility(
        components, normal_p90, depth_span, coverage, retrieval, proposal, refinement,
        physical.content_sha256, field.content_sha256,
        metadata={
            "artifact_type": SCHEMA,
            "proposal_normal_p90_degrees": float(proposal_normal_p90_degrees),
            "proposal_relative_depth_span": float(proposal_relative_depth_span),
            "refinement_normal_p90_degrees": float(refinement_normal_p90_degrees),
            "refinement_relative_depth_span": float(refinement_relative_depth_span),
            "refinement_minimum_feature_coverage": float(refinement_minimum_feature_coverage),
            "stored_feature_type_count": 1,
            "stored_downstream_embedding_count": 0,
            "stores_mapping_rgb": False,
            "stores_mapping_image_paths": False,
            "stores_mapping_image_ids": False,
            "uses_alike_descriptors": False,
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
        },
    )
