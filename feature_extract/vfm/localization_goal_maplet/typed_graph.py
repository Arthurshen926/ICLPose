"""Typed physical graph over Goal-Maplet context parents.

The graph stores geometry/statistical relations only.  It never duplicates the
canonical VFM field or retains mapping-view identities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .canonical_field import CanonicalSurfaceField, readout_canonical_field
from .lineage import arrays_sha256, validate_deployment_metadata
from .pfir import _primitive_to_maplet_links
from .physical_map import GoalMapletPhysicalMap


GEOMETRY_ADJACENCY = np.uint8(1)
SURFACE_CONTINUITY = np.uint8(2)
MAPPING_COVISIBILITY = np.uint8(3)
DISTINCTIVE_CONTEXT = np.uint8(4)
SCHEMA = "goal_maplet_typed_parent_graph_v1"


@dataclass(frozen=True)
class TypedParentGraph:
    edge_source: np.ndarray
    edge_target: np.ndarray
    edge_type: np.ndarray
    # distance, normal cosine, co-visibility Jaccard, appearance cosine,
    # source distinctiveness, target distinctiveness
    edge_features: np.ndarray
    parent_view_count: np.ndarray
    parent_distinctiveness: np.ndarray
    physical_map_sha256: str
    canonical_field_sha256: str
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        source = np.asarray(self.edge_source, dtype=np.int32).reshape(-1)
        target = np.asarray(self.edge_target, dtype=np.int32).reshape(-1)
        kind = np.asarray(self.edge_type, dtype=np.uint8).reshape(-1)
        feature = np.asarray(self.edge_features, dtype=np.float32)
        view_count = np.asarray(self.parent_view_count, dtype=np.int32).reshape(-1)
        distinctive = np.asarray(self.parent_distinctiveness, dtype=np.float32).reshape(-1)
        parent_count = view_count.size
        if (
            source.shape != target.shape or kind.shape != source.shape
            or feature.shape != (source.size, 6)
            or distinctive.shape != view_count.shape
            or np.any(source < 0) or np.any(target < 0)
            or np.any(source >= parent_count) or np.any(target >= parent_count)
            or np.any(source >= target)
            or not set(np.unique(kind).tolist()).issubset({1, 2, 3, 4})
        ):
            raise ValueError("invalid typed parent graph")
        metadata = dict(self.metadata or {})
        if metadata.get("artifact_type", SCHEMA) != SCHEMA:
            raise ValueError("not a Goal-Maplet typed graph")
        validate_deployment_metadata(metadata)
        if int(metadata.get("stored_downstream_embedding_count", 0)) != 0:
            raise ValueError("typed graph cannot store downstream embeddings")
        object.__setattr__(self, "edge_source", source)
        object.__setattr__(self, "edge_target", target)
        object.__setattr__(self, "edge_type", kind)
        object.__setattr__(self, "edge_features", feature)
        object.__setattr__(self, "parent_view_count", view_count)
        object.__setattr__(self, "parent_distinctiveness", distinctive)
        object.__setattr__(self, "metadata", metadata)

    @property
    def content_sha256(self) -> str:
        return arrays_sha256({
            "edge_source": self.edge_source,
            "edge_target": self.edge_target,
            "edge_type": self.edge_type,
            "edge_features": self.edge_features,
            "parent_view_count": self.parent_view_count,
            "parent_distinctiveness": self.parent_distinctiveness,
        })

    def save_npz(self, path: Path) -> None:
        metadata = {**dict(self.metadata or {}), "artifact_type": SCHEMA, "content_sha256": self.content_sha256}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            Path(path),
            edge_source=self.edge_source,
            edge_target=self.edge_target,
            edge_type=self.edge_type,
            edge_features=self.edge_features,
            parent_view_count=self.parent_view_count,
            parent_distinctiveness=self.parent_distinctiveness,
            physical_map_sha256=np.asarray(self.physical_map_sha256),
            canonical_field_sha256=np.asarray(self.canonical_field_sha256),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "TypedParentGraph":
        with np.load(Path(path), allow_pickle=False) as data:
            result = cls(
                edge_source=data["edge_source"], edge_target=data["edge_target"],
                edge_type=data["edge_type"], edge_features=data["edge_features"],
                parent_view_count=data["parent_view_count"],
                parent_distinctiveness=data["parent_distinctiveness"],
                physical_map_sha256=str(data["physical_map_sha256"].item()),
                canonical_field_sha256=str(data["canonical_field_sha256"].item()),
                metadata=json.loads(str(data["metadata_json"].item())),
            )
        declared = str(result.metadata.get("content_sha256", ""))
        if declared and declared != result.content_sha256:
            raise ValueError("typed graph content hash mismatch")
        return result

    def covisibility_matrix(self) -> np.ndarray:
        count = self.parent_view_count.size
        matrix = np.zeros((count, count), dtype=np.float32)
        selected = self.edge_type == MAPPING_COVISIBILITY
        source, target = self.edge_source[selected], self.edge_target[selected]
        value = self.edge_features[selected, 2]
        matrix[source, target] = value
        matrix[target, source] = value
        np.fill_diagonal(matrix, 1.0)
        return matrix


def build_typed_parent_graph(
    physical: GoalMapletPhysicalMap,
    field: CanonicalSurfaceField,
    contributor_paths: list[Path],
    *,
    parent_descriptors: np.ndarray | None = None,
    geometry_neighbors: int = 8,
    continuity_neighbors: int = 16,
    distinctive_anchor_count: int = 64,
    distinctive_edges_per_parent: int = 2,
    minimum_covisibility_views: int = 2,
    metadata: Mapping[str, object] | None = None,
) -> TypedParentGraph:
    if field.physical_map_sha256 != physical.content_sha256:
        raise ValueError("canonical field and physical map lineage differ")
    readout = readout_canonical_field(field, physical)
    descriptor = np.asarray(
        readout.parent_descriptors if parent_descriptors is None else parent_descriptors,
        dtype=np.float64,
    )
    if descriptor.shape != readout.parent_descriptors.shape:
        raise ValueError("typed-graph parent readout shape differs")
    descriptor /= np.maximum(np.linalg.norm(descriptor, axis=1, keepdims=True), 1e-8)
    similarity = descriptor @ descriptor.T
    np.fill_diagonal(similarity, -1.0)
    distinctiveness = np.clip(1.0 - np.max(similarity, axis=1), 0.0, 2.0).astype(np.float32)
    centers = physical.maplet_centers
    delta = centers[:, None] - centers[None]
    distance = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(distance, np.inf)
    neighbor_order = np.argsort(distance, axis=1, kind="stable")
    edges: dict[tuple[int, int, int], tuple[float, float]] = {}

    def add(source: int, target: int, kind: int, covis: float = 0.0) -> None:
        if source == target:
            return
        left, right = sorted((int(source), int(target)))
        edges[(left, right, int(kind))] = (float(covis), float(similarity[left, right]))

    for source in range(centers.shape[0]):
        for target in neighbor_order[source, : int(geometry_neighbors)].tolist():
            add(source, int(target), int(GEOMETRY_ADJACENCY))
        for target in neighbor_order[source, : int(continuity_neighbors)].tolist():
            normal_cosine = float(np.dot(physical.maplet_normals[source], physical.maplet_normals[target]))
            scale = float(np.linalg.norm(physical.maplet_extents[source]) + np.linalg.norm(physical.maplet_extents[target]))
            if normal_cosine >= 0.90 and distance[source, target] <= max(2.0, 1.5 * scale):
                add(source, int(target), int(SURFACE_CONTINUITY))

    primitive_row_by_id = np.full((int(np.max(physical.primitive_ids)) + 1,), -1, dtype=np.int64)
    primitive_row_by_id[physical.primitive_ids] = np.arange(physical.primitive_ids.size, dtype=np.int64)
    link_offsets, link_parent, link_weight = _primitive_to_maplet_links(physical)
    dominant_parent = np.full((physical.primitive_ids.size,), -1, dtype=np.int64)
    dominant_weight = np.full((physical.primitive_ids.size,), -np.inf, dtype=np.float32)
    for primitive in range(physical.primitive_ids.size):
        start, end = int(link_offsets[primitive]), int(link_offsets[primitive + 1])
        if end > start:
            local = int(np.argmax(link_weight[start:end]))
            dominant_parent[primitive] = int(link_parent[start + local])
            dominant_weight[primitive] = float(link_weight[start + local])
    view_count = np.zeros((physical.maplet_ids.size,), dtype=np.int32)
    pair_count: dict[tuple[int, int], int] = {}
    for path in contributor_paths:
        with np.load(path, allow_pickle=False) as data:
            ids = np.asarray(data["topk_ids"], dtype=np.int64).reshape(-1)
            weight = np.asarray(data["topk_weights"], dtype=np.float32).reshape(-1)
        valid = (ids >= 0) & (ids < primitive_row_by_id.size) & (weight > 0.0)
        primitive_rows = primitive_row_by_id[ids[valid]]
        mass = weight[valid]
        keep = primitive_rows >= 0
        primitive_rows, mass = primitive_rows[keep], mass[keep]
        owner = dominant_parent[primitive_rows]
        linked = owner >= 0
        parent_mass = np.bincount(
            owner[linked],
            weights=mass[linked] * dominant_weight[primitive_rows[linked]],
            minlength=physical.maplet_ids.size,
        ).astype(np.float64)
        threshold = max(1.0, 5e-4 * float(np.sum(parent_mass)))
        visible = np.flatnonzero(parent_mass >= threshold)
        if visible.size > 128:
            visible = visible[np.argsort(-parent_mass[visible], kind="stable")[:128]]
        view_count[visible] += 1
        for offset, source in enumerate(visible.tolist()):
            for target in visible[offset + 1 :].tolist():
                key = (min(int(source), int(target)), max(int(source), int(target)))
                pair_count[key] = pair_count.get(key, 0) + 1
    for (source, target), count in pair_count.items():
        if count < int(minimum_covisibility_views):
            continue
        union = int(view_count[source]) + int(view_count[target]) - int(count)
        jaccard = float(count / max(union, 1))
        add(source, target, int(MAPPING_COVISIBILITY), jaccard)

    valid_parent = np.flatnonzero(readout.parent_coverage > 0.0)
    anchors = valid_parent[np.argsort(-distinctiveness[valid_parent], kind="stable")[: int(distinctive_anchor_count)]]
    for source in range(centers.shape[0]):
        candidates = anchors[anchors != source]
        if candidates.size:
            order = np.argsort(distance[source, candidates], kind="stable")[: int(distinctive_edges_per_parent)]
            for target in candidates[order].tolist():
                add(source, int(target), int(DISTINCTIVE_CONTEXT))

    rows = sorted(edges)
    source = np.asarray([value[0] for value in rows], dtype=np.int32)
    target = np.asarray([value[1] for value in rows], dtype=np.int32)
    kind = np.asarray([value[2] for value in rows], dtype=np.uint8)
    features = np.zeros((len(rows), 6), dtype=np.float32)
    for index, (left, right, edge_kind) in enumerate(rows):
        covis, appearance = edges[(left, right, edge_kind)]
        features[index] = [
            distance[left, right],
            np.dot(physical.maplet_normals[left], physical.maplet_normals[right]),
            covis,
            appearance,
            distinctiveness[left],
            distinctiveness[right],
        ]
    return TypedParentGraph(
        edge_source=source, edge_target=target, edge_type=kind, edge_features=features,
        parent_view_count=view_count, parent_distinctiveness=distinctiveness,
        physical_map_sha256=physical.content_sha256,
        canonical_field_sha256=field.content_sha256,
        metadata={
            "artifact_type": SCHEMA,
            "stored_feature_type_count": 1,
            "stored_downstream_embedding_count": 0,
            "stores_mapping_rgb": False,
            "stores_mapping_image_paths": False,
            "stores_mapping_image_ids": False,
            "uses_alike_descriptors": False,
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "parent_child_relation": "physical_map_child_parent_rows",
            "mapping_view_count": len(contributor_paths),
            **dict(metadata or {}),
        },
    )
