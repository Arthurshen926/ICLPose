"""Feature-free mapping-view nodes for Goal-Maplet pose proposals.

The deployed map already contains physical maplets and their typed relations.
This module adds the missing higher-order relation: which physical maplets were
jointly visible from one calibrated mapping pose.  A node stores only a pose
and a sparse distribution over physical-parent rows.  It stores no image ID,
RGB, VFM tensor, downstream embedding, point correspondence, or SfM track.

At query time RADIO is used once to produce the existing sparse parent
posterior.  Mapping-view proposals then use only that posterior and the sparse
graph statistics.  In particular, low absolute retrieval probability cannot
delete a physical identity: retained in-map mass is normalized conditionally
inside each query support before it is compared with a view node.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .lineage import arrays_sha256, validate_deployment_metadata
from .pfir import ContributorLabels, _primitive_to_maplet_links
from .physical_map import GoalMapletPhysicalMap
from .pose_proposal import CoarsePoseModes, _rotation_distance_degrees


SCHEMA = "goal_maplet_mapping_view_graph_v1"


@dataclass(frozen=True)
class MappingViewGraph:
    poses_w2c: np.ndarray
    parent_offsets: np.ndarray
    parent_rows: np.ndarray
    parent_weights: np.ndarray
    physical_map_sha256: str
    canonical_field_sha256: str
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        pose = np.asarray(self.poses_w2c, dtype=np.float64)
        offsets = np.asarray(self.parent_offsets, dtype=np.int64).reshape(-1)
        rows = np.asarray(self.parent_rows, dtype=np.int32).reshape(-1)
        weight = np.asarray(self.parent_weights, dtype=np.float32).reshape(-1)
        if (
            pose.ndim != 3 or pose.shape[1:] != (4, 4)
            or offsets.shape != (pose.shape[0] + 1,)
            or offsets[0] != 0 or offsets[-1] != rows.size
            or np.any(np.diff(offsets) < 0)
            or rows.shape != weight.shape
            or np.any(rows < 0)
            or np.any(~np.isfinite(pose))
            or np.any(~np.isfinite(weight))
            or np.any((weight <= 0.0) | (weight > 1.0))
        ):
            raise ValueError("invalid mapping-view graph")
        for view in range(pose.shape[0]):
            start, end = int(offsets[view]), int(offsets[view + 1])
            if np.unique(rows[start:end]).size != end - start:
                raise ValueError("mapping-view parent rows must be unique")
        metadata = dict(self.metadata or {})
        if metadata.get("artifact_type", SCHEMA) != SCHEMA:
            raise ValueError("not a Goal-Maplet mapping-view graph")
        validate_deployment_metadata(metadata)
        if int(metadata.get("stored_downstream_embedding_count", 0)) != 0:
            raise ValueError("mapping-view graph cannot store downstream embeddings")
        object.__setattr__(self, "poses_w2c", pose)
        object.__setattr__(self, "parent_offsets", offsets)
        object.__setattr__(self, "parent_rows", rows)
        object.__setattr__(self, "parent_weights", weight)
        object.__setattr__(self, "metadata", metadata)

    @property
    def content_sha256(self) -> str:
        return arrays_sha256({
            "poses_w2c": self.poses_w2c,
            "parent_offsets": self.parent_offsets,
            "parent_rows": self.parent_rows,
            "parent_weights": self.parent_weights,
        })

    def save_npz(self, path: Path) -> None:
        metadata = {
            **dict(self.metadata or {}),
            "artifact_type": SCHEMA,
            "content_sha256": self.content_sha256,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            Path(path),
            poses_w2c=self.poses_w2c,
            parent_offsets=self.parent_offsets,
            parent_rows=self.parent_rows,
            parent_weights=self.parent_weights,
            physical_map_sha256=np.asarray(self.physical_map_sha256),
            canonical_field_sha256=np.asarray(self.canonical_field_sha256),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "MappingViewGraph":
        with np.load(Path(path), allow_pickle=False) as data:
            result = cls(
                poses_w2c=data["poses_w2c"],
                parent_offsets=data["parent_offsets"],
                parent_rows=data["parent_rows"],
                parent_weights=data["parent_weights"],
                physical_map_sha256=str(data["physical_map_sha256"].item()),
                canonical_field_sha256=str(data["canonical_field_sha256"].item()),
                metadata=json.loads(str(data["metadata_json"].item())),
            )
        declared = str(result.metadata.get("content_sha256", ""))
        if declared and declared != result.content_sha256:
            raise ValueError("mapping-view graph content hash mismatch")
        return result

    def dense_parent_weights(self, parent_count: int) -> np.ndarray:
        result = np.zeros((self.poses_w2c.shape[0], int(parent_count)), dtype=np.float32)
        for view in range(self.poses_w2c.shape[0]):
            start, end = int(self.parent_offsets[view]), int(self.parent_offsets[view + 1])
            result[view, self.parent_rows[start:end]] = self.parent_weights[start:end]
        return result


@dataclass(frozen=True)
class MappingViewPosterior:
    view_rows: np.ndarray
    scores: np.ndarray
    probabilities: np.ndarray
    null_probability: float
    support_coverage: np.ndarray


def build_mapping_view_graph(
    physical: GoalMapletPhysicalMap,
    canonical_field_sha256: str,
    contributor_paths: Sequence[Path],
    *,
    maximum_parents_per_view: int = 128,
    minimum_parent_mass_fraction: float = 5.0e-4,
    metadata: Mapping[str, object] | None = None,
) -> MappingViewGraph:
    """Build calibrated-pose/view incidence without retaining image identity."""

    if int(maximum_parents_per_view) <= 0:
        raise ValueError("maximum_parents_per_view must be positive")
    primitive_row_by_id = np.full(
        (int(np.max(physical.primitive_ids)) + 1,), -1, dtype=np.int64,
    )
    primitive_row_by_id[physical.primitive_ids] = np.arange(
        physical.primitive_ids.size, dtype=np.int64,
    )
    link_offsets, link_parent, link_weight = _primitive_to_maplet_links(physical)
    poses: list[np.ndarray] = []
    rows_out: list[int] = []
    weights_out: list[float] = []
    offsets = [0]
    for path in contributor_paths:
        labels = ContributorLabels.load_npz(Path(path))
        ids = labels.topk_primitive_ids.reshape(-1)
        mass = labels.topk_weights.reshape(-1).astype(np.float64)
        valid = (ids >= 0) & (ids < primitive_row_by_id.size) & (mass > 0.0)
        primitive_rows = primitive_row_by_id[ids[valid]]
        mass = mass[valid]
        keep = primitive_rows >= 0
        primitive_rows, mass = primitive_rows[keep], mass[keep]
        parent_mass = np.zeros((physical.maplet_ids.size,), dtype=np.float64)
        # Accumulate the exact overlapping primitive-to-maplet membership.
        unique, inverse = np.unique(primitive_rows, return_inverse=True)
        primitive_mass = np.bincount(inverse, weights=mass, minlength=unique.size)
        for primitive, value in zip(unique.tolist(), primitive_mass.tolist()):
            start, end = int(link_offsets[primitive]), int(link_offsets[primitive + 1])
            if end > start:
                np.add.at(
                    parent_mass,
                    link_parent[start:end],
                    float(value) * link_weight[start:end],
                )
        threshold = max(
            1.0,
            float(minimum_parent_mass_fraction) * float(np.sum(parent_mass)),
        )
        selected = np.flatnonzero(parent_mass >= threshold)
        if selected.size > int(maximum_parents_per_view):
            order = np.argsort(-parent_mass[selected], kind="stable")
            selected = selected[order[: int(maximum_parents_per_view)]]
        else:
            selected = selected[np.argsort(-parent_mass[selected], kind="stable")]
        if selected.size == 0:
            continue
        # Relative visibility is a bounded relation strength, not a feature.
        value = parent_mass[selected] / max(float(np.max(parent_mass[selected])), 1e-12)
        poses.append(labels.pose_w2c)
        rows_out.extend(selected.tolist())
        weights_out.extend(value.tolist())
        offsets.append(len(rows_out))
    return MappingViewGraph(
        poses_w2c=np.asarray(poses, dtype=np.float64).reshape(-1, 4, 4),
        parent_offsets=np.asarray(offsets, dtype=np.int64),
        parent_rows=np.asarray(rows_out, dtype=np.int32),
        parent_weights=np.asarray(weights_out, dtype=np.float32),
        physical_map_sha256=physical.content_sha256,
        canonical_field_sha256=str(canonical_field_sha256),
        metadata={
            "artifact_type": SCHEMA,
            "stored_feature_type_count": 1,
            "stored_downstream_embedding_count": 0,
            "stores_mapping_rgb": False,
            "stores_mapping_image_ids": False,
            "stores_mapping_image_paths": False,
            "uses_alike_descriptors": False,
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "view_node_count": len(poses),
            "relation": "calibrated_mapping_pose_to_visible_physical_maplet_distribution",
            **dict(metadata or {}),
        },
    )


def retrieve_mapping_view_posterior(
    graph: MappingViewGraph,
    physical: GoalMapletPhysicalMap,
    candidate_maplet_ids: np.ndarray,
    candidate_probabilities: np.ndarray,
    out_of_map_probabilities: np.ndarray,
    unresolved_probabilities: np.ndarray,
    *,
    maximum_views: int = 64,
    missing_view_probability: float = 0.02,
    temperature: float = 0.10,
) -> MappingViewPosterior:
    """Retrieve pose-bearing view nodes with separated null semantics.

    Candidate probabilities are normalized by retained mass per query support.
    This makes the retrieval posterior a proposal distribution: an ambiguous
    but retained parent cannot be erased merely because another support has a
    much sharper absolute score.  Out-of-map and truncated mass stay in a
    typed query-null branch and never become evidence for a particular view.
    The returned null is computed before normalizing over view nodes, so it is
    invariant to ``maximum_views`` and to the number of graph nodes.
    """

    if graph.physical_map_sha256 != physical.content_sha256:
        raise ValueError("mapping-view graph and physical map differ")
    ids = np.asarray(candidate_maplet_ids, dtype=np.int64)
    probability = np.asarray(candidate_probabilities, dtype=np.float64)
    out_of_map = np.asarray(out_of_map_probabilities, dtype=np.float64).reshape(-1)
    unresolved = np.asarray(unresolved_probabilities, dtype=np.float64).reshape(-1)
    if (
        ids.ndim != 2 or probability.shape != ids.shape
        or out_of_map.shape != (ids.shape[0],)
        or unresolved.shape != out_of_map.shape
        or int(maximum_views) <= 0
        or not 0.0 < float(missing_view_probability) < 1.0
        or float(temperature) <= 0.0
        or np.any(~np.isfinite(probability))
        or np.any(~np.isfinite(out_of_map))
        or np.any(~np.isfinite(unresolved))
        or np.any(probability < 0.0)
        or np.any((out_of_map < 0.0) | (out_of_map > 1.0))
        or np.any((unresolved < out_of_map) | (unresolved > 1.0))
    ):
        raise ValueError("invalid mapping-view retrieval inputs")
    row_by_id = {int(value): row for row, value in enumerate(physical.maplet_ids.tolist())}
    rows = np.full(ids.shape, -1, dtype=np.int64)
    for support in range(ids.shape[0]):
        for slot in range(ids.shape[1]):
            rows[support, slot] = row_by_id.get(int(ids[support, slot]), -1)
    valid = (rows >= 0) & (probability > 0.0)
    retained = np.sum(np.where(valid, probability, 0.0), axis=1)
    conditional = np.divide(
        probability,
        retained[:, None],
        out=np.zeros_like(probability),
        where=retained[:, None] > 0.0,
    )
    dense_view = graph.dense_parent_weights(physical.maplet_ids.size).astype(np.float64)
    safe_rows = np.maximum(rows, 0)
    explained = np.sum(
        dense_view[:, safe_rows] * conditional[None] * valid[None], axis=2,
    )
    explained = np.clip(explained, 0.0, 1.0)
    # Resolved in-map mass is the evidence that a support can identify a view.
    # Truncated in-map mass remains uncertainty and out-of-map mass remains H0;
    # neither is allowed to become an arbitrary identity when retained mass is
    # zero.  Averaging supports avoids treating correlated token groups as
    # independent repeated trials.
    reliability = np.clip(1.0 - unresolved, 0.0, 1.0)
    h1_probability = float(np.mean(reliability)) if reliability.size else 0.0
    h1_probability = float(np.clip(h1_probability, 0.0, 1.0))
    null_probability = 1.0 - h1_probability
    likelihood = (
        float(missing_view_probability)
        + (1.0 - float(missing_view_probability)) * explained
    )
    reliability_sum = float(np.sum(reliability))
    if reliability_sum > 0.0:
        score = np.sum(
            reliability[None] * np.log(np.maximum(likelihood, 1e-12)), axis=1,
        ) / reliability_sum
        coverage = np.sum(reliability[None] * explained, axis=1) / reliability_sum
    else:
        score = np.full(
            (dense_view.shape[0],), np.log(float(missing_view_probability)),
            dtype=np.float64,
        )
        coverage = np.zeros((dense_view.shape[0],), dtype=np.float64)
    take = min(int(maximum_views), int(score.size))
    order = np.argsort(-score, kind="stable")[:take]
    logits = score[order] / float(temperature)
    if logits.size and h1_probability > 0.0:
        logits -= float(np.max(logits))
        conditional_view = np.exp(logits)
        conditional_view /= max(float(np.sum(conditional_view)), 1e-12)
        view_probability = h1_probability * conditional_view
    else:
        view_probability = np.zeros((take,), dtype=np.float64)
    return MappingViewPosterior(
        view_rows=order.astype(np.int64),
        scores=score[order].astype(np.float32),
        probabilities=view_probability.astype(np.float32),
        null_probability=float(null_probability),
        support_coverage=coverage[order].astype(np.float32),
    )


def mapping_view_pose_modes(
    graph: MappingViewGraph,
    posterior: MappingViewPosterior,
    *,
    maximum_modes: int = 32,
    translation_nms_m: float = 0.20,
    rotation_nms_deg: float = 3.0,
) -> CoarsePoseModes:
    """Convert the multi-modal view posterior to an SE(3)-diverse pose set."""

    retained: list[int] = []
    for candidate, view in enumerate(posterior.view_rows.tolist()):
        if float(posterior.probabilities[candidate]) <= 0.0:
            continue
        pose = graph.poses_w2c[int(view)]
        center = -pose[:3, :3].T @ pose[:3, 3]
        duplicate = False
        for other_candidate in retained:
            other_view = int(posterior.view_rows[other_candidate])
            other = graph.poses_w2c[other_view]
            other_center = -other[:3, :3].T @ other[:3, 3]
            if (
                np.linalg.norm(center - other_center) < float(translation_nms_m)
                and _rotation_distance_degrees(pose, other) < float(rotation_nms_deg)
            ):
                duplicate = True
                break
        if duplicate:
            continue
        retained.append(candidate)
        if len(retained) >= int(maximum_modes):
            break
    if not retained:
        return CoarsePoseModes(
            np.zeros((0, 4, 4), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
        )
    index = np.asarray(retained, dtype=np.int64)
    views = posterior.view_rows[index]
    return CoarsePoseModes(
        graph.poses_w2c[views],
        posterior.scores[index],
        np.rint(posterior.support_coverage[index] * 1000.0).astype(np.int64),
    )
