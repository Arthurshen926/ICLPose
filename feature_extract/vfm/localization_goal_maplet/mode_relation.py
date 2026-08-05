"""Sparse child-local mode relations for Goal-Maplet configuration inference.

The option universe and the query relation graph are constructed before a
candidate pose is inspected.  A pose may project and score those fixed
primitive modes, but it cannot change the denominator or select its own
evidence.  This is the central deployment contract of the relation stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np

from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion

from .latent_configuration import correlated_support_clusters
from .physical_map import GoalMapletPhysicalMap


EDGE_FAMILIES = ("local", "long_range", "depth_normal")
RELATION_NULL_TYPES = (
    "valid", "missing_endpoint", "behind_camera", "back_facing", "primitive_conflict",
)

NODE_NULL_TYPES = (
    "retrieval_null",
    "child_shortlist_omitted",
    "mode_shortlist_omitted",
    "geometry_invalid",
    "field_missing",
)

# Fixed, deliberately conservative physical-null evidence.  Query/map
# uncertainty is neutral; a state which claims a physical endpoint but places
# it behind the camera or on a back-facing surface is evidence against the
# pose.  Primitive conflicts remain impossible.  These are likelihood ratios,
# not priors and are therefore added only after the mass-conserving state prior
# has been constructed.
RELATION_NULL_LOG_LIKELIHOOD_RATIOS = np.asarray(
    [0.0, 0.0, np.log(0.05), np.log(0.05), -1.0e6], dtype=np.float64,
)

FEATURE_NAMES = (
    "normalized_vector_residual",
    "normalized_x_residual",
    "normalized_y_residual",
    "direction_cosine",
    "direction_sine_abs",
    "absolute_log_length_ratio",
    "query_distance_diagonal_fraction",
    "query_log_scale_ratio",
    "projected_log_scale_ratio",
    "absolute_scale_ratio_residual",
    "depth_order_agreement",
    "normalized_depth_difference",
    "normal_cosine",
    "bearing_diversity",
    "same_primitive",
    "same_child",
    "same_parent",
    "minimum_incidence",
    "local_edge",
    "long_range_edge",
    "depth_normal_edge",
)


@dataclass(frozen=True)
class SparseRelationEdges:
    """Candidate-independent fit tree and disjoint verification edges."""

    fit_left: np.ndarray
    fit_right: np.ndarray
    fit_family: np.ndarray
    verify_left: np.ndarray
    verify_right: np.ndarray
    verify_family: np.ndarray
    representative_groups: np.ndarray
    support_cluster_rows: np.ndarray | None = None
    legacy_connected_cluster_rows: np.ndarray | None = None

    def __post_init__(self) -> None:
        for prefix in ("fit", "verify"):
            left = np.asarray(getattr(self, f"{prefix}_left"), dtype=np.int64).reshape(-1)
            right = np.asarray(getattr(self, f"{prefix}_right"), dtype=np.int64).reshape(-1)
            family = np.asarray(getattr(self, f"{prefix}_family"), dtype=np.int64).reshape(-1)
            if left.shape != right.shape or left.shape != family.shape:
                raise ValueError("relation edge arrays differ")
            if np.any(left == right) or np.any((family < 0) | (family >= len(EDGE_FAMILIES))):
                raise ValueError("invalid sparse relation edge")
            object.__setattr__(self, f"{prefix}_left", left)
            object.__setattr__(self, f"{prefix}_right", right)
            object.__setattr__(self, f"{prefix}_family", family)
        representative = np.asarray(self.representative_groups, dtype=np.int64).reshape(-1)
        object.__setattr__(self, "representative_groups", representative)
        cluster = (
            np.asarray(self.support_cluster_rows, dtype=np.int64).reshape(-1)
            if self.support_cluster_rows is not None else np.arange(
                int(np.max(representative)) + 1 if representative.size else 0, dtype=np.int64,
            )
        )
        legacy = (
            np.asarray(self.legacy_connected_cluster_rows, dtype=np.int64).reshape(-1)
            if self.legacy_connected_cluster_rows is not None else cluster.copy()
        )
        if cluster.shape != legacy.shape:
            raise ValueError("relation support cluster diagnostics differ")
        object.__setattr__(self, "support_cluster_rows", cluster)
        object.__setattr__(self, "legacy_connected_cluster_rows", legacy)
        fit = {tuple(sorted(value)) for value in zip(self.fit_left.tolist(), self.fit_right.tolist())}
        verify = {tuple(sorted(value)) for value in zip(self.verify_left.tolist(), self.verify_right.tolist())}
        if fit & verify:
            raise ValueError("fit and verification relation evidence overlap")


def _legacy_connected_overlap_clusters(xy: np.ndarray, extent: np.ndarray) -> np.ndarray:
    """Return the retired G14 components, only for chain-collapse auditing."""

    count = xy.shape[0]
    parent = np.arange(count, dtype=np.int64)

    def root(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for left in range(count):
        low_left, high_left = xy[left] - extent[left], xy[left] + extent[left]
        for right in range(left + 1, count):
            low = np.maximum(low_left, xy[right] - extent[right])
            high = np.minimum(high_left, xy[right] + extent[right])
            intersection = float(np.prod(np.maximum(high - low, 0.0)))
            area_left = float(np.prod(np.maximum(2.0 * extent[left], 1.0)))
            area_right = float(np.prod(np.maximum(2.0 * extent[right], 1.0)))
            if intersection / max(min(area_left, area_right), 1.0) >= 0.45:
                a, b = root(left), root(right)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    roots = np.asarray([root(index) for index in range(count)], dtype=np.int64)
    _, inverse = np.unique(roots, return_inverse=True)
    return inverse.astype(np.int64)


def _maximum_spanning_tree(
    node_count: int,
    left: np.ndarray,
    right: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    parent = np.arange(node_count, dtype=np.int64)

    def root(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    # Stable lexicographic tie-breaking makes the edge set reproducible.
    order = np.lexsort((right, left, -weight))
    chosen: list[int] = []
    for edge in order.tolist():
        a, b = root(int(left[edge])), root(int(right[edge]))
        if a == b:
            continue
        parent[max(a, b)] = min(a, b)
        chosen.append(int(edge))
        if len(chosen) == max(node_count - 1, 0):
            break
    return np.asarray(chosen, dtype=np.int64)


def build_sparse_relation_edges(
    query_xy_px: np.ndarray,
    query_extent_px: np.ndarray,
    query_descriptors: np.ndarray,
    query_scale_px: np.ndarray,
    *,
    query_priority: np.ndarray | None = None,
    maximum_verify_edges: int | None = None,
) -> SparseRelationEdges:
    """Build a query-only information tree plus held-out relation edges.

    One representative is used per overlapping support cluster.  Candidate
    edges combine two local neighbours, descriptor-distinct long-range pairs,
    and support-scale diversity probes.  No child, primitive or pose is read.
    """

    xy = np.asarray(query_xy_px, dtype=np.float64).reshape(-1, 2)
    extent = np.asarray(query_extent_px, dtype=np.float64).reshape(-1, 2)
    descriptor = np.asarray(query_descriptors, dtype=np.float64)
    scale = np.asarray(query_scale_px, dtype=np.float64).reshape(-1)
    priority = (
        np.ones((xy.shape[0],), dtype=np.float64)
        if query_priority is None
        else np.asarray(query_priority, dtype=np.float64).reshape(-1)
    )
    count = xy.shape[0]
    if (
        extent.shape != xy.shape or descriptor.ndim != 2
        or descriptor.shape[0] != count or scale.size != count or priority.size != count
        or not np.all(np.isfinite(priority))
    ):
        raise ValueError("relation query evidence differs")
    if count == 0:
        empty = np.zeros((0,), dtype=np.int64)
        return SparseRelationEdges(empty, empty, empty, empty, empty, empty, empty, empty, empty)
    # Reuse the G12 complete-link definition.  Connected components are kept
    # only as an audit output so A--B--C overlap chains can be measured rather
    # than silently deleting the distant A/C relation node.
    cluster = correlated_support_clusters(xy, extent, minimum_iou=0.50)
    legacy_cluster = _legacy_connected_overlap_clusters(xy, extent)
    representatives: list[int] = []
    for value in np.unique(cluster).tolist():
        members = np.flatnonzero(cluster == int(value))
        maximum_priority = float(np.max(priority[members]))
        quality = members[np.isclose(priority[members], maximum_priority, rtol=0.0, atol=1e-12)]
        if quality.size == 1:
            representatives.append(int(quality[0]))
            continue
        # A stable query-only medoid breaks equal-quality ties without raster
        # order becoming an accidental source of physical identity.
        distance = np.linalg.norm(xy[quality, None] - xy[members][None], axis=2)
        representatives.append(int(quality[int(np.argmin(np.sum(distance, axis=1)))]))
    representative = np.asarray(representatives, dtype=np.int64)
    if representative.size < 2:
        empty = np.zeros((0,), dtype=np.int64)
        return SparseRelationEdges(
            empty, empty, empty, empty, empty, empty, representative, cluster, legacy_cluster,
        )
    point = xy[representative]
    desc = descriptor[representative]
    desc /= np.maximum(np.linalg.norm(desc, axis=1, keepdims=True), 1e-8)
    local_scale = np.maximum(scale[representative], 1.0)
    left, right = np.triu_indices(representative.size, 1)
    delta = point[right] - point[left]
    distance = np.linalg.norm(delta, axis=1)
    distance_norm = distance / max(float(np.max(distance)), 1.0)
    distinct = np.clip(0.5 * (1.0 - np.sum(desc[left] * desc[right], axis=1)), 0.0, 1.0)
    scale_diversity = np.minimum(
        np.abs(np.log(local_scale[right] / local_scale[left])) / np.log(4.0), 1.0,
    )
    # Each family contributes a sparse candidate set.  The eventual tree is
    # selected by information, while verification uses unused long/diverse
    # relations and therefore cannot validate itself with its fit evidence.
    candidate: dict[tuple[int, int], tuple[int, float]] = {}

    def add(edge: int, family: int, information: float) -> None:
        key = (int(left[edge]), int(right[edge]))
        current = candidate.get(key)
        value = (int(family), float(information))
        if current is None:
            candidate[key] = value
        elif current[0] == 0 or family == 0:
            # Spatially local is an intrinsic pair identity.  A pair may also
            # have high descriptor/scale information, but that must not relabel
            # it and silently remove the local family from the protocol.
            candidate[key] = (0, max(current[1], value[1]))
        elif value[1] > current[1] or (value[1] == current[1] and value[0] > current[0]):
            candidate[key] = value

    for node in range(representative.size):
        incident = np.flatnonzero((left == node) | (right == node))
        for edge in incident[np.argsort(distance[incident], kind="stable")[:2]].tolist():
            add(edge, 0, 1.0 + 0.25 * distinct[edge])
        long_score = distance_norm[incident] * (0.35 + 0.65 * distinct[incident])
        for edge in incident[np.argsort(-long_score, kind="stable")[:2]].tolist():
            add(edge, 1, 1.05 + float(long_score[np.flatnonzero(incident == edge)[0]]))
        diversity_score = 0.5 * distance_norm[incident] + 0.5 * scale_diversity[incident]
        for edge in incident[np.argsort(-diversity_score, kind="stable")[:2]].tolist():
            add(edge, 2, 1.10 + float(diversity_score[np.flatnonzero(incident == edge)[0]]))
    keys = sorted(candidate)
    edge_left = np.asarray([key[0] for key in keys], dtype=np.int64)
    edge_right = np.asarray([key[1] for key in keys], dtype=np.int64)
    family = np.asarray([candidate[key][0] for key in keys], dtype=np.int64)
    information = np.asarray([candidate[key][1] for key in keys], dtype=np.float64)
    tree = _maximum_spanning_tree(representative.size, edge_left, edge_right, information)
    if tree.size != representative.size - 1:
        # The complete local graph is a fail-closed connectivity fallback and
        # still depends only on query coordinates.
        complete_information = 1.0 / np.maximum(distance, 1.0)
        tree = _maximum_spanning_tree(representative.size, left, right, complete_information)
        edge_left, edge_right = left, right
        family = np.zeros(left.shape, dtype=np.int64)
        information = complete_information
    fit_pairs = {tuple(sorted((int(edge_left[row]), int(edge_right[row])))) for row in tree.tolist()}
    verify_rows = np.asarray([
        row for row in np.argsort(-information, kind="stable").tolist()
        if tuple(sorted((int(edge_left[row]), int(edge_right[row])))) not in fit_pairs
        and family[row] in (1, 2)
    ], dtype=np.int64)
    limit = int(maximum_verify_edges) if maximum_verify_edges is not None else int(tree.size)
    verify_rows = verify_rows[: max(limit, 0)]
    return SparseRelationEdges(
        fit_left=representative[edge_left[tree]],
        fit_right=representative[edge_right[tree]],
        fit_family=family[tree],
        verify_left=representative[edge_left[verify_rows]],
        verify_right=representative[edge_right[verify_rows]],
        verify_family=family[verify_rows],
        representative_groups=representative,
        support_cluster_rows=cluster,
        legacy_connected_cluster_rows=legacy_cluster,
    )


def mode_relation_runtime_features(
    query_xy_px: np.ndarray,
    query_scale_px: np.ndarray,
    edge_left_groups: np.ndarray,
    edge_right_groups: np.ndarray,
    edge_family: np.ndarray,
    left_primitive_rows: np.ndarray,
    right_primitive_rows: np.ndarray,
    left_child_rows: np.ndarray,
    right_child_rows: np.ndarray,
    pose_w2c: np.ndarray,
    camera,
    physical: GoalMapletPhysicalMap,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate fixed primitive-pair relations under one candidate pose."""

    import cv2

    xy = np.asarray(query_xy_px, dtype=np.float64).reshape(-1, 2)
    scale = np.asarray(query_scale_px, dtype=np.float64).reshape(-1)
    left_group = np.asarray(edge_left_groups, dtype=np.int64).reshape(-1)
    right_group = np.asarray(edge_right_groups, dtype=np.int64).reshape(-1)
    family = np.asarray(edge_family, dtype=np.int64).reshape(-1)
    left_primitive = np.asarray(left_primitive_rows, dtype=np.int64).reshape(-1)
    right_primitive = np.asarray(right_primitive_rows, dtype=np.int64).reshape(-1)
    left_child = np.asarray(left_child_rows, dtype=np.int64).reshape(-1)
    right_child = np.asarray(right_child_rows, dtype=np.int64).reshape(-1)
    size = left_group.size
    if any(value.size != size for value in (
        right_group, family, left_primitive, right_primitive, left_child, right_child,
    )):
        raise ValueError("relation pair arrays differ")
    feature = np.zeros((size, len(FEATURE_NAMES)), dtype=np.float64)
    valid = np.ones((size,), dtype=bool)
    null_type = np.zeros((size,), dtype=np.int64)
    missing = (
        (left_primitive < 0) | (right_primitive < 0)
        | (left_child < 0) | (right_child < 0)
    )
    valid[missing] = False
    null_type[missing] = RELATION_NULL_TYPES.index("missing_endpoint")
    safe_left = np.maximum(left_primitive, 0)
    safe_right = np.maximum(right_primitive, 0)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    point = np.concatenate([
        physical.primitive_centers[safe_left], physical.primitive_centers[safe_right],
    ], axis=0)
    matrix, distortion = camera_matrix_and_distortion(camera)
    rotation_vector, _ = cv2.Rodrigues(pose[:3, :3])
    projected, _ = cv2.projectPoints(point, rotation_vector, pose[:3, 3], matrix, distortion)
    projected = projected.reshape(2, size, 2)
    camera_xyz = point @ pose[:3, :3].T + pose[:3, 3]
    camera_xyz = camera_xyz.reshape(2, size, 3)
    camera_center = -pose[:3, :3].T @ pose[:3, 3]
    world_left, world_right = point[:size], point[size:]
    view_left = camera_center[None] - world_left
    view_right = camera_center[None] - world_right
    view_left /= np.maximum(np.linalg.norm(view_left, axis=1, keepdims=True), 1e-8)
    view_right /= np.maximum(np.linalg.norm(view_right, axis=1, keepdims=True), 1e-8)
    incidence_left = np.sum(physical.primitive_normals[safe_left] * view_left, axis=1)
    incidence_right = np.sum(physical.primitive_normals[safe_right] * view_right, axis=1)
    behind = (camera_xyz[0, :, 2] <= 0.05) | (camera_xyz[1, :, 2] <= 0.05)
    front_left = (physical.primitive_sidedness[safe_left] == 2) | (incidence_left >= 0.02)
    front_right = (physical.primitive_sidedness[safe_right] == 2) | (incidence_right >= 0.02)
    back = ~(front_left & front_right)
    conflict = (safe_left == safe_right) & ~missing
    update = valid & behind
    null_type[update] = RELATION_NULL_TYPES.index("behind_camera")
    valid[update] = False
    update = valid & back
    null_type[update] = RELATION_NULL_TYPES.index("back_facing")
    valid[update] = False
    update = valid & conflict
    null_type[update] = RELATION_NULL_TYPES.index("primitive_conflict")
    valid[update] = False

    query_delta = xy[right_group] - xy[left_group]
    map_delta = projected[1] - projected[0]
    depth_left, depth_right = camera_xyz[0, :, 2], camera_xyz[1, :, 2]
    focal = float(np.sqrt(max(matrix[0, 0] * matrix[1, 1], 1.0)))
    radius_left = np.sqrt(
        np.maximum(physical.primitive_scale1[safe_left] * physical.primitive_scale2[safe_left], 1e-8)
    )
    radius_right = np.sqrt(
        np.maximum(physical.primitive_scale1[safe_right] * physical.primitive_scale2[safe_right], 1e-8)
    )
    projected_scale_left = focal * radius_left / np.maximum(depth_left, 0.05)
    projected_scale_right = focal * radius_right / np.maximum(depth_right, 0.05)
    sigma = np.sqrt(
        np.square(scale[left_group]) + np.square(scale[right_group])
        + np.square(projected_scale_left) + np.square(projected_scale_right)
    )
    sigma = np.maximum(sigma, 8.0)
    vector_residual = map_delta - query_delta
    query_length = np.maximum(np.linalg.norm(query_delta, axis=1), 1e-6)
    map_length = np.maximum(np.linalg.norm(map_delta, axis=1), 1e-6)
    direction_cosine = np.sum(query_delta * map_delta, axis=1) / (query_length * map_length)
    direction_sine = (
        query_delta[:, 0] * map_delta[:, 1] - query_delta[:, 1] * map_delta[:, 0]
    ) / (query_length * map_length)
    query_log_scale = np.log(np.maximum(scale[right_group], 1.0) / np.maximum(scale[left_group], 1.0))
    projected_log_scale = np.log(
        np.maximum(projected_scale_right, 1e-4) / np.maximum(projected_scale_left, 1e-4)
    )
    query_depth_order = np.sign(query_log_scale)
    map_depth_order = np.sign(depth_left - depth_right)
    normal_cosine = np.sum(
        physical.primitive_normals[safe_left] * physical.primitive_normals[safe_right], axis=1,
    )
    bearing_diversity = 1.0 - np.sum(view_left * view_right, axis=1)
    parent_left = physical.child_parent_rows[np.maximum(left_child, 0)]
    parent_right = physical.child_parent_rows[np.maximum(right_child, 0)]
    diagonal = float(np.hypot(camera.width, camera.height))
    feature[:] = np.stack([
        np.linalg.norm(vector_residual, axis=1) / sigma,
        vector_residual[:, 0] / sigma,
        vector_residual[:, 1] / sigma,
        np.clip(direction_cosine, -1.0, 1.0),
        np.abs(direction_sine),
        np.abs(np.log(map_length / query_length)),
        query_length / max(diagonal, 1.0),
        query_log_scale,
        projected_log_scale,
        np.abs(projected_log_scale - query_log_scale),
        (query_depth_order == map_depth_order).astype(np.float64),
        np.abs(depth_right - depth_left) / np.maximum(0.5 * (depth_right + depth_left), 0.05),
        np.clip(normal_cosine, -1.0, 1.0),
        np.maximum(bearing_diversity, 0.0),
        conflict.astype(np.float64),
        (left_child == right_child).astype(np.float64),
        (parent_left == parent_right).astype(np.float64),
        np.minimum(incidence_left, incidence_right),
        (family == 0).astype(np.float64),
        (family == 1).astype(np.float64),
        (family == 2).astype(np.float64),
    ], axis=1)
    feature[~np.isfinite(feature)] = 0.0
    feature[missing] = 0.0
    return feature.astype(np.float32), valid, null_type


def analytic_relation_score(features: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fixed, parameter-free compatibility diagnostic used before learning."""

    value = np.asarray(features, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    if value.ndim != 2 or value.shape[1] != len(FEATURE_NAMES) or value.shape[0] != mask.size:
        raise ValueError("analytic relation inputs differ")
    index = {name: row for row, name in enumerate(FEATURE_NAMES)}
    # All terms have a direct optimum (zero residual or unit agreement); this
    # is an audit statistic, not a fitted fusion weight.
    loss = (
        np.minimum(value[:, index["normalized_vector_residual"]], 6.0)
        + np.minimum(value[:, index["absolute_log_length_ratio"]], 3.0)
        + np.minimum(value[:, index["absolute_scale_ratio_residual"]], 3.0)
        + 0.5 * (1.0 - value[:, index["direction_cosine"]])
        + 0.5 * (1.0 - value[:, index["depth_order_agreement"]])
    )
    score = -loss
    score[~mask] = 0.0
    return score


@dataclass(frozen=True)
class ModeRelationLikelihoodRatioArtifact:
    pair_estimator: object
    calibration_scale: float
    calibration_intercept: float
    metadata: Mapping[str, object]

    def score_log_likelihood_ratio(self, features: np.ndarray) -> np.ndarray:
        value = np.asarray(features, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != len(FEATURE_NAMES):
            raise ValueError("mode-relation feature dimension differs")
        if value.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)
        raw = np.asarray(self.pair_estimator.decision_function(value), dtype=np.float64).reshape(-1)
        score = float(self.calibration_scale) * raw + float(self.calibration_intercept)
        if not np.all(np.isfinite(score)):
            raise ValueError("mode-relation likelihood ratio contains non-finite values")
        return score

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pair_estimator": self.pair_estimator,
            "calibration_scale": float(self.calibration_scale),
            "calibration_intercept": float(self.calibration_intercept),
            "metadata": dict(self.metadata),
        }, Path(path))

    @classmethod
    def load(cls, path: Path) -> "ModeRelationLikelihoodRatioArtifact":
        payload = joblib.load(Path(path))
        metadata = dict(payload["metadata"])
        if metadata.get("artifact_type") != "goal_maplet_mode_relation_likelihood_ratio_v1":
            raise ValueError("not a Goal-Maplet mode-relation likelihood ratio")
        if tuple(metadata.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("mode-relation feature contract differs")
        if metadata.get("pairing_contract") != "same_image_same_query_edge_fixed_options_v1":
            raise ValueError("mode-relation pairing contract differs")
        if metadata.get("edge_contract") not in (
            "query_only_fit_tree_disjoint_verify_v1",
            "query_only_complete_link_fit_tree_disjoint_verify_v2",
        ):
            raise ValueError("mode-relation edge contract differs")
        return cls(
            payload["pair_estimator"], float(payload["calibration_scale"]),
            float(payload["calibration_intercept"]), metadata,
        )


@dataclass(frozen=True)
class TreeInferenceResult:
    score: float
    state_rows: np.ndarray


@dataclass(frozen=True)
class TreeMarginalResult:
    log_partition: float
    node_log_marginals: tuple[np.ndarray, ...]
    directed_log_messages: Mapping[tuple[int, int], np.ndarray]


def _logsumexp(value: np.ndarray, axis=None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    maximum = np.max(array, axis=axis, keepdims=True)
    maximum[~np.isfinite(maximum)] = 0.0
    result = maximum + np.log(np.maximum(np.sum(np.exp(array - maximum), axis=axis, keepdims=True), 1e-300))
    if axis is None:
        return np.asarray(result).reshape(())
    return np.squeeze(result, axis=axis)


def sum_product_forest(
    unary_scores: Sequence[np.ndarray],
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    pair_scores: Sequence[np.ndarray],
) -> TreeMarginalResult:
    """Exact log-partition and node marginals on a heterogeneous forest."""

    unary = [np.asarray(value, dtype=np.float64).reshape(-1) for value in unary_scores]
    count = len(unary)
    left = np.asarray(edge_left, dtype=np.int64).reshape(-1)
    right = np.asarray(edge_right, dtype=np.int64).reshape(-1)
    if left.shape != right.shape or len(pair_scores) != left.size:
        raise ValueError("tree relation arrays differ")
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(count)]
    matrices = []
    for edge, (a, b) in enumerate(zip(left.tolist(), right.tolist())):
        if a < 0 or b < 0 or a >= count or b >= count or a == b:
            raise ValueError("invalid tree endpoint")
        matrix = np.asarray(pair_scores[edge], dtype=np.float64)
        if matrix.shape != (unary[a].size, unary[b].size):
            raise ValueError("tree pair potential shape differs")
        matrices.append(matrix)
        adjacency[a].append((b, edge))
        adjacency[b].append((a, edge))
    message: dict[tuple[int, int], np.ndarray] = {}
    components: list[tuple[int, list[int], np.ndarray]] = []
    visited = np.zeros((count,), dtype=bool)
    for root in range(count):
        if visited[root]:
            continue
        parent = np.full((count,), -1, dtype=np.int64)
        order, stack = [], [root]
        visited[root] = True
        while stack:
            node = stack.pop()
            order.append(node)
            for neighbour, _ in adjacency[node]:
                if neighbour == parent[node]:
                    continue
                if visited[neighbour]:
                    raise ValueError("sum-product graph contains a cycle")
                visited[neighbour] = True
                parent[neighbour] = node
                stack.append(neighbour)
        components.append((root, order, parent))
        for node in reversed(order[1:]):
            destination = int(parent[node])
            local = unary[node].copy()
            for neighbour, _ in adjacency[node]:
                if neighbour != destination:
                    local += message[(neighbour, node)]
            edge = next(edge for neighbour, edge in adjacency[node] if neighbour == destination)
            matrix = matrices[edge]
            oriented = matrix if int(left[edge]) == destination else matrix.T
            message[(node, destination)] = _logsumexp(oriented + local[None, :], axis=1)
        for node in order:
            for destination, edge in adjacency[node]:
                if int(parent[node]) == destination:
                    continue
                local = unary[node].copy()
                for neighbour, _ in adjacency[node]:
                    if neighbour != destination:
                        local += message[(neighbour, node)]
                matrix = matrices[edge]
                oriented = matrix if int(left[edge]) == node else matrix.T
                message[(node, destination)] = _logsumexp(oriented + local[:, None], axis=0)
    marginal = []
    for node in range(count):
        belief = unary[node].copy()
        for neighbour, _ in adjacency[node]:
            belief += message[(neighbour, node)]
        marginal.append(belief - float(_logsumexp(belief)))
    log_partition = 0.0
    for root, _, _ in components:
        belief = unary[root].copy()
        for neighbour, _ in adjacency[root]:
            belief += message[(neighbour, root)]
        log_partition += float(_logsumexp(belief))
    return TreeMarginalResult(log_partition, tuple(marginal), message)


def exact_pair_log_marginal(
    inference: TreeMarginalResult,
    unary_scores: Sequence[np.ndarray],
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    pair_scores: Sequence[np.ndarray],
    node_left: int,
    node_right: int,
) -> np.ndarray:
    """Return exact ``log P(z_left,z_right | fit tree)``.

    Verification edges are usually not fit-tree neighbours.  Multiplying two
    node marginals discards every dependency along their tree path.  This
    routine integrates the off-path subtrees through the already-computed
    directed messages and eliminates every internal path state exactly.
    """

    unary = [np.asarray(value, dtype=np.float64).reshape(-1) for value in unary_scores]
    left = np.asarray(edge_left, dtype=np.int64).reshape(-1)
    right = np.asarray(edge_right, dtype=np.int64).reshape(-1)
    a, b = int(node_left), int(node_right)
    if a < 0 or b < 0 or a >= len(unary) or b >= len(unary) or a == b:
        raise ValueError("invalid pair-marginal endpoints")
    adjacency: list[list[tuple[int, int]]] = [[] for _ in unary]
    for edge, (u, v) in enumerate(zip(left.tolist(), right.tolist())):
        adjacency[u].append((v, edge))
        adjacency[v].append((u, edge))
    parent = np.full((len(unary),), -1, dtype=np.int64)
    parent_edge = np.full((len(unary),), -1, dtype=np.int64)
    queue = [a]
    parent[a] = a
    for node in queue:
        if node == b:
            break
        for neighbour, edge in adjacency[node]:
            if parent[neighbour] >= 0:
                continue
            parent[neighbour] = node
            parent_edge[neighbour] = edge
            queue.append(neighbour)
    if parent[b] < 0:
        # Distinct forest components are independent after conditioning on the
        # fit evidence.  This is exact, unlike using the same approximation for
        # connected endpoints.
        output = (
            inference.node_log_marginals[a][:, None]
            + inference.node_log_marginals[b][None, :]
        )
        return output - float(_logsumexp(output))
    path = [b]
    while path[-1] != a:
        path.append(int(parent[path[-1]]))
    path.reverse()

    def cavity(node: int, excluded: set[int]) -> np.ndarray:
        value = unary[node].copy()
        for neighbour, _ in adjacency[node]:
            if neighbour not in excluded:
                value += np.asarray(
                    inference.directed_log_messages[(neighbour, node)], dtype=np.float64,
                )
        return value

    first, second = path[0], path[1]
    first_edge = int(parent_edge[second])
    matrix = np.asarray(pair_scores[first_edge], dtype=np.float64)
    oriented = matrix if int(left[first_edge]) == first else matrix.T
    transfer = cavity(first, {second})[:, None] + oriented
    for position in range(1, len(path) - 1):
        node, destination = path[position], path[position + 1]
        previous = path[position - 1]
        edge = int(parent_edge[destination])
        matrix = np.asarray(pair_scores[edge], dtype=np.float64)
        oriented = matrix if int(left[edge]) == node else matrix.T
        local = cavity(node, {previous, destination})
        # transfer axes are (left endpoint, current path state).
        transfer = _logsumexp(
            transfer[:, :, None] + local[None, :, None] + oriented[None, :, :],
            axis=1,
        )
    last, previous = path[-1], path[-2]
    transfer += cavity(last, {previous})[None, :]
    return transfer - float(_logsumexp(transfer))


def max_sum_forest(
    unary_scores: Sequence[np.ndarray],
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    pair_scores: Sequence[np.ndarray],
) -> TreeInferenceResult:
    """Exact max-sum on a tree/forest with heterogeneous state counts."""

    unary = [np.asarray(value, dtype=np.float64).reshape(-1) for value in unary_scores]
    count = len(unary)
    left = np.asarray(edge_left, dtype=np.int64).reshape(-1)
    right = np.asarray(edge_right, dtype=np.int64).reshape(-1)
    if left.shape != right.shape or len(pair_scores) != left.size:
        raise ValueError("tree relation arrays differ")
    adjacency: list[list[tuple[int, int, bool]]] = [[] for _ in range(count)]
    for edge, (a, b) in enumerate(zip(left.tolist(), right.tolist())):
        if a < 0 or b < 0 or a >= count or b >= count or a == b:
            raise ValueError("invalid tree endpoint")
        matrix = np.asarray(pair_scores[edge], dtype=np.float64)
        if matrix.shape != (unary[a].size, unary[b].size):
            raise ValueError("tree pair potential shape differs")
        adjacency[a].append((b, edge, True))
        adjacency[b].append((a, edge, False))
    state = np.full((count,), -1, dtype=np.int64)
    total = 0.0
    visited = np.zeros((count,), dtype=bool)
    for root in range(count):
        if visited[root]:
            continue
        parent = np.full((count,), -1, dtype=np.int64)
        parent_edge = np.full((count,), -1, dtype=np.int64)
        parent_forward = np.zeros((count,), dtype=bool)
        order, stack = [], [root]
        visited[root] = True
        while stack:
            node = stack.pop()
            order.append(node)
            for neighbour, edge, forward in adjacency[node]:
                if neighbour == parent[node]:
                    continue
                if visited[neighbour]:
                    raise ValueError("max-sum graph contains a cycle")
                visited[neighbour] = True
                parent[neighbour] = node
                parent_edge[neighbour] = edge
                parent_forward[neighbour] = forward
                stack.append(neighbour)
        accumulated = [value.copy() for value in unary]
        backpointer: dict[int, np.ndarray] = {}
        for node in reversed(order[1:]):
            edge = int(parent_edge[node])
            matrix = np.asarray(pair_scores[edge], dtype=np.float64)
            # parent_forward records whether traversal parent->child follows
            # stored left->right orientation.
            if parent_forward[node]:
                # Traversal node(parent endpoint in adjacency) -> neighbour
                # set this flag on the child from the parent's adjacency.
                oriented = matrix
            else:
                oriented = matrix.T
            # The flag above is stored from the parent side in construction.
            # Recover orientation robustly from explicit endpoints.
            p = int(parent[node])
            oriented = matrix if int(left[edge]) == p else matrix.T
            value = oriented + accumulated[node][None, :]
            backpointer[node] = np.argmax(value, axis=1).astype(np.int64)
            accumulated[p] += np.max(value, axis=1)
        state[root] = int(np.argmax(accumulated[root]))
        total += float(accumulated[root][state[root]])
        for node in order[1:]:
            state[node] = int(backpointer[node][state[int(parent[node])]])
    return TreeInferenceResult(score=total, state_rows=state)


RELATION_EVIDENCE_NAMES = (
    "relation_tree_fit_score_mean",
    "relation_verify_llr_median",
    "relation_verify_valid_fraction",
    "relation_fit_valid_fraction",
    "relation_assigned_group_fraction",
    "relation_unique_child_fraction",
    "relation_local_llr_median",
    "relation_long_range_llr_median",
    "relation_depth_normal_llr_median",
    "relation_null_edge_fraction",
    "relation_node_log_evidence_mean",
    "relation_fit_incremental_llr_mean",
    "relation_verify_predictive_llr_mean",
    "relation_posterior_non_null_mass_mean",
    "relation_posterior_entropy_mean",
    "relation_maxsum_non_null_fraction",
    "relation_prior_retrieval_null_mass_mean",
    "relation_prior_child_omitted_mass_mean",
    "relation_prior_mode_omitted_mass_mean",
    "relation_prior_geometry_invalid_mass_mean",
    "relation_prior_field_missing_mass_mean",
    "relation_prior_mass_residual_max",
    "relation_complete_link_node_count",
    "relation_legacy_connected_node_count",
    "relation_chain_restored_node_count",
)


@dataclass(frozen=True)
class RelationConfigurationEvidence:
    features: np.ndarray
    selected_child_rows: np.ndarray
    selected_primitive_rows: np.ndarray
    relation_edges: SparseRelationEdges
    query_diagnostics: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        value = np.asarray(self.features, dtype=np.float32)
        child = np.asarray(self.selected_child_rows, dtype=np.int64)
        primitive = np.asarray(self.selected_primitive_rows, dtype=np.int64)
        if value.ndim != 2 or value.shape[1] != len(RELATION_EVIDENCE_NAMES):
            raise ValueError("relation configuration feature shape differs")
        if child.shape != primitive.shape or child.shape[0] != value.shape[0]:
            raise ValueError("relation configuration assignment shape differs")


def _relation_pair_potential(
    query_xy: np.ndarray,
    query_scale: np.ndarray,
    left: int,
    right: int,
    family: int,
    state_child: Sequence[np.ndarray],
    state_primitive: Sequence[np.ndarray],
    state_null_type: Sequence[np.ndarray],
    pose: np.ndarray,
    camera,
    physical: GoalMapletPhysicalMap,
    likelihood_ratio: ModeRelationLikelihoodRatioArtifact,
) -> tuple[np.ndarray, np.ndarray]:
    left_count, right_count = state_child[left].size, state_child[right].size
    left_state = np.repeat(np.arange(left_count), right_count)
    right_state = np.tile(np.arange(right_count), left_count)
    feature, relation_valid, null_type = mode_relation_runtime_features(
        query_xy, query_scale,
        np.full(left_state.shape, left), np.full(left_state.shape, right),
        np.full(left_state.shape, family),
        state_primitive[left][left_state], state_primitive[right][right_state],
        state_child[left][left_state], state_child[right][right_state],
        pose, camera, physical,
    )
    score = RELATION_NULL_LOG_LIKELIHOOD_RATIOS[null_type].copy()
    score[relation_valid] = likelihood_ratio.score_log_likelihood_ratio(feature[relation_valid])
    # A missing pair caused by ordinary retrieval/shortlist uncertainty is
    # unobserved, not contradictory.  Geometry-invalid node states are
    # different: they are candidate-induced and retain their fixed coverage
    # loss instead of making a wrong pose's edge disappear for free.
    missing = null_type == RELATION_NULL_TYPES.index("missing_endpoint")
    if np.any(missing):
        node_null_llr = np.zeros((len(NODE_NULL_TYPES),), dtype=np.float64)
        node_null_llr[NODE_NULL_TYPES.index("geometry_invalid")] = np.log(0.05)
        left_null = state_null_type[left][left_state]
        right_null = state_null_type[right][right_state]
        missing_score = np.zeros(left_state.shape, dtype=np.float64)
        has_left = left_null >= 0
        has_right = right_null >= 0
        missing_score[has_left] += node_null_llr[left_null[has_left]]
        missing_score[has_right] += node_null_llr[right_null[has_right]]
        score[missing] = missing_score[missing]
    return score.reshape(left_count, right_count), relation_valid.reshape(left_count, right_count)


def family_preserving_child_shortlist(
    child_rows: np.ndarray,
    child_probabilities: np.ndarray,
    physical: GoalMapletPhysicalMap,
    *,
    maximum_children: int = 8,
    maximum_per_parent: int = 2,
) -> np.ndarray:
    """Return a fixed query-only mask that preserves distinct parent families."""

    child = np.asarray(child_rows, dtype=np.int64)
    probability = np.asarray(child_probabilities, dtype=np.float64)
    if child.shape != probability.shape or child.ndim != 2:
        raise ValueError("child shortlist arrays differ")
    output = np.zeros(child.shape, dtype=bool)
    for group in range(child.shape[0]):
        used: dict[int, int] = {}
        count = 0
        order = np.argsort(-probability[group], kind="stable")
        for slot in order.tolist():
            current = int(child[group, slot])
            if current < 0 or probability[group, slot] <= 0.0:
                continue
            parent = int(physical.child_parent_rows[current])
            if used.get(parent, 0) >= int(maximum_per_parent):
                continue
            output[group, slot] = True
            used[parent] = used.get(parent, 0) + 1
            count += 1
            if count >= int(maximum_children):
                break
    return output


def configuration_mode_relation_evidence(
    poses_w2c: np.ndarray,
    query_descriptors: np.ndarray,
    query_xy_px: np.ndarray,
    query_extent_px: np.ndarray,
    query_scale_px: np.ndarray,
    parent_candidate_ids: np.ndarray,
    parent_probabilities: np.ndarray,
    parent_null_probabilities: np.ndarray,
    child_posterior,
    physical: GoalMapletPhysicalMap,
    field,
    eligibility,
    pose_likelihood_ratio,
    relation_likelihood_ratio: ModeRelationLikelihoodRatioArtifact,
    camera,
    *,
    maximum_groups: int = 64,
    retrieval_maximum_children: int = 16,
    shortlist_maximum_children: int = 8,
    shortlist_children_per_parent: int = 2,
    maximum_modes: int = 8,
    shortlist_modes_per_child: int = 2,
    temperature: float = 0.07,
) -> RelationConfigurationEvidence:
    """Mass-conserving exact fit-tree inference and held-out prediction."""

    from .child_local_factor import child_local_factor_runtime_features
    from .child_local_likelihood import predict_child_local_surface_likelihood
    from .child_local_mode_ranker import child_local_mode_runtime_features
    from .latent_configuration import effective_group_weights

    poses = np.asarray(poses_w2c, dtype=np.float64).reshape(-1, 4, 4)
    descriptor = np.asarray(query_descriptors, dtype=np.float32)
    xy = np.asarray(query_xy_px, dtype=np.float64).reshape(-1, 2)
    extent = np.asarray(query_extent_px, dtype=np.float64).reshape(-1, 2)
    scale = np.asarray(query_scale_px, dtype=np.float64).reshape(-1)
    parent_ids = np.asarray(parent_candidate_ids, dtype=np.int64)
    parent_probability = np.asarray(parent_probabilities, dtype=np.float64)
    parent_null = np.asarray(parent_null_probabilities, dtype=np.float64).reshape(-1)
    if descriptor.shape[0] != xy.shape[0] or xy.shape != extent.shape or scale.size != xy.shape[0]:
        raise ValueError("relation configuration query evidence differs")
    priority = 1.0 - parent_null
    groups = np.argsort(-priority, kind="stable")[: int(maximum_groups)]
    groups = groups[priority[groups] > 0.02]
    output = np.zeros((poses.shape[0], len(RELATION_EVIDENCE_NAMES)), dtype=np.float32)
    assignments = np.full((poses.shape[0], groups.size), -1, dtype=np.int64)
    primitive_assignments = np.full_like(assignments, -1)
    empty_edges = build_sparse_relation_edges(
        xy[groups], extent[groups], descriptor[groups], scale[groups],
        query_priority=priority[groups],
    ) if groups.size else SparseRelationEdges(*([np.zeros(0, dtype=np.int64)] * 9))
    if groups.size == 0:
        return RelationConfigurationEvidence(output, assignments, primitive_assignments, empty_edges)
    group_xy, group_extent, group_descriptor, group_scale = (
        xy[groups], extent[groups], descriptor[groups], scale[groups],
    )
    edges = build_sparse_relation_edges(
        group_xy, group_extent, group_descriptor, group_scale,
        query_priority=priority[groups],
    )
    cluster = np.asarray(edges.support_cluster_rows, dtype=np.int64)
    group_weight = effective_group_weights(cluster)
    child_rows = np.asarray(
        child_posterior.candidate_child_rows[groups, : int(retrieval_maximum_children)], dtype=np.int64,
    )
    child_probability = np.asarray(
        child_posterior.candidate_probabilities[groups, : int(retrieval_maximum_children)], dtype=np.float64,
    )
    shortlist = family_preserving_child_shortlist(
        child_rows, child_probability, physical,
        maximum_children=int(shortlist_maximum_children),
        maximum_per_parent=int(shortlist_children_per_parent),
    )
    shortlist &= child_rows >= 0
    shortlist &= eligibility.proposal_qualified[np.maximum(child_rows, 0)]
    group_index, child_slot = np.nonzero(shortlist)
    if group_index.size == 0:
        return RelationConfigurationEvidence(output, assignments, primitive_assignments, edges)
    selected_child = child_rows[group_index, child_slot]
    # Normalize the complete retrieval posterior exactly once.  Runtime Top-16
    # and the family shortlist only move omitted mass into an explicit null;
    # they never renormalize the surviving child identities.
    all_child_rows = np.asarray(child_posterior.candidate_child_rows[groups], dtype=np.int64)
    all_child_probability = np.asarray(child_posterior.candidate_probabilities[groups], dtype=np.float64)
    all_child_probability = np.where(all_child_rows >= 0, all_child_probability, 0.0)
    raw_retrieval_null = np.asarray(child_posterior.null_probabilities[groups], dtype=np.float64)
    posterior_total = raw_retrieval_null + np.sum(all_child_probability, axis=1)
    posterior_total = np.maximum(posterior_total, 1e-12)
    normalized_all_child_probability = all_child_probability / posterior_total[:, None]
    retrieval_null_mass = raw_retrieval_null / posterior_total
    selected_child_probability = child_probability[group_index, child_slot] / posterior_total[group_index]
    selected_parent_rows = physical.child_parent_rows[selected_child]
    selected_parent_ids = physical.maplet_ids[selected_parent_rows]
    selected_parent_probability = np.asarray([
        np.sum(parent_probability[groups[group]][parent_ids[groups[group]] == parent_id])
        for group, parent_id in zip(group_index.tolist(), selected_parent_ids.tolist())
    ], dtype=np.float64)
    likelihood = predict_child_local_surface_likelihood(
        group_descriptor[group_index], selected_child, physical, field,
        temperature=float(temperature), maximum_modes=int(maximum_modes),
    )
    primitive_matrix = np.asarray(likelihood.mode_primitive_rows, dtype=np.int64)
    probability_matrix = np.asarray(likelihood.mode_probabilities, dtype=np.float64)
    fixed_mode = (
        (primitive_matrix >= 0) & (probability_matrix > 0.0)
        & (np.arange(primitive_matrix.shape[1])[None] < int(shortlist_modes_per_child))
    )
    factor_row_fixed, mode_row_fixed = np.nonzero(fixed_mode)
    fixed_group = group_index[factor_row_fixed]
    fixed_child = selected_child[factor_row_fixed]
    fixed_primitive = primitive_matrix[factor_row_fixed, mode_row_fixed]
    if fixed_group.size == 0:
        return RelationConfigurationEvidence(output, assignments, primitive_assignments, edges)
    for pose_row, pose in enumerate(poses):
        mode_feature, mode_valid = child_local_mode_runtime_features(
            likelihood, selected_child, group_xy[group_index], group_scale[group_index],
            pose, camera, physical, field,
        )
        factor_feature = child_local_factor_runtime_features(
            likelihood, mode_feature, mode_valid, selected_child, physical,
            parent_probability=selected_parent_probability,
            child_probability=selected_child_probability,
            parent_null_probability=parent_null[groups[group_index]],
            query_scale_px=group_scale[group_index],
            image_diagonal_px=float(np.hypot(camera.width, camera.height)),
        )
        factor_llr = pose_likelihood_ratio.score_log_likelihood_ratio(factor_feature)
        # Construct one common state measure.  Canonical coverage, omitted
        # modes and candidate geometry can only transfer mass into typed nulls.
        # The sum of all non-null and null priors is one for every group/pose.
        coverage = np.clip(1.0 - np.asarray(likelihood.null_probabilities, dtype=np.float64), 0.0, 1.0)
        retained_mode_probability = np.zeros_like(probability_matrix)
        retained_mode_probability[fixed_mode] = probability_matrix[fixed_mode]
        retained_mode_mass = np.sum(retained_mode_probability, axis=1)
        fixed_geometry_probability = np.exp(np.clip(
            np.asarray(mode_feature, dtype=np.float64)[factor_row_fixed, mode_row_fixed, 5],
            -60.0, 0.0,
        ))
        fixed_geometry_probability *= mode_valid[factor_row_fixed, mode_row_fixed]
        fixed_base_mass = (
            selected_child_probability[factor_row_fixed]
            * coverage[factor_row_fixed]
            * probability_matrix[factor_row_fixed, mode_row_fixed]
        )
        fixed_non_null_mass = fixed_base_mass * fixed_geometry_probability
        fixed_geometry_invalid_mass = fixed_base_mass - fixed_non_null_mass

        selected_child_mass = np.bincount(
            group_index, weights=selected_child_probability, minlength=groups.size,
        )
        child_omitted_mass = np.maximum(
            np.sum(normalized_all_child_probability, axis=1) - selected_child_mass, 0.0,
        )
        field_missing_mass = np.bincount(
            group_index,
            weights=selected_child_probability * (1.0 - coverage),
            minlength=groups.size,
        )
        mode_omitted_mass = np.bincount(
            group_index,
            weights=selected_child_probability * coverage * np.maximum(1.0 - retained_mode_mass, 0.0),
            minlength=groups.size,
        )
        geometry_invalid_mass = np.bincount(
            fixed_group, weights=fixed_geometry_invalid_mass, minlength=groups.size,
        )
        unary: list[np.ndarray] = []
        state_child: list[np.ndarray] = []
        state_primitive: list[np.ndarray] = []
        state_null_type: list[np.ndarray] = []
        state_prior_mass: list[np.ndarray] = []
        null_mass_matrix = np.stack([
            retrieval_null_mass, child_omitted_mass, mode_omitted_mass,
            geometry_invalid_mass, field_missing_mass,
        ], axis=1)
        mass_residual = np.zeros((groups.size,), dtype=np.float64)
        for group in range(groups.size):
            current = np.flatnonzero(fixed_group == group)
            non_null = fixed_non_null_mass[current]
            prior = np.concatenate([non_null, null_mass_matrix[group]])
            total = float(np.sum(prior))
            mass_residual[group] = abs(total - 1.0)
            if total <= 1e-12:
                prior[-len(NODE_NULL_TYPES) + NODE_NULL_TYPES.index("retrieval_null")] = 1.0
                total = 1.0
            prior /= total
            score = np.log(np.maximum(prior, 1e-300))
            if current.size:
                score[: current.size] += group_weight[group] * factor_llr[factor_row_fixed[current]]
            unary.append(score)
            state_prior_mass.append(prior)
            state_child.append(np.concatenate([
                fixed_child[current], np.full((len(NODE_NULL_TYPES),), -1, dtype=np.int64),
            ]))
            state_primitive.append(np.concatenate([
                fixed_primitive[current], np.full((len(NODE_NULL_TYPES),), -1, dtype=np.int64),
            ]))
            state_null_type.append(np.concatenate([
                np.full((current.size,), -1, dtype=np.int64),
                np.arange(len(NODE_NULL_TYPES), dtype=np.int64),
            ]))

        pair_matrices: list[np.ndarray] = []
        pair_valid_masks: list[np.ndarray] = []
        for left, right, family in zip(
            edges.fit_left.tolist(), edges.fit_right.tolist(), edges.fit_family.tolist(),
        ):
            matrix, relation_valid = _relation_pair_potential(
                group_xy, group_scale, left, right, family,
                state_child, state_primitive, state_null_type,
                pose, camera, physical, relation_likelihood_ratio,
            )
            pair_matrices.append(matrix)
            pair_valid_masks.append(relation_valid)
        node_log_partition = float(sum(float(_logsumexp(value)) for value in unary))
        result = max_sum_forest(
            unary, edges.fit_left, edges.fit_right, pair_matrices,
        )
        marginal = sum_product_forest(
            unary, edges.fit_left, edges.fit_right, pair_matrices,
        )
        selected_child_for_pose = np.asarray([
            state_child[group][result.state_rows[group]] for group in range(groups.size)
        ], dtype=np.int64)
        selected_primitive_for_pose = np.asarray([
            state_primitive[group][result.state_rows[group]] for group in range(groups.size)
        ], dtype=np.int64)
        assignments[pose_row] = selected_child_for_pose
        primitive_assignments[pose_row] = selected_primitive_for_pose
        fit_valid_mass = []
        for edge, (left, right) in enumerate(zip(edges.fit_left.tolist(), edges.fit_right.tolist())):
            log_joint = exact_pair_log_marginal(
                marginal, unary, edges.fit_left, edges.fit_right, pair_matrices, left, right,
            )
            fit_valid_mass.append(float(np.sum(np.exp(log_joint)[pair_valid_masks[edge]])))
        verify_scores, verify_families = [], []
        verify_valid_mass, null_mass = [], []
        for left, right, family in zip(
            edges.verify_left.tolist(), edges.verify_right.tolist(), edges.verify_family.tolist(),
        ):
            pair_score, relation_valid = _relation_pair_potential(
                group_xy, group_scale, left, right, family,
                state_child, state_primitive, state_null_type,
                pose, camera, physical, relation_likelihood_ratio,
            )
            log_joint = exact_pair_log_marginal(
                marginal, unary, edges.fit_left, edges.fit_right, pair_matrices, left, right,
            )
            predictive = float(_logsumexp(log_joint + pair_score))
            verify_scores.append(predictive)
            verify_families.append(int(family))
            probability = np.exp(log_joint)
            verify_valid_mass.append(float(np.sum(probability[relation_valid])))
            null_mass.append(float(np.sum(probability[~relation_valid])))
        verify_array = np.asarray(verify_scores, dtype=np.float64)
        family_array = np.asarray(verify_families, dtype=np.int64)
        family_median = [
            float(np.median(verify_array[family_array == family]))
            if np.any(family_array == family) else 0.0
            for family in range(len(EDGE_FAMILIES))
        ]
        assigned = selected_child_for_pose >= 0
        posterior_non_null = []
        posterior_entropy = []
        for group in range(groups.size):
            probability = np.exp(marginal.node_log_marginals[group])
            posterior_non_null.append(float(np.sum(probability[state_null_type[group] < 0])))
            posterior_entropy.append(float(-np.sum(probability * np.log(np.maximum(probability, 1e-12)))))
        normalizer = max(float(np.sum(group_weight)) + edges.fit_left.size, 1.0)
        complete_count = int(np.unique(edges.support_cluster_rows).size)
        legacy_count = int(np.unique(edges.legacy_connected_cluster_rows).size)
        output[pose_row] = np.asarray([
            float(marginal.log_partition / normalizer),
            float(np.median(verify_array)) if verify_array.size else 0.0,
            float(np.mean(verify_valid_mass)) if verify_valid_mass else 0.0,
            float(np.mean(fit_valid_mass)) if fit_valid_mass else 0.0,
            float(np.mean(assigned)),
            float(np.unique(selected_child_for_pose[assigned]).size / max(int(np.sum(assigned)), 1)),
            *family_median,
            float(np.mean(null_mass)) if null_mass else 1.0,
            float(node_log_partition / max(float(np.sum(group_weight)), 1.0)),
            float((marginal.log_partition - node_log_partition) / max(edges.fit_left.size, 1)),
            float(np.mean(verify_array)) if verify_array.size else 0.0,
            float(np.mean(posterior_non_null)),
            float(np.mean(posterior_entropy)),
            float(np.mean(assigned)),
            *np.mean(null_mass_matrix, axis=0).tolist(),
            float(np.max(mass_residual)),
            float(complete_count), float(legacy_count), float(max(complete_count - legacy_count, 0)),
        ], dtype=np.float32)
    if not np.all(np.isfinite(output)):
        raise ValueError("relation configuration evidence contains non-finite values")
    query_diagnostics = {
        "selected_group_rows": groups.tolist(),
        "selected_group_count": int(groups.size),
        "complete_link_relation_node_count": int(np.unique(edges.support_cluster_rows).size),
        "legacy_connected_relation_node_count": int(np.unique(edges.legacy_connected_cluster_rows).size),
        "chain_restored_relation_node_count": int(max(
            np.unique(edges.support_cluster_rows).size
            - np.unique(edges.legacy_connected_cluster_rows).size,
            0,
        )),
        "retrieval_top16_probability_mass_mean": float(np.mean(np.sum(child_probability, axis=1))),
        "family_shortlist_probability_mass_mean": float(np.mean(selected_child_mass)),
        "fixed_mode_top2_probability_mass_mean": float(np.mean(retained_mode_mass)),
        "probability_mass_contract": "sum_non_null_plus_five_typed_nulls_equals_one_v2",
    }
    return RelationConfigurationEvidence(
        output, assignments, primitive_assignments, edges, query_diagnostics,
    )
