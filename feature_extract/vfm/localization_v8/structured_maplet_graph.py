"""Image-free physical-maplet graph and pose-conditioned query alignment.

The graph artifact deliberately contains no descriptor array.  It references
the one canonical RADIO localization bank by hash and stores only physical
geometry and sparse relations.  Query nodes retain every regional posterior;
two observations assigned to the same maplet are never averaged together.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.special import logsumexp

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.continuous_surface_alignment import (
    project_world_points,
)
from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.localization_v6.maplet_retrieval import (
    MapletRetrievalResult,
)
from feature_extract.vfm.localization_v6.se3_update import se3_exp


GRAPH_ARTIFACT = "v8_single_feature_physical_maplet_graph"
GRAPH_VERSION = 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class PhysicalMapletGraph:
    """Sparse multi-scale graph referencing, but not duplicating, VFM data."""

    maplet_ids: np.ndarray
    centers: np.ndarray
    normals: np.ndarray
    tangent_frames: np.ndarray
    extents: np.ndarray
    edge_source: np.ndarray
    edge_target: np.ndarray
    # dx,dy,dz,distance,normal cosine,log area ratio,scale class
    edge_features: np.ndarray
    feature_bank_sha256: str
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        ids = np.asarray(self.maplet_ids, dtype=np.int64).reshape(-1)
        count = int(ids.size)
        centers = np.asarray(self.centers, dtype=np.float32)
        normals = np.asarray(self.normals, dtype=np.float32)
        frames = np.asarray(self.tangent_frames, dtype=np.float32)
        extents = np.asarray(self.extents, dtype=np.float32)
        source = np.asarray(self.edge_source, dtype=np.int32).reshape(-1)
        target = np.asarray(self.edge_target, dtype=np.int32).reshape(-1)
        features = np.asarray(self.edge_features, dtype=np.float32)
        if np.unique(ids).size != count:
            raise ValueError("physical maplet IDs must be unique")
        if centers.shape != (count, 3) or normals.shape != (count, 3):
            raise ValueError("maplet center/normal shapes differ")
        if frames.shape not in ((count, 2, 3), (count, 3, 3)) or extents.shape not in (
            (count, 2),
            (count, 3),
        ):
            raise ValueError("maplet frame/extent shapes differ")
        if source.shape != target.shape or features.shape != (source.size, 7):
            raise ValueError("physical graph edge shapes differ")
        if source.size and (
            np.min(source) < 0
            or np.min(target) < 0
            or np.max(source) >= count
            or np.max(target) >= count
            or np.any(source == target)
        ):
            raise ValueError("physical graph contains invalid endpoints")
        if len(str(self.feature_bank_sha256)) != 64:
            raise ValueError("graph requires canonical feature-bank lineage")
        metadata = dict(self.metadata or {})
        if metadata.get("artifact_type", GRAPH_ARTIFACT) != GRAPH_ARTIFACT:
            raise ValueError("not a V8 physical maplet graph")
        forbidden = (
            "stores_mapping_rgb",
            "stores_mapping_image_ids",
            "stores_mapping_image_paths",
            "stores_downstream_embeddings",
            "uses_sfm_points",
            "uses_sfm_tracks",
            "uses_alike_descriptors",
            "uses_radio_intermediate",
        )
        for key in forbidden:
            if bool(metadata.get(key, False)):
                raise ValueError(f"physical graph violates contract: {key}")
        metadata.update(
            artifact_type=GRAPH_ARTIFACT,
            artifact_version=GRAPH_VERSION,
            feature_representation="single_canonical_radio_localization_bank",
            stores_mapping_rgb=False,
            stores_mapping_image_ids=False,
            stores_mapping_image_paths=False,
            stores_downstream_embeddings=False,
            uses_sfm_points=False,
            uses_sfm_tracks=False,
            uses_alike_descriptors=False,
            uses_radio_intermediate=False,
        )
        object.__setattr__(self, "maplet_ids", ids)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "tangent_frames", frames)
        object.__setattr__(self, "extents", extents)
        object.__setattr__(self, "edge_source", source)
        object.__setattr__(self, "edge_target", target)
        object.__setattr__(self, "edge_features", features)
        object.__setattr__(self, "metadata", metadata)

    def save_npz(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            maplet_ids=self.maplet_ids,
            centers=self.centers,
            normals=self.normals,
            tangent_frames=self.tangent_frames,
            extents=self.extents,
            edge_source=self.edge_source,
            edge_target=self.edge_target,
            edge_features=self.edge_features,
            feature_bank_sha256=np.asarray(self.feature_bank_sha256),
            metadata_json=np.asarray(json.dumps(dict(self.metadata), sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "PhysicalMapletGraph":
        with np.load(path, allow_pickle=False) as data:
            allowed = {
                "maplet_ids", "centers", "normals", "tangent_frames",
                "extents", "edge_source", "edge_target", "edge_features",
                "feature_bank_sha256", "metadata_json",
            }
            if set(data.files) != allowed:
                raise ValueError("non-canonical V8 physical graph fields")
            return cls(
                maplet_ids=data["maplet_ids"],
                centers=data["centers"],
                normals=data["normals"],
                tangent_frames=data["tangent_frames"],
                extents=data["extents"],
                edge_source=data["edge_source"],
                edge_target=data["edge_target"],
                edge_features=data["edge_features"],
                feature_bank_sha256=str(data["feature_bank_sha256"].item()),
                metadata=json.loads(str(data["metadata_json"].item())),
            )


def build_physical_maplet_graph(
    bank: SurfaceRetrievalMapletBank,
    *,
    feature_bank_path: Path,
    local_neighbors: int = 6,
    medium_neighbors: int = 6,
    long_neighbors: int = 4,
) -> PhysicalMapletGraph:
    """Build local/medium/long-range physical edges without feature copies."""

    centers = np.asarray(bank.centers, dtype=np.float64)
    count = int(centers.shape[0])
    distance = np.linalg.norm(centers[:, None] - centers[None], axis=2)
    np.fill_diagonal(distance, np.inf)
    order = np.argsort(distance, axis=1, kind="stable")
    local_end = min(max(int(local_neighbors), 1), max(count - 1, 0))
    medium_end = min(local_end + max(int(medium_neighbors), 0), max(count - 1, 0))
    long_count = min(max(int(long_neighbors), 0), max(count - 1 - medium_end, 0))
    edges: dict[tuple[int, int], int] = {}
    for source in range(count):
        for target in order[source, :local_end].tolist():
            edges[(source, int(target))] = 0
        for target in order[source, local_end:medium_end].tolist():
            edges[(source, int(target))] = 1
        if long_count:
            # Quantile-spaced long edges carry phase-breaking context without
            # turning the graph into a dense all-pairs structure.
            candidates = order[source, medium_end : count - 1]
            positions = np.linspace(0, candidates.size - 1, long_count, dtype=np.int64)
            for target in candidates[positions].tolist():
                edges[(source, int(target))] = 2
    # Make every selected relation bidirectional and retain the finer class.
    for (source, target), level in list(edges.items()):
        reverse = edges.get((target, source), level)
        edges[(target, source)] = min(int(level), int(reverse))
    pairs = sorted(edges)
    source = np.asarray([x[0] for x in pairs], dtype=np.int32)
    target = np.asarray([x[1] for x in pairs], dtype=np.int32)
    delta = centers[target] - centers[source]
    dist = np.linalg.norm(delta, axis=1)
    direction = delta / np.maximum(dist[:, None], 1e-8)
    normal_cosine = np.sum(
        np.asarray(bank.normals)[source] * np.asarray(bank.normals)[target], axis=1
    )
    area = np.prod(np.maximum(np.asarray(bank.extents)[:, :2], 1e-4), axis=1)
    log_area_ratio = np.log(area[target] / area[source])
    level = np.asarray([edges[x] for x in pairs], dtype=np.float64)
    features = np.c_[direction, dist, normal_cosine, log_area_ratio, level]
    return PhysicalMapletGraph(
        maplet_ids=bank.maplet_ids,
        centers=bank.centers,
        normals=bank.normals,
        tangent_frames=bank.tangent_frames,
        extents=bank.extents,
        edge_source=source,
        edge_target=target,
        edge_features=features,
        feature_bank_sha256=_sha256(feature_bank_path),
        metadata={
            "local_neighbors": int(local_neighbors),
            "medium_neighbors": int(medium_neighbors),
            "long_neighbors": int(long_neighbors),
            "edge_count": int(source.size),
        },
    )


@dataclass(frozen=True)
class QueryRegionGraph:
    xy: np.ndarray
    extent: np.ndarray
    candidate_rows: np.ndarray
    candidate_probabilities: np.ndarray
    null_probability: np.ndarray
    edge_source: np.ndarray
    edge_target: np.ndarray
    # dx,dy,distance,log scale,overlap IoU,scale class
    edge_features: np.ndarray


def _rectangle_iou(xy_a, extent_a, xy_b, extent_b) -> float:
    low = np.maximum(xy_a - extent_a, xy_b - extent_b)
    high = np.minimum(xy_a + extent_a, xy_b + extent_b)
    intersection = float(np.prod(np.maximum(high - low, 0.0)))
    area_a = float(4.0 * np.prod(extent_a))
    area_b = float(4.0 * np.prod(extent_b))
    return intersection / max(area_a + area_b - intersection, 1e-8)


def build_query_region_graph(
    retrieval: MapletRetrievalResult,
    physical: PhysicalMapletGraph,
    *,
    image_size_wh: tuple[int, int],
    candidates_per_region: int = 8,
    local_neighbors: int = 6,
    medium_neighbors: int = 4,
    long_neighbors: int = 2,
) -> QueryRegionGraph:
    """Retain every regional posterior and construct sparse spatial edges."""

    groups = retrieval.groups
    count = len(groups)
    size = np.asarray(image_size_wh, dtype=np.float64)
    xy = np.asarray([g.query_region_xy for g in groups], dtype=np.float64) / size
    extent = np.asarray([g.query_region_extent for g in groups], dtype=np.float64) / size
    width = max(int(candidates_per_region), 1)
    rows = np.full((count, width), -1, dtype=np.int32)
    probabilities = np.zeros((count, width), dtype=np.float32)
    row_by_id = {int(value): row for row, value in enumerate(physical.maplet_ids.tolist())}
    null = np.empty((count,), dtype=np.float32)
    for index, group in enumerate(groups):
        take = min(width, int(group.maplet_ids.size))
        for local, maplet_id in enumerate(group.maplet_ids[:take].tolist()):
            rows[index, local] = int(row_by_id.get(int(maplet_id), -1))
        probabilities[index, :take] = np.asarray(group.probabilities[:take], dtype=np.float32)
        # All probability not represented in candidate_rows is the explicit
        # uninformative/null branch and is conserved exactly.
        null[index] = float(np.clip(1.0 - np.sum(probabilities[index]), 0.0, 1.0))
    pair_distance = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    np.fill_diagonal(pair_distance, np.inf)
    order = np.argsort(pair_distance, axis=1, kind="stable")
    local_end = min(max(int(local_neighbors), 1), max(count - 1, 0))
    medium_end = min(local_end + max(int(medium_neighbors), 0), max(count - 1, 0))
    long_count = min(max(int(long_neighbors), 0), max(count - 1 - medium_end, 0))
    edges: dict[tuple[int, int], int] = {}
    for source in range(count):
        for target in order[source, :local_end].tolist():
            edges[(source, int(target))] = 0
        for target in order[source, local_end:medium_end].tolist():
            edges[(source, int(target))] = 1
        if long_count:
            candidates = order[source, medium_end : count - 1]
            positions = np.linspace(0, candidates.size - 1, long_count, dtype=np.int64)
            for target in candidates[positions].tolist():
                edges[(source, int(target))] = 2
    pairs = sorted(edges)
    source = np.asarray([x[0] for x in pairs], dtype=np.int32)
    target = np.asarray([x[1] for x in pairs], dtype=np.int32)
    delta = xy[target] - xy[source]
    dist = np.linalg.norm(delta, axis=1)
    scale = np.log(
        np.linalg.norm(extent[target], axis=1)
        / np.maximum(np.linalg.norm(extent[source], axis=1), 1e-8)
    )
    overlap = np.asarray([
        _rectangle_iou(xy[a], extent[a], xy[b], extent[b])
        for a, b in pairs
    ])
    level = np.asarray([edges[x] for x in pairs], dtype=np.float64)
    return QueryRegionGraph(
        xy=xy.astype(np.float32),
        extent=extent.astype(np.float32),
        candidate_rows=rows,
        candidate_probabilities=probabilities,
        null_probability=null,
        edge_source=source,
        edge_target=target,
        edge_features=np.c_[delta, dist, scale, overlap, level].astype(np.float32),
    )


def _project_maplet_geometry(
    pose_w2c: np.ndarray,
    physical: PhysicalMapletGraph,
    camera: ColmapCamera,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray(physical.centers, dtype=np.float64)
    pixels, depth = project_world_points(centers, pose_w2c, camera)
    signs = np.asarray([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    corners = (
        centers[:, None]
        + signs[None, :, 0, None] * physical.extents[:, None, 0, None] * physical.tangent_frames[:, None, 0]
        + signs[None, :, 1, None] * physical.extents[:, None, 1, None] * physical.tangent_frames[:, None, 1]
    )
    corner_pixels, corner_depth = project_world_points(corners.reshape(-1, 3), pose_w2c, camera)
    corner_pixels = corner_pixels.reshape(-1, 4, 2)
    corner_depth = corner_depth.reshape(-1, 4)
    low = np.min(corner_pixels, axis=1)
    high = np.max(corner_pixels, axis=1)
    projected_extent = 0.5 * np.maximum(high - low, 1.0)
    rotation = np.asarray(pose_w2c, dtype=np.float64)[:3, :3]
    camera_points = (rotation @ centers.T + np.asarray(pose_w2c)[:3, 3:4]).T
    normal_camera = (rotation @ np.asarray(physical.normals, dtype=np.float64).T).T
    view = -camera_points / np.maximum(np.linalg.norm(camera_points, axis=1, keepdims=True), 1e-8)
    incidence = np.abs(np.sum(normal_camera * view, axis=1))
    finite = (
        np.isfinite(pixels).all(axis=1)
        & (depth > 0.1)
        & np.isfinite(corner_pixels).all(axis=(1, 2))
        & np.all(corner_depth > 0.1, axis=1)
        & (high[:, 0] >= 0.0) & (low[:, 0] < camera.width)
        & (high[:, 1] >= 0.0) & (low[:, 1] < camera.height)
    )
    return pixels, projected_extent, incidence, finite


def score_structured_maplet_pose(
    pose_w2c: np.ndarray,
    query: QueryRegionGraph,
    physical: PhysicalMapletGraph,
    camera: ColmapCamera,
    *,
    pair_weight: float = 0.50,
) -> tuple[float, dict[str, float]]:
    """Marginal unary + multi-scale relation likelihood for one SE(3) pose.

    This is region-to-surface alignment, not point correspondence/PnP.  Each
    candidate identity is marginalized and each query instance remains a
    separate latent variable.  Long-range edges receive the largest weight
    because they are the relations able to break facade periodicity.
    """

    pixels, projected_extent, incidence, visible = _project_maplet_geometry(
        pose_w2c, physical, camera
    )
    physical_relation_level = np.full(
        (physical.maplet_ids.size, physical.maplet_ids.size),
        3,
        dtype=np.int8,
    )
    np.fill_diagonal(physical_relation_level, 0)
    physical_relation_level[
        physical.edge_source, physical.edge_target
    ] = np.minimum(
        physical_relation_level[physical.edge_source, physical.edge_target],
        np.rint(physical.edge_features[:, 6]).astype(np.int8),
    )
    image_size = np.asarray([camera.width, camera.height], dtype=np.float64)
    projected_xy = pixels / image_size[None]
    projected_scale = projected_extent / image_size[None]
    candidate_rows = query.candidate_rows
    valid = candidate_rows >= 0
    safe_rows = np.maximum(candidate_rows, 0)
    candidate_visible = valid & visible[safe_rows]
    sigma = np.maximum(query.extent[:, None, :] * 0.75, 0.018)
    residual = (projected_xy[safe_rows] - query.xy[:, None]) / sigma
    scale_residual = np.log(
        np.maximum(np.linalg.norm(projected_scale[safe_rows], axis=2), 1e-5)
        / np.maximum(np.linalg.norm(query.extent, axis=1)[:, None], 1e-5)
    )
    # Density ratios are normalized against a uniform image baseline; unlike
    # a bare exp(-error), a correct match can provide positive evidence over
    # the explicit null branch.
    unary_ratio = (
        1.0 / np.maximum(2.0 * np.pi * sigma[:, :, 0] * sigma[:, :, 1], 1e-5)
        * np.exp(np.maximum(-0.5 * np.sum(residual * residual, axis=2), -60.0))
        * np.exp(np.maximum(-0.5 * (scale_residual / 1.0) ** 2, -20.0))
        * (0.25 + 0.75 * incidence[safe_rows])
    )
    unary_ratio = np.where(candidate_visible, unary_ratio, 0.0)
    unary_evidence = query.null_probability + np.sum(
        query.candidate_probabilities * unary_ratio, axis=1
    )
    unary_log = np.log(np.maximum(unary_evidence, 1e-12))

    edge_logs = []
    edge_weights = []
    for edge, (source, target) in enumerate(
        zip(query.edge_source.tolist(), query.edge_target.tolist())
    ):
        rows_a = candidate_rows[source]
        rows_b = candidate_rows[target]
        valid_pair = (rows_a[:, None] >= 0) & (rows_b[None] >= 0)
        safe_a = np.maximum(rows_a, 0)
        safe_b = np.maximum(rows_b, 0)
        valid_pair &= visible[safe_a, None] & visible[safe_b[None]]
        predicted_delta = (
            projected_xy[safe_b][None] - projected_xy[safe_a][:, None]
        )
        observed_delta = query.edge_features[edge, :2]
        level = int(round(float(query.edge_features[edge, 5])))
        relation_sigma = (0.055, 0.085, 0.14)[min(max(level, 0), 2)]
        relation_residual = np.sum(
            ((predicted_delta - observed_delta[None, None]) / relation_sigma) ** 2,
            axis=2,
        )
        observed_distance = max(float(query.edge_features[edge, 2]), 1e-4)
        predicted_distance = np.linalg.norm(predicted_delta, axis=2)
        distance_residual = np.log(
            np.maximum(predicted_distance, 1e-4) / observed_distance
        )
        ratio = np.exp(
            np.maximum(
                -0.5 * relation_residual - 0.5 * (distance_residual / 0.55) ** 2,
                -60.0,
            )
        ) / max(2.0 * np.pi * relation_sigma * relation_sigma, 1e-5)
        map_level = physical_relation_level[safe_a[:, None], safe_b[None]]
        # Explicitly match query edge scale to the sparse physical map graph.
        # Missing physical edges remain possible (the graph is intentionally
        # sparse), but cannot outvote a geometrically compatible stored
        # local/medium/long-range relation solely through projection chance.
        level_compatibility = np.asarray(
            (
                (1.00, 0.80, 0.45, 0.20),
                (0.85, 1.00, 0.80, 0.30),
                (0.65, 0.85, 1.00, 0.45),
            )[min(max(level, 0), 2)],
            dtype=np.float64,
        )
        ratio *= level_compatibility[map_level]
        # Overlapping query supports may legitimately select the same physical
        # maplet.  Non-overlapping instance nodes may not collapse to one ID.
        if float(query.edge_features[edge, 4]) < 0.10:
            # Same-maplet cross-group geometry is unknown until a maplet-local
            # child coordinate exists; it is neutral rather than impossible.
            ratio = np.where(safe_a[:, None] == safe_b[None], 1.0, ratio)
        ratio = np.where(valid_pair, ratio, 0.0)
        pair_probability = (
            query.candidate_probabilities[source, :, None]
            * query.candidate_probabilities[target, None, :]
        )
        unmatched = 1.0 - float(np.sum(pair_probability))
        evidence = max(unmatched + float(np.sum(pair_probability * ratio)), 1e-12)
        edge_logs.append(float(np.log(evidence)))
        # Downweight redundant overlapping local regions and emphasize
        # medium/long relations that break repeated-window phase.
        overlap = float(query.edge_features[edge, 4])
        edge_weights.append((0.35, 0.70, 1.0)[min(max(level, 0), 2)] * (1.0 - 0.65 * overlap))
    if edge_logs:
        graph_log = float(np.average(edge_logs, weights=edge_weights))
    else:
        graph_log = 0.0
    score = float(np.mean(unary_log) + float(pair_weight) * graph_log)
    return score, {
        "unary_log_likelihood": float(np.mean(unary_log)),
        "graph_log_likelihood": graph_log,
        "supported_region_fraction": float(np.mean(unary_evidence > 1.0)),
        "visible_candidate_fraction": float(np.mean(candidate_visible)),
    }


def refine_structured_maplet_pose(
    initial_pose_w2c: np.ndarray,
    query: QueryRegionGraph,
    physical: PhysicalMapletGraph,
    camera: ColmapCamera,
    *,
    pair_weight: float = 0.50,
    stages: int = 4,
    initial_translation_step_m: float = 0.75,
    initial_rotation_step_deg: float = 8.0,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Deterministic coarse-to-fine SE(3) coordinate search on graph score."""

    pose = np.asarray(initial_pose_w2c, dtype=np.float64).copy()
    score, parts = score_structured_maplet_pose(
        pose, query, physical, camera, pair_weight=float(pair_weight)
    )
    for stage in range(max(int(stages), 0)):
        rotation_step = np.deg2rad(float(initial_rotation_step_deg)) / (2.0 ** stage)
        translation_step = float(initial_translation_step_m) / (2.0 ** stage)
        step = np.asarray([rotation_step] * 3 + [translation_step] * 3)
        winner_pose = pose
        winner_score = score
        winner_parts = parts
        for axis in range(6):
            for sign in (-1.0, 1.0):
                delta = np.zeros((6,), dtype=np.float64)
                delta[axis] = sign * step[axis]
                candidate = se3_exp(delta) @ pose
                candidate_score, candidate_parts = score_structured_maplet_pose(
                    candidate,
                    query,
                    physical,
                    camera,
                    pair_weight=float(pair_weight),
                )
                if candidate_score > winner_score:
                    winner_pose = candidate
                    winner_score = candidate_score
                    winner_parts = candidate_parts
        pose, score, parts = winner_pose, winner_score, winner_parts
    return pose, float(score), parts
