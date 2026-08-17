"""Pose-free handoff from RADIO child evidence to mapping visibility poses.

The atlas is deliberately a *proposal* mechanism.  A mapping pose stores no
RGB or image descriptor; it stores only a sparse physical-child visibility
distribution, both globally and on a fixed coarse image grid.  At query time
the same distributions are formed from the existing token-to-child evidence
and compared with a Bhattacharyya affinity.  No point correspondence, depth,
PnP, query pose, or query ground truth is part of this module.

The returned poses are chart centres, not final localization estimates.  They
are intended to answer the first retrieval-to-pose question: does the retrieved
surface set acquire a correct, geographically distinct SE(3) basin?
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from .lineage import arrays_sha256, canonical_json_sha256
from .physical_map import GoalMapletPhysicalMap
from .pose_proposal import _rotation_distance_degrees
from .pure_retrieval import PureRadioPhysicalRetrieval, all_radio_token_coordinates


SCHEMA = "goal_maplet_child_visibility_pose_atlas_v3"
SCORE_SEMANTICS = "global_and_joint_grid_child_bhattacharyya_affinity_v3_semantic_hash"


_LOCATION_NEIGHBOR_CACHE: dict[tuple[str, float], tuple[np.ndarray, ...]] = {}


@dataclass(frozen=True)
class NestedWideNearPoseBasins:
    """Two proposal queues merged as continuous location basins.

    ``wide`` protects geographic acquisition while ``near`` protects already
    close view samples.  Budgets are applied independently, so a wide row can
    never consume the near-view quota.  Exact duplicates are represented once
    with origin ``wide+near`` rather than silently deleting near evidence.
    """

    pose_rows: np.ndarray
    location_ids: np.ndarray
    proposal_origins: tuple[str, ...]
    translation_search_radius_m: np.ndarray
    rotation_search_radius_deg: np.ndarray


def nested_wide_near_pose_basins(
    poses_w2c: np.ndarray,
    wide_global_scores: np.ndarray,
    wide_layout_scores: np.ndarray,
    near_view_scores: np.ndarray,
    *,
    wide_budget: int = 48,
    near_budget: int = 16,
    location_radius_m: float = 2.0,
    near_translation_nms_m: float = 0.5,
    near_rotation_nms_deg: float = 5.0,
    continuous_translation_radius_m: float = 2.0,
    continuous_rotation_radius_deg: float = 45.0,
) -> NestedWideNearPoseBasins:
    """Build nested wide-location and near-view queues without score mixing."""

    pose = _validate_pose_batch(poses_w2c)
    count = int(pose.shape[0])
    near = np.asarray(near_view_scores, dtype=np.float64).reshape(-1)
    if (
        near.shape != (count,) or np.any(~np.isfinite(near))
        or int(wide_budget) <= 0 or int(near_budget) <= 0
        or float(location_radius_m) <= 0.0
        or float(continuous_translation_radius_m) <= 0.0
        or float(continuous_rotation_radius_deg) <= 0.0
    ):
        raise ValueError("invalid nested wide/near proposal inputs")
    wide_rows = hierarchical_location_orientation_pose_rows(
        pose, wide_global_scores, wide_layout_scores,
        maximum_modes=int(wide_budget), orientations_per_location=1,
        location_radius_m=float(location_radius_m),
        translation_nms_m=float(near_translation_nms_m),
        rotation_nms_deg=float(near_rotation_nms_deg),
    )
    near_rows = diverse_pose_rows(
        pose, near, maximum_modes=int(near_budget),
        translation_nms_m=float(near_translation_nms_m),
        rotation_nms_deg=float(near_rotation_nms_deg),
    )
    selected = [int(row) for row in wide_rows]
    origins = ["wide" for _ in selected]
    centers = -np.swapaxes(pose[:, :3, :3], 1, 2) @ pose[:, :3, 3, None]
    centers = centers[..., 0]
    for row_value in near_rows:
        row = int(row_value)
        duplicate = next((
            index for index, prior in enumerate(selected)
            if np.linalg.norm(centers[row] - centers[prior]) <= float(near_translation_nms_m)
            and _rotation_distance_degrees(pose[row], pose[prior]) <= float(near_rotation_nms_deg)
        ), None)
        if duplicate is None:
            selected.append(row)
            origins.append("near")
        elif origins[duplicate] == "wide":
            origins[duplicate] = "wide+near"
    location_centers: list[int] = []
    location_ids: list[int] = []
    for row in selected:
        distance = np.asarray([
            np.linalg.norm(centers[row] - centers[seed]) for seed in location_centers
        ])
        if not location_centers or float(np.min(distance)) > float(location_radius_m):
            location_centers.append(row)
            location_ids.append(len(location_centers) - 1)
        else:
            location_ids.append(int(np.argmin(distance)))
    rows = np.asarray(selected, dtype=np.int64)
    return NestedWideNearPoseBasins(
        pose_rows=rows,
        location_ids=np.asarray(location_ids, dtype=np.int64),
        proposal_origins=tuple(origins),
        translation_search_radius_m=np.full(rows.shape, float(continuous_translation_radius_m)),
        rotation_search_radius_deg=np.full(rows.shape, float(continuous_rotation_radius_deg)),
    )


def _cached_location_neighborhoods(
    centers: np.ndarray, radius_m: float,
) -> tuple[np.ndarray, ...]:
    value = np.ascontiguousarray(centers, dtype=np.float64)
    digest = hashlib.sha256(value.view(np.uint8)).hexdigest()
    key = (digest, float(radius_m))
    cached = _LOCATION_NEIGHBOR_CACHE.get(key)
    if cached is not None:
        return cached
    tree = cKDTree(value)
    rows = tuple(
        np.asarray(sorted(neighbor), dtype=np.int64)
        for neighbor in tree.query_ball_point(value, r=float(radius_m))
    )
    _LOCATION_NEIGHBOR_CACHE[key] = rows
    return rows


def _validate_pose_batch(poses_w2c: np.ndarray) -> np.ndarray:
    pose = np.asarray(poses_w2c, dtype=np.float64)
    if pose.ndim != 3 or pose.shape[1:] != (4, 4) or np.any(~np.isfinite(pose)):
        raise ValueError("visibility-atlas poses must have shape [view,4,4]")
    if np.any(np.abs(pose[:, 3] - np.asarray([0.0, 0.0, 0.0, 1.0])) > 1e-8):
        raise ValueError("visibility-atlas poses must be homogeneous transforms")
    rotation = pose[:, :3, :3]
    error = np.linalg.norm(
        rotation @ np.swapaxes(rotation, 1, 2) - np.eye(3), axis=(1, 2)
    )
    if np.any(error > 1e-6) or np.any(np.linalg.det(rotation) <= 0.0):
        raise ValueError("visibility-atlas poses must contain proper rotations")
    return pose


def _validate_csr(
    offsets: np.ndarray,
    rows: np.ndarray,
    weights: np.ndarray,
    *,
    view_count: int,
    column_count: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indptr = np.asarray(offsets, dtype=np.int64).reshape(-1)
    index = np.asarray(rows, dtype=np.int32).reshape(-1)
    value = np.asarray(weights, dtype=np.float32).reshape(-1)
    if (
        indptr.shape != (int(view_count) + 1,)
        or indptr[0] != 0
        or indptr[-1] != index.size
        or np.any(np.diff(indptr) < 0)
        or index.shape != value.shape
        or np.any(index < 0)
        or np.any(index >= int(column_count))
        or np.any(~np.isfinite(value))
        or np.any(value <= 0.0)
    ):
        raise ValueError(f"invalid {name} visibility CSR")
    for view in range(int(view_count)):
        start, end = int(indptr[view]), int(indptr[view + 1])
        if np.unique(index[start:end]).size != end - start:
            raise ValueError(f"duplicate {name} visibility columns")
    return indptr, index, value


@dataclass(frozen=True)
class ChildVisibilityPoseAtlas:
    poses_w2c: np.ndarray
    global_offsets: np.ndarray
    global_child_rows: np.ndarray
    global_weights: np.ndarray
    layout_offsets: np.ndarray
    layout_keys: np.ndarray
    layout_weights: np.ndarray
    child_count: int
    grid_rows: int
    grid_cols: int
    physical_map_sha256: str
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        pose = _validate_pose_batch(self.poses_w2c)
        child_count = int(self.child_count)
        grid_rows, grid_cols = int(self.grid_rows), int(self.grid_cols)
        if child_count <= 0 or grid_rows <= 0 or grid_cols <= 0:
            raise ValueError("invalid child/grid dimensions")
        global_csr = _validate_csr(
            self.global_offsets,
            self.global_child_rows,
            self.global_weights,
            view_count=pose.shape[0],
            column_count=child_count,
            name="global-child",
        )
        layout_csr = _validate_csr(
            self.layout_offsets,
            self.layout_keys,
            self.layout_weights,
            view_count=pose.shape[0],
            column_count=grid_rows * grid_cols * child_count,
            name="layout-child",
        )
        metadata = dict(self.metadata or {})
        if metadata.get("artifact_type", SCHEMA) != SCHEMA:
            raise ValueError("not a child visibility pose atlas")
        if any(bool(metadata.get(key, False)) for key in (
            "stores_mapping_rgb", "stores_mapping_image_descriptors",
            "uses_query_pose", "uses_query_ground_truth", "uses_pnp",
            "uses_alike", "uses_sfm_points", "uses_sfm_tracks",
        )):
            raise ValueError("visibility atlas violates the pose-free handoff contract")
        object.__setattr__(self, "poses_w2c", pose)
        object.__setattr__(self, "global_offsets", global_csr[0])
        object.__setattr__(self, "global_child_rows", global_csr[1])
        object.__setattr__(self, "global_weights", global_csr[2])
        object.__setattr__(self, "layout_offsets", layout_csr[0])
        object.__setattr__(self, "layout_keys", layout_csr[1])
        object.__setattr__(self, "layout_weights", layout_csr[2])
        object.__setattr__(self, "metadata", metadata)

    @property
    def view_count(self) -> int:
        return int(self.poses_w2c.shape[0])

    @property
    def content_sha256(self) -> str:
        array_sha256 = arrays_sha256({
            "poses_w2c": self.poses_w2c,
            "global_offsets": self.global_offsets,
            "global_child_rows": self.global_child_rows,
            "global_weights": self.global_weights,
            "layout_offsets": self.layout_offsets,
            "layout_keys": self.layout_keys,
            "layout_weights": self.layout_weights,
        })
        return canonical_json_sha256({
            "artifact_type": SCHEMA,
            "score_semantics": SCORE_SEMANTICS,
            "array_sha256": array_sha256,
            "child_count": int(self.child_count),
            "grid_rows": int(self.grid_rows),
            "grid_cols": int(self.grid_cols),
            "physical_map_sha256": str(self.physical_map_sha256),
        })

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
            global_offsets=self.global_offsets,
            global_child_rows=self.global_child_rows,
            global_weights=self.global_weights,
            layout_offsets=self.layout_offsets,
            layout_keys=self.layout_keys,
            layout_weights=self.layout_weights,
            child_count=np.asarray(self.child_count, dtype=np.int64),
            grid_rows=np.asarray(self.grid_rows, dtype=np.int64),
            grid_cols=np.asarray(self.grid_cols, dtype=np.int64),
            physical_map_sha256=np.asarray(self.physical_map_sha256),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "ChildVisibilityPoseAtlas":
        with np.load(Path(path), allow_pickle=False) as data:
            value = cls(
                poses_w2c=data["poses_w2c"],
                global_offsets=data["global_offsets"],
                global_child_rows=data["global_child_rows"],
                global_weights=data["global_weights"],
                layout_offsets=data["layout_offsets"],
                layout_keys=data["layout_keys"],
                layout_weights=data["layout_weights"],
                child_count=int(data["child_count"].item()),
                grid_rows=int(data["grid_rows"].item()),
                grid_cols=int(data["grid_cols"].item()),
                physical_map_sha256=str(data["physical_map_sha256"].item()),
                metadata=json.loads(str(data["metadata_json"].item())),
            )
        if value.metadata.get("score_semantics") != SCORE_SEMANTICS:
            raise ValueError("visibility atlas score semantics differ")
        if value.metadata.get("content_sha256") != value.content_sha256:
            raise ValueError("visibility atlas content hash mismatch")
        return value

    def sparse_matrices(self) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        global_matrix = sparse.csr_matrix(
            (self.global_weights, self.global_child_rows, self.global_offsets),
            shape=(self.view_count, self.child_count),
            dtype=np.float32,
        )
        layout_matrix = sparse.csr_matrix(
            (self.layout_weights, self.layout_keys, self.layout_offsets),
            shape=(self.view_count, self.grid_rows * self.grid_cols * self.child_count),
            dtype=np.float32,
        )
        return global_matrix, layout_matrix


def _primitive_to_child_owner(physical: GoalMapletPhysicalMap) -> np.ndarray:
    owner = np.full((physical.primitive_ids.size,), -1, dtype=np.int32)
    for child in range(physical.child_parent_rows.size):
        start = int(physical.child_member_offsets[child])
        end = int(physical.child_member_offsets[child + 1])
        rows = np.asarray(physical.child_member_primitive_rows[start:end], dtype=np.int64)
        if np.any((rows < 0) | (rows >= owner.size)) or np.any(owner[rows] >= 0):
            raise ValueError("physical children must be a disjoint primitive partition")
        owner[rows] = int(child)
    if np.any(owner < 0):
        raise ValueError("physical child partition does not cover every primitive")
    return owner


def _top_normalized_sqrt(
    mass: np.ndarray,
    *,
    maximum_entries: int,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(mass, dtype=np.float64).reshape(-1)
    rows = np.flatnonzero(value > 0.0)
    if rows.size > int(maximum_entries):
        order = np.lexsort((rows, -value[rows]))[: int(maximum_entries)]
        rows = rows[order]
    rows = np.sort(rows)
    total = float(np.sum(value[rows]))
    if total <= 0.0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    weight = np.sqrt(value[rows] / total) * float(scale)
    return rows.astype(np.int32), weight.astype(np.float32)


def _joint_layout_normalized_sqrt(
    layout_mass: np.ndarray,
    *,
    maximum_children_per_cell: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Truncate per cell, then normalize the retained joint ``(cell,child)`` mass.

    Normalizing every cell separately makes a nearly empty cell carry the same
    norm as a reliable one.  The joint distribution preserves that reliability
    while keeping the same deterministic per-cell sparsity budget.
    """

    mass = np.asarray(layout_mass, dtype=np.float64)
    if mass.ndim != 2 or np.any(~np.isfinite(mass)) or np.any(mass < 0.0):
        raise ValueError("layout mass must be a finite nonnegative matrix")
    child_count = int(mass.shape[1])
    keys_out: list[np.ndarray] = []
    mass_out: list[np.ndarray] = []
    for cell_row in range(int(mass.shape[0])):
        value = mass[cell_row]
        rows = np.flatnonzero(value > 0.0)
        if rows.size > int(maximum_children_per_cell):
            order = np.lexsort((rows, -value[rows]))[
                : int(maximum_children_per_cell)
            ]
            rows = rows[order]
        rows = np.sort(rows)
        keys_out.append((cell_row * child_count + rows).astype(np.int32))
        mass_out.append(value[rows])
    keys = np.concatenate(keys_out)
    retained_mass = np.concatenate(mass_out)
    total = float(np.sum(retained_mass))
    if total <= 0.0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    return keys, np.sqrt(retained_mass / total).astype(np.float32)


def build_child_visibility_pose_atlas(
    physical: GoalMapletPhysicalMap,
    contributor_paths: Sequence[Path],
    *,
    grid_rows: int = 4,
    grid_cols: int = 4,
    maximum_global_children: int = 1024,
    maximum_children_per_cell: int = 256,
    metadata: Mapping[str, object] | None = None,
) -> ChildVisibilityPoseAtlas:
    """Build a feature-free child/coarse-layout atlas from frozen contributors."""

    paths = tuple(Path(path) for path in contributor_paths)
    if not paths or int(grid_rows) <= 0 or int(grid_cols) <= 0:
        raise ValueError("visibility atlas needs contributors and a positive grid")
    if int(maximum_global_children) <= 0 or int(maximum_children_per_cell) <= 0:
        raise ValueError("visibility atlas truncation budgets must be positive")
    primitive_ids = np.asarray(physical.primitive_ids, dtype=np.int64)
    if np.unique(primitive_ids).size != primitive_ids.size or np.any(primitive_ids < 0):
        raise ValueError("physical primitive IDs must be unique and non-negative")
    row_by_id = np.full((int(np.max(primitive_ids)) + 1,), -1, dtype=np.int32)
    row_by_id[primitive_ids] = np.arange(primitive_ids.size, dtype=np.int32)
    owner = _primitive_to_child_owner(physical)
    child_count = int(physical.child_parent_rows.size)
    cell_count = int(grid_rows) * int(grid_cols)
    poses: list[np.ndarray] = []
    global_rows_out: list[np.ndarray] = []
    global_weights_out: list[np.ndarray] = []
    layout_keys_out: list[np.ndarray] = []
    layout_weights_out: list[np.ndarray] = []
    global_offsets = [0]
    layout_offsets = [0]
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            ids = np.asarray(data["topk_ids"], dtype=np.int64)
            weights = np.asarray(data["topk_weights"], dtype=np.float64)
            pose = np.asarray(data["pose_w2c"], dtype=np.float64)
        if ids.ndim != 3 or weights.shape != ids.shape or np.any(~np.isfinite(weights)):
            raise ValueError(f"invalid contributor cache {path}")
        height, width, _ = ids.shape
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        cell = (
            np.minimum(yy * int(grid_rows) // height, int(grid_rows) - 1)
            * int(grid_cols)
            + np.minimum(xx * int(grid_cols) // width, int(grid_cols) - 1)
        )
        cell = np.broadcast_to(cell[..., None], ids.shape)
        valid = (
            (ids >= 0) & (ids < row_by_id.size) & (weights > 0.0)
        )
        safe_id = np.clip(ids, 0, row_by_id.size - 1)
        primitive_row = np.where(valid, row_by_id[safe_id], -1)
        valid &= primitive_row >= 0
        child = np.where(valid, owner[np.maximum(primitive_row, 0)], -1)
        valid &= child >= 0
        layout_mass = np.bincount(
            (cell[valid] * child_count + child[valid]).astype(np.int64),
            weights=weights[valid],
            minlength=cell_count * child_count,
        ).reshape(cell_count, child_count)
        global_mass = np.sum(layout_mass, axis=0)
        global_rows, global_weight = _top_normalized_sqrt(
            global_mass, maximum_entries=int(maximum_global_children)
        )
        keys, values = _joint_layout_normalized_sqrt(
            layout_mass,
            maximum_children_per_cell=int(maximum_children_per_cell),
        )
        poses.append(pose)
        global_rows_out.append(global_rows)
        global_weights_out.append(global_weight)
        layout_keys_out.append(keys)
        layout_weights_out.append(values)
        global_offsets.append(global_offsets[-1] + global_rows.size)
        layout_offsets.append(layout_offsets[-1] + keys.size)
    return ChildVisibilityPoseAtlas(
        poses_w2c=np.asarray(poses, dtype=np.float64),
        global_offsets=np.asarray(global_offsets, dtype=np.int64),
        global_child_rows=np.concatenate(global_rows_out),
        global_weights=np.concatenate(global_weights_out),
        layout_offsets=np.asarray(layout_offsets, dtype=np.int64),
        layout_keys=np.concatenate(layout_keys_out),
        layout_weights=np.concatenate(layout_weights_out),
        child_count=child_count,
        grid_rows=int(grid_rows),
        grid_cols=int(grid_cols),
        physical_map_sha256=physical.content_sha256,
        metadata={
            "artifact_type": SCHEMA,
            "score_semantics": SCORE_SEMANTICS,
            "stores_mapping_rgb": False,
            "stores_mapping_image_descriptors": False,
            "uses_query_pose": False,
            "uses_query_ground_truth": False,
            "uses_pnp": False,
            "uses_alike": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "mapping_pose_role": "visibility_chart_sample_not_final_pose",
            "grid_semantics": "fixed_blocks_joint_cell_child_probability",
            "source_contributor_count": len(paths),
            **dict(metadata or {}),
        },
    )


def query_child_affinity_vectors(
    retrieval: PureRadioPhysicalRetrieval,
    atlas: ChildVisibilityPoseAtlas,
    *,
    selected_children_only: bool = True,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Return global/layout query vectors under the atlas affinity contract."""

    if retrieval.physical_map_sha256 != atlas.physical_map_sha256:
        raise ValueError("retrieval and visibility atlas use different physical maps")
    xy = np.asarray(retrieval.token_xy, dtype=np.int64)
    expected = all_radio_token_coordinates(36, 64).astype(np.int64)
    if xy.shape != expected.shape or not np.array_equal(xy, expected):
        raise ValueError("visibility handoff requires the complete 36x64 RADIO grid")
    rows = np.asarray(retrieval.token_child_rows, dtype=np.int64)
    probability = np.asarray(retrieval.token_child_probabilities, dtype=np.float64)
    valid = (
        (rows >= 0) & (rows < atlas.child_count) & (probability > 0.0)
    )
    if selected_children_only:
        selected = np.zeros((atlas.child_count,), dtype=bool)
        selected[np.asarray(retrieval.scene_child_rows, dtype=np.int64)] = True
        safe_rows = np.clip(rows, 0, atlas.child_count - 1)
        valid &= selected[safe_rows]
    y_cell = np.minimum(xy[:, 1] * atlas.grid_rows // 36, atlas.grid_rows - 1)
    x_cell = np.minimum(xy[:, 0] * atlas.grid_cols // 64, atlas.grid_cols - 1)
    cell = np.broadcast_to((y_cell * atlas.grid_cols + x_cell)[:, None], rows.shape)
    layout_mass = np.bincount(
        (cell[valid] * atlas.child_count + rows[valid]).astype(np.int64),
        weights=probability[valid],
        minlength=atlas.grid_rows * atlas.grid_cols * atlas.child_count,
    ).reshape(atlas.grid_rows * atlas.grid_cols, atlas.child_count)
    global_mass = np.sum(layout_mass, axis=0)
    global_rows, global_weight = _top_normalized_sqrt(
        global_mass, maximum_entries=atlas.child_count
    )
    cell_count = atlas.grid_rows * atlas.grid_cols
    keys, values = _joint_layout_normalized_sqrt(
        layout_mass,
        maximum_children_per_cell=atlas.child_count,
    )
    return (
        sparse.csr_matrix(
            (global_weight, global_rows, np.asarray([0, global_rows.size])),
            shape=(1, atlas.child_count),
        ),
        sparse.csr_matrix(
            (values, keys, np.asarray([0, keys.size])),
            shape=(1, cell_count * atlas.child_count),
        ),
    )


def score_visibility_pose_atlas(
    atlas: ChildVisibilityPoseAtlas,
    retrieval: PureRadioPhysicalRetrieval,
    *,
    layout_weight: float = 0.5,
    layout_tolerance_cells: int = 0,
    selected_children_only: bool = True,
    matrices: tuple[sparse.csr_matrix, sparse.csr_matrix] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score every chart centre without claiming a calibrated pose posterior."""

    alpha = float(layout_weight)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("layout_weight must lie in [0,1]")
    if int(layout_tolerance_cells) < 0:
        raise ValueError("layout_tolerance_cells must be nonnegative")
    global_query, layout_query = query_child_affinity_vectors(
        retrieval, atlas, selected_children_only=selected_children_only
    )
    global_atlas, layout_atlas = atlas.sparse_matrices() if matrices is None else matrices
    global_product = global_atlas @ global_query.T
    global_score = np.asarray(global_product.toarray()).reshape(-1)
    radius = int(layout_tolerance_cells)
    layout_scores: list[np.ndarray] = []
    base_keys = layout_query.indices.astype(np.int64)
    base_values = layout_query.data.astype(np.float32)
    base_cell = base_keys // atlas.child_count
    base_child = base_keys % atlas.child_count
    base_y = base_cell // atlas.grid_cols
    base_x = base_cell % atlas.grid_cols
    for delta_y in range(-radius, radius + 1):
        for delta_x in range(-radius, radius + 1):
            shifted_y = base_y + delta_y
            shifted_x = base_x + delta_x
            valid = (
                (shifted_y >= 0) & (shifted_y < atlas.grid_rows)
                & (shifted_x >= 0) & (shifted_x < atlas.grid_cols)
            )
            shifted_keys = (
                (shifted_y[valid] * atlas.grid_cols + shifted_x[valid])
                * atlas.child_count + base_child[valid]
            )
            shifted = sparse.csr_matrix(
                (
                    base_values[valid],
                    shifted_keys.astype(np.int32),
                    np.asarray([0, int(np.sum(valid))], dtype=np.int64),
                ),
                shape=(1, atlas.grid_rows * atlas.grid_cols * atlas.child_count),
            )
            layout_scores.append(
                np.asarray((layout_atlas @ shifted.T).toarray()).reshape(-1)
            )
    layout_score = np.max(np.stack(layout_scores, axis=0), axis=0)
    score = (1.0 - alpha) * global_score + alpha * layout_score
    return score.astype(np.float32), global_score.astype(np.float32), layout_score.astype(np.float32)


def diverse_pose_rows(
    poses_w2c: np.ndarray,
    scores: np.ndarray,
    *,
    maximum_modes: int = 32,
    translation_nms_m: float = 0.5,
    rotation_nms_deg: float = 5.0,
) -> np.ndarray:
    """Greedy stable SE(3) NMS over chart centres."""

    pose = _validate_pose_batch(poses_w2c)
    value = np.asarray(scores, dtype=np.float64).reshape(-1)
    if value.shape != (pose.shape[0],) or np.any(~np.isfinite(value)):
        raise ValueError("pose scores differ from atlas views")
    order = np.lexsort((np.arange(value.size, dtype=np.int64), -value))
    retained: list[int] = []
    centers = -np.swapaxes(pose[:, :3, :3], 1, 2) @ pose[:, :3, 3, None]
    centers = centers[..., 0]
    for row in order.tolist():
        duplicate = any(
            np.linalg.norm(centers[row] - centers[prior]) <= float(translation_nms_m)
            and _rotation_distance_degrees(pose[row], pose[prior]) <= float(rotation_nms_deg)
            for prior in retained
        )
        if not duplicate:
            retained.append(int(row))
            if len(retained) >= int(maximum_modes):
                break
    return np.asarray(retained, dtype=np.int64)


def diverse_dual_queue_pose_rows(
    poses_w2c: np.ndarray,
    global_scores: np.ndarray,
    layout_scores: np.ndarray,
    *,
    maximum_modes: int = 32,
    global_to_layout_ratio: int = 3,
    translation_nms_m: float = 0.5,
    rotation_nms_deg: float = 5.0,
) -> np.ndarray:
    """Interleave global/layout queues without averaging away either mode.

    Three global proposals followed by one layout proposal is the default.
    Duplicate or physically equivalent rows are skipped, and each queue keeps
    advancing until the shared basin budget is filled.
    """

    pose = _validate_pose_batch(poses_w2c)
    global_value = np.asarray(global_scores, dtype=np.float64).reshape(-1)
    layout_value = np.asarray(layout_scores, dtype=np.float64).reshape(-1)
    if (
        global_value.shape != (pose.shape[0],)
        or layout_value.shape != global_value.shape
        or np.any(~np.isfinite(global_value))
        or np.any(~np.isfinite(layout_value))
        or int(maximum_modes) <= 0
        or int(global_to_layout_ratio) <= 0
    ):
        raise ValueError("invalid dual-queue pose proposal inputs")
    stable = np.arange(pose.shape[0], dtype=np.int64)
    orders = (
        np.lexsort((stable, -global_value)),
        np.lexsort((stable, -layout_value)),
    )
    cursors = [0, 0]
    schedule = [0] * int(global_to_layout_ratio) + [1]
    retained: list[int] = []
    retained_set: set[int] = set()
    centers = -np.swapaxes(pose[:, :3, :3], 1, 2) @ pose[:, :3, 3, None]
    centers = centers[..., 0]
    schedule_cursor = 0
    while len(retained) < int(maximum_modes) and (
        cursors[0] < pose.shape[0] or cursors[1] < pose.shape[0]
    ):
        queue = schedule[schedule_cursor % len(schedule)]
        schedule_cursor += 1
        if cursors[queue] >= pose.shape[0]:
            queue = 1 - queue
            if cursors[queue] >= pose.shape[0]:
                break
        row = int(orders[queue][cursors[queue]])
        cursors[queue] += 1
        if row in retained_set:
            continue
        duplicate = any(
            np.linalg.norm(centers[row] - centers[prior]) <= float(translation_nms_m)
            and _rotation_distance_degrees(pose[row], pose[prior]) <= float(rotation_nms_deg)
            for prior in retained
        )
        if not duplicate:
            retained.append(row)
            retained_set.add(row)
    return np.asarray(retained, dtype=np.int64)


def hierarchical_location_orientation_pose_rows(
    poses_w2c: np.ndarray,
    global_scores: np.ndarray,
    layout_scores: np.ndarray,
    *,
    maximum_modes: int = 64,
    orientations_per_location: int = 2,
    location_radius_m: float = 2.0,
    orientation_nms_degrees: float = 10.0,
    translation_nms_m: float = 0.5,
    rotation_nms_deg: float = 5.0,
) -> np.ndarray:
    """Select geography with global evidence, then orientation with layout.

    This is deliberately staged rather than a linear global/layout sum.  The
    first pass chooses spatially distinct location seeds; the second ranks
    orientations only inside each fixed-radius location neighbourhood.
    """

    pose = _validate_pose_batch(poses_w2c)
    global_value = np.asarray(global_scores, dtype=np.float64).reshape(-1)
    layout_value = np.asarray(layout_scores, dtype=np.float64).reshape(-1)
    count = pose.shape[0]
    if (
        global_value.shape != (count,) or layout_value.shape != (count,)
        or np.any(~np.isfinite(global_value)) or np.any(~np.isfinite(layout_value))
        or int(maximum_modes) <= 0 or int(orientations_per_location) <= 0
        or float(location_radius_m) <= 0.0 or float(orientation_nms_degrees) <= 0.0
    ):
        raise ValueError("invalid hierarchical location-orientation inputs")
    centers = -np.swapaxes(pose[:, :3, :3], 1, 2) @ pose[:, :3, 3, None]
    centers = centers[..., 0]
    location_budget = int(np.ceil(int(maximum_modes) / int(orientations_per_location)))
    # Pose samples near one location represent alternative orientations, not
    # independent geographical evidence.  Use a density-corrected local
    # log-mean-exp before spatial NMS so dense acquisition trajectories do not
    # receive a multiplicity prior.
    location_neighbors = _cached_location_neighborhoods(
        centers, float(location_radius_m)
    )
    location_value = np.empty((count,), dtype=np.float64)
    for row, neighborhood in enumerate(location_neighbors):
        values = global_value[neighborhood]
        maximum = float(np.max(values))
        location_value[row] = maximum + np.log(np.mean(np.exp(values - maximum)))
    global_order = np.lexsort((np.arange(count), -global_value, -location_value))
    location_seeds: list[int] = []
    for row in global_order:
        if all(
            np.linalg.norm(centers[int(row)] - centers[seed]) > float(location_radius_m)
            for seed in location_seeds
        ):
            location_seeds.append(int(row))
            if len(location_seeds) >= location_budget:
                break
    queues: list[list[int]] = []
    for seed in location_seeds:
        neighborhood = location_neighbors[seed]
        order = neighborhood[
            np.lexsort((neighborhood, -global_value[neighborhood], -layout_value[neighborhood]))
        ]
        orientation_rows: list[int] = []
        for row in order:
            if all(
                _rotation_distance_degrees(pose[int(row)], pose[kept])
                > float(orientation_nms_degrees)
                for kept in orientation_rows
            ):
                orientation_rows.append(int(row))
                if len(orientation_rows) >= int(orientations_per_location):
                    break
        queues.append(orientation_rows)
    retained: list[int] = []
    retained_by_location: list[list[int]] = [[] for _ in queues]
    for depth in range(int(orientations_per_location)):
        for location, queue in enumerate(queues):
            if depth >= len(queue):
                continue
            row = queue[depth]
            if row in retained:
                continue
            if any(
                np.linalg.norm(centers[row] - centers[kept]) <= float(translation_nms_m)
                and _rotation_distance_degrees(pose[row], pose[kept]) <= float(rotation_nms_deg)
                for kept in retained
            ):
                continue
            retained.append(row)
            retained_by_location[location].append(row)
            if len(retained) >= int(maximum_modes):
                return np.asarray(retained, dtype=np.int64)
    # Deterministic fallback preserves the hierarchy: first open new spatial
    # locations, then fill missing orientation quota inside existing ones, and
    # only then use an unrestricted global fallback.
    def conflicts(row: int) -> bool:
        return any(
            np.linalg.norm(centers[row] - centers[kept]) <= float(translation_nms_m)
            and _rotation_distance_degrees(pose[row], pose[kept]) <= float(rotation_nms_deg)
            for kept in retained
        )

    # Scan the complete stable global order.  Running a second full-library
    # greedy NMS here is both semantically redundant and O(N*selected) before
    # the hierarchy even sees a candidate.
    fallback_order = [int(row) for row in global_order if int(row) not in retained]
    for row in fallback_order:
        if conflicts(row):
            continue
        if all(
            np.linalg.norm(centers[row] - centers[seed]) > float(location_radius_m)
            for seed in location_seeds
        ):
            retained.append(row)
            location_seeds.append(row)
            queues.append([row])
            retained_by_location.append([row])
            if len(retained) >= int(maximum_modes):
                return np.asarray(retained, dtype=np.int64)

    for row in fallback_order:
        if row in retained or conflicts(row) or not location_seeds:
            continue
        distance = np.asarray([
            np.linalg.norm(centers[row] - centers[seed]) for seed in location_seeds
        ])
        location = int(np.argmin(distance))
        if distance[location] > float(location_radius_m):
            continue
        location_rows = retained_by_location[location]
        if len(location_rows) >= int(orientations_per_location):
            continue
        if any(
            _rotation_distance_degrees(pose[row], pose[kept])
            <= float(orientation_nms_degrees)
            for kept in location_rows
        ):
            continue
        retained.append(row)
        location_rows.append(row)
        if len(retained) >= int(maximum_modes):
            return np.asarray(retained, dtype=np.int64)

    for row in fallback_order:
        if row not in retained and not conflicts(row):
            retained.append(row)
            if len(retained) >= int(maximum_modes):
                return np.asarray(retained, dtype=np.int64)
    return np.asarray(retained, dtype=np.int64)
