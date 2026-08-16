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
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse

from .lineage import arrays_sha256
from .physical_map import GoalMapletPhysicalMap
from .pose_proposal import _rotation_distance_degrees
from .pure_retrieval import PureRadioPhysicalRetrieval, all_radio_token_coordinates


SCHEMA = "goal_maplet_child_visibility_pose_atlas_v1"
SCORE_SEMANTICS = "global_and_fixed_grid_child_bhattacharyya_affinity_v1"


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
        return arrays_sha256({
            "poses_w2c": self.poses_w2c,
            "global_offsets": self.global_offsets,
            "global_child_rows": self.global_child_rows,
            "global_weights": self.global_weights,
            "layout_offsets": self.layout_offsets,
            "layout_keys": self.layout_keys,
            "layout_weights": self.layout_weights,
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
        local_keys: list[np.ndarray] = []
        local_weight: list[np.ndarray] = []
        for cell_row in range(cell_count):
            rows, value = _top_normalized_sqrt(
                layout_mass[cell_row],
                maximum_entries=int(maximum_children_per_cell),
                scale=1.0 / np.sqrt(float(cell_count)),
            )
            local_keys.append((cell_row * child_count + rows).astype(np.int32))
            local_weight.append(value)
        keys = np.concatenate(local_keys)
        values = np.concatenate(local_weight)
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
            "grid_semantics": "fixed_equal_image_blocks_coarse_layout",
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
    layout_keys: list[np.ndarray] = []
    layout_weight: list[np.ndarray] = []
    cell_count = atlas.grid_rows * atlas.grid_cols
    for cell_row in range(cell_count):
        local_rows, local_value = _top_normalized_sqrt(
            layout_mass[cell_row],
            maximum_entries=atlas.child_count,
            scale=1.0 / np.sqrt(float(cell_count)),
        )
        layout_keys.append(cell_row * atlas.child_count + local_rows)
        layout_weight.append(local_value)
    keys = np.concatenate(layout_keys).astype(np.int32)
    values = np.concatenate(layout_weight).astype(np.float32)
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
    selected_children_only: bool = True,
    matrices: tuple[sparse.csr_matrix, sparse.csr_matrix] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score every chart centre without claiming a calibrated pose posterior."""

    alpha = float(layout_weight)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("layout_weight must lie in [0,1]")
    global_query, layout_query = query_child_affinity_vectors(
        retrieval, atlas, selected_children_only=selected_children_only
    )
    global_atlas, layout_atlas = atlas.sparse_matrices() if matrices is None else matrices
    global_product = global_atlas @ global_query.T
    layout_product = layout_atlas @ layout_query.T
    global_score = np.asarray(global_product.toarray()).reshape(-1)
    layout_score = np.asarray(layout_product.toarray()).reshape(-1)
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
