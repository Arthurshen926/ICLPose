"""Exact 2DGS primitive -> child tile -> physical maplet hierarchy.

This module deliberately contains geometry and identity only.  It does not
store mapping images, observations, per-view descriptors, or pose signatures.
The canonical VFM field and its regenerable heads are separate artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    validate_deployment_metadata,
)
from feature_extract.vfm.surface_maplet_bank import VfmSurfaceMapletBank
from feature_extract.vfm.vfm_2dgs_mapping import Vfm2DgsAnchorMap


SINGLE_SIDED = np.uint8(1)
DOUBLE_SIDED = np.uint8(2)
SCHEMA = "goal_maplet_physical_map_v1"


def _unit(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    if np.all(np.abs(norm - 1.0) <= 1e-7):
        # Preserve already-canonical bytes so save/load lineage hashes remain
        # stable instead of renormalizing by 1+floating-point epsilon.
        return array.copy()
    return array / np.maximum(norm, 1e-12)


def _offsets(name: str, values: np.ndarray, rows: int, value_count: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.int64).reshape(-1)
    if (
        result.shape != (rows + 1,)
        or result[0] != 0
        or result[-1] != value_count
        or np.any(np.diff(result) < 0)
    ):
        raise ValueError(f"invalid {name}")
    return result


@dataclass(frozen=True)
class SurfacePrimitiveGeometry:
    """Geometry-only view of a surface-elements NPZ; adjacency is not loaded."""

    primitive_ids: np.ndarray
    centers: np.ndarray
    tangent1: np.ndarray
    tangent2: np.ndarray
    normals: np.ndarray
    scale1: np.ndarray
    scale2: np.ndarray
    opacity: np.ndarray

    def __post_init__(self) -> None:
        ids = np.asarray(self.primitive_ids, dtype=np.int64).reshape(-1)
        count = ids.size
        if np.unique(ids).size != count:
            raise ValueError("primitive IDs must be unique")
        object.__setattr__(self, "primitive_ids", ids)
        for name in ("centers", "tangent1", "tangent2", "normals"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (count, 3):
                raise ValueError(f"{name} must have shape (P,3)")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "normals", _unit(self.normals))
        for name in ("scale1", "scale2", "opacity"):
            value = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape (P,)")
            object.__setattr__(self, name, value)


def load_surface_primitive_geometry(path: Path) -> SurfacePrimitiveGeometry:
    """Load only geometry arrays, avoiding the historical 91M-edge adjacency."""

    with np.load(Path(path), allow_pickle=False) as data:
        return SurfacePrimitiveGeometry(
            primitive_ids=np.asarray(data["element_ids"], dtype=np.int64),
            centers=np.asarray(data["centers"], dtype=np.float64),
            tangent1=np.asarray(data["tangent1"], dtype=np.float64),
            tangent2=np.asarray(data["tangent2"], dtype=np.float64),
            normals=np.asarray(data["normals"], dtype=np.float64),
            scale1=np.asarray(data["scale1"], dtype=np.float64),
            scale2=np.asarray(data["scale2"], dtype=np.float64),
            opacity=np.asarray(data["opacity"], dtype=np.float64),
        )


@dataclass(frozen=True)
class GoalMapletPhysicalMap:
    maplet_ids: np.ndarray
    maplet_centers: np.ndarray
    maplet_normals: np.ndarray
    maplet_frames: np.ndarray
    maplet_extents: np.ndarray
    maplet_sidedness: np.ndarray
    maplet_orientation_confidence: np.ndarray
    primitive_ids: np.ndarray
    primitive_centers: np.ndarray
    primitive_tangent1: np.ndarray
    primitive_tangent2: np.ndarray
    primitive_normals: np.ndarray
    primitive_scale1: np.ndarray
    primitive_scale2: np.ndarray
    primitive_opacity: np.ndarray
    primitive_sidedness: np.ndarray
    primitive_orientation_confidence: np.ndarray
    membership_offsets: np.ndarray
    membership_primitive_rows: np.ndarray
    membership_weights: np.ndarray
    membership_boundary_weights: np.ndarray
    maplet_child_offsets: np.ndarray
    child_parent_rows: np.ndarray
    child_centers: np.ndarray
    child_normals: np.ndarray
    child_frames: np.ndarray
    child_extents: np.ndarray
    child_member_offsets: np.ndarray
    child_member_primitive_rows: np.ndarray
    child_member_weights: np.ndarray
    child_member_local_uv: np.ndarray
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        maplet_ids = np.asarray(self.maplet_ids, dtype=np.int64).reshape(-1)
        primitive_ids = np.asarray(self.primitive_ids, dtype=np.int64).reshape(-1)
        maplet_count, primitive_count = maplet_ids.size, primitive_ids.size
        if np.unique(maplet_ids).size != maplet_count:
            raise ValueError("maplet IDs must be unique")
        if np.unique(primitive_ids).size != primitive_count:
            raise ValueError("primitive IDs must be unique")
        object.__setattr__(self, "maplet_ids", maplet_ids)
        object.__setattr__(self, "primitive_ids", primitive_ids)
        for name, shape in (
            ("maplet_centers", (maplet_count, 3)),
            ("maplet_normals", (maplet_count, 3)),
            ("maplet_frames", (maplet_count, 3, 3)),
            ("maplet_extents", (maplet_count, 3)),
            ("primitive_centers", (primitive_count, 3)),
            ("primitive_tangent1", (primitive_count, 3)),
            ("primitive_tangent2", (primitive_count, 3)),
            ("primitive_normals", (primitive_count, 3)),
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "maplet_normals", _unit(self.maplet_normals))
        object.__setattr__(self, "primitive_normals", _unit(self.primitive_normals))
        for name, count in (
            ("maplet_sidedness", maplet_count),
            ("maplet_orientation_confidence", maplet_count),
            ("primitive_scale1", primitive_count),
            ("primitive_scale2", primitive_count),
            ("primitive_opacity", primitive_count),
            ("primitive_sidedness", primitive_count),
            ("primitive_orientation_confidence", primitive_count),
        ):
            value = np.asarray(getattr(self, name)).reshape(-1)
            if value.shape != (count,) or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value)
        if not set(np.unique(self.maplet_sidedness).tolist()).issubset({1, 2}):
            raise ValueError("invalid maplet sidedness")
        if not set(np.unique(self.primitive_sidedness).tolist()).issubset({1, 2}):
            raise ValueError("invalid primitive sidedness")
        member_rows = np.asarray(self.membership_primitive_rows, dtype=np.int64).reshape(-1)
        member_weights = np.asarray(self.membership_weights, dtype=np.float32).reshape(-1)
        boundary = np.asarray(self.membership_boundary_weights, dtype=np.float32).reshape(-1)
        if (
            member_rows.shape != member_weights.shape
            or boundary.shape != member_rows.shape
            or np.any(member_rows < 0)
            or np.any(member_rows >= primitive_count)
            or np.any(member_weights <= 0.0)
            or np.any((boundary < 0.0) | (boundary > 1.0))
        ):
            raise ValueError("invalid exact primitive membership")
        object.__setattr__(self, "membership_primitive_rows", member_rows)
        object.__setattr__(self, "membership_weights", member_weights)
        object.__setattr__(self, "membership_boundary_weights", boundary)
        object.__setattr__(self, "membership_offsets", _offsets(
            "membership_offsets", self.membership_offsets, maplet_count, member_rows.size
        ))
        child_parent = np.asarray(self.child_parent_rows, dtype=np.int64).reshape(-1)
        child_count = child_parent.size
        if np.any(child_parent < 0) or np.any(child_parent >= maplet_count):
            raise ValueError("invalid child parent rows")
        object.__setattr__(self, "child_parent_rows", child_parent)
        object.__setattr__(self, "maplet_child_offsets", _offsets(
            "maplet_child_offsets", self.maplet_child_offsets, maplet_count, child_count
        ))
        for name, shape in (
            ("child_centers", (child_count, 3)),
            ("child_normals", (child_count, 3)),
            ("child_frames", (child_count, 3, 3)),
            ("child_extents", (child_count, 3)),
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "child_normals", _unit(self.child_normals))
        child_member_rows = np.asarray(self.child_member_primitive_rows, dtype=np.int64).reshape(-1)
        child_member_weights = np.asarray(self.child_member_weights, dtype=np.float32).reshape(-1)
        local_uv = np.asarray(self.child_member_local_uv, dtype=np.float32)
        if (
            child_member_rows.shape != child_member_weights.shape
            or local_uv.shape != (child_member_rows.size, 2)
            or np.any(child_member_rows < 0)
            or np.any(child_member_rows >= primitive_count)
            or np.any(child_member_weights <= 0.0)
        ):
            raise ValueError("invalid child primitive membership")
        object.__setattr__(self, "child_member_primitive_rows", child_member_rows)
        object.__setattr__(self, "child_member_weights", child_member_weights)
        object.__setattr__(self, "child_member_local_uv", local_uv)
        object.__setattr__(self, "child_member_offsets", _offsets(
            "child_member_offsets", self.child_member_offsets, child_count, child_member_rows.size
        ))
        metadata = dict(self.metadata or {})
        if metadata.get("artifact_type", SCHEMA) != SCHEMA:
            raise ValueError("not a Goal-Maplet physical map")
        validate_deployment_metadata(metadata)
        object.__setattr__(self, "metadata", metadata)

    def _arrays(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(getattr(self, name))
            for name in self.__dataclass_fields__
            if name != "metadata"
        }

    @property
    def content_sha256(self) -> str:
        return arrays_sha256(self._arrays())

    def member_slice(self, maplet_row: int) -> slice:
        return slice(int(self.membership_offsets[maplet_row]), int(self.membership_offsets[maplet_row + 1]))

    def save_npz(self, path: Path) -> None:
        metadata = dict(self.metadata or {})
        metadata["artifact_type"] = SCHEMA
        metadata["content_sha256"] = self.content_sha256
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            Path(path),
            **self._arrays(),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path, *, verify_hash: bool = True) -> "GoalMapletPhysicalMap":
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
            kwargs = {
                name: np.asarray(data[name])
                for name in cls.__dataclass_fields__
                if name != "metadata"
            }
        declared = str(metadata.get("content_sha256", ""))
        if verify_hash and declared and declared != arrays_sha256(kwargs):
            raise ValueError("Goal-Maplet physical map content hash mismatch")
        return cls(**kwargs, metadata=metadata)


def _camera_center(pose_w2c: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64)
    return -pose[:3, :3].T @ pose[:3, 3]


def _oriented_frame(normal: np.ndarray, seed_frame: np.ndarray, points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n = _unit(np.asarray(normal).reshape(1, 3))[0]
    offsets = np.asarray(points, dtype=np.float64) - np.average(points, axis=0, weights=weights)
    covariance = (offsets * weights[:, None]).T @ offsets / max(float(np.sum(weights)), 1e-12)
    values, vectors = np.linalg.eigh(covariance)
    tangent = vectors[:, int(np.argmax(values))]
    tangent -= n * float(np.dot(tangent, n))
    if np.linalg.norm(tangent) < 1e-8:
        tangent = np.asarray(seed_frame, dtype=np.float64)[0]
        tangent -= n * float(np.dot(tangent, n))
    tangent = _unit(tangent.reshape(1, 3))[0]
    if float(np.dot(tangent, np.asarray(seed_frame)[0])) < 0.0:
        tangent *= -1.0
    second = _unit(np.cross(n, tangent).reshape(1, 3))[0]
    return np.stack([tangent, second, n], axis=0)


def _recursive_spatial_partition(local: np.ndarray, target: int, minimum_size: int = 2) -> list[np.ndarray]:
    groups = [np.arange(local.shape[0], dtype=np.int64)]
    while len(groups) < target:
        choices = [
            (float(np.max(np.ptp(local[group], axis=0))), int(group.size), idx)
            for idx, group in enumerate(groups)
            if group.size >= 2 * int(minimum_size)
        ]
        if not choices:
            break
        _, _, selected = max(choices)
        group = groups.pop(selected)
        axis = int(np.argmax(np.ptp(local[group], axis=0)))
        order = group[np.argsort(local[group, axis], kind="stable")]
        cut = max(int(minimum_size), min(order.size - int(minimum_size), order.size // 2))
        groups.extend([order[:cut], order[cut:]])
    return sorted(groups, key=lambda row: tuple(np.mean(local[row], axis=0).tolist()))


def _refine_surface_groups_to_target(
    local: np.ndarray,
    groups: list[np.ndarray],
    target: int,
    minimum_size: int = 2,
) -> list[np.ndarray]:
    result = list(groups)
    while len(result) < int(target):
        choices = [
            (float(np.max(np.ptp(local[group], axis=0))), int(group.size), idx)
            for idx, group in enumerate(result)
            if group.size >= 2 * int(minimum_size)
        ]
        if not choices:
            break
        _, _, selected = max(choices)
        group = result.pop(selected)
        axis = int(np.argmax(np.ptp(local[group], axis=0)))
        order = group[np.argsort(local[group, axis], kind="stable")]
        cut = max(int(minimum_size), min(order.size - int(minimum_size), order.size // 2))
        result.extend([order[:cut], order[cut:]])
    return sorted(result, key=lambda row: tuple(np.mean(local[row], axis=0).tolist()))


def _hard_surface_partition(
    local: np.ndarray,
    normals: np.ndarray,
    *,
    maximum_groups: int,
    minimum_size: int = 2,
    maximum_normal_p90_degrees: float = 20.0,
    maximum_relative_depth_span: float = 0.20,
) -> list[np.ndarray]:
    """Split front/back, normal discontinuities, and distinct depth layers."""

    pending = [np.arange(local.shape[0], dtype=np.int64)]
    accepted: list[np.ndarray] = []
    while pending:
        group = pending.pop(0)
        if len(accepted) + len(pending) + 1 >= int(maximum_groups) or group.size < 2 * int(minimum_size):
            accepted.append(group)
            continue
        group_normals = np.asarray(normals[group], dtype=np.float64)
        mean_normal = _unit(np.sum(group_normals, axis=0).reshape(1, 3))[0]
        angles = np.degrees(np.arccos(np.clip(group_normals @ mean_normal, -1.0, 1.0)))
        split_order: np.ndarray | None = None
        if float(np.percentile(angles, 90.0)) > float(maximum_normal_p90_degrees):
            centered = group_normals - np.mean(group_normals, axis=0, keepdims=True)
            _, _, axes = np.linalg.svd(centered, full_matrices=False)
            scores = centered @ axes[0]
            split_order = group[np.argsort(scores, kind="stable")]
        if split_order is None:
            values = local[group, 2]
            order = np.argsort(values, kind="stable")
            ordered = values[order]
            gaps = np.diff(ordered)
            planar_span = max(float(np.ptp(local[group, :2], axis=0).max()), 1e-6)
            relative_depth = float(np.ptp(values) / planar_span)
            if gaps.size and relative_depth > float(maximum_relative_depth_span):
                cut_at = int(np.argmax(gaps)) + 1
                if float(gaps[cut_at - 1]) >= max(0.02, 0.05 * planar_span):
                    split_order = group[order]
                    cut = cut_at
                else:
                    cut = group.size // 2
            else:
                cut = group.size // 2
        else:
            cut = group.size // 2
        if split_order is None or cut < minimum_size or split_order.size - cut < minimum_size:
            accepted.append(group)
        else:
            pending.extend([split_order[:cut], split_order[cut:]])
    return accepted


def _boundary_weights(local_uv: np.ndarray) -> np.ndarray:
    if local_uv.shape[0] <= 3:
        return np.ones((local_uv.shape[0],), dtype=np.float32)
    lo, hi = np.min(local_uv, axis=0), np.max(local_uv, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    edge_distance = np.min(np.stack([
        (local_uv[:, 0] - lo[0]) / span[0],
        (hi[0] - local_uv[:, 0]) / span[0],
        (local_uv[:, 1] - lo[1]) / span[1],
        (hi[1] - local_uv[:, 1]) / span[1],
    ], axis=1), axis=1)
    return np.clip(1.0 - 4.0 * edge_distance, 0.0, 1.0).astype(np.float32)


def build_goal_maplet_physical_map(
    maplets: VfmSurfaceMapletBank,
    region_map: Vfm2DgsAnchorMap,
    geometry: SurfacePrimitiveGeometry,
    pose_w2c_by_image: Mapping[str, np.ndarray],
    *,
    clean_primitive_ids: np.ndarray | None = None,
    minimum_orientation_confidence: float = 0.15,
    minimum_child_count: int = 8,
    maximum_child_count: int = 32,
    metadata: Mapping[str, object] | None = None,
) -> GoalMapletPhysicalMap:
    """Recover exact physical membership and build metric child surface tiles."""

    geometry_row = {int(value): row for row, value in enumerate(geometry.primitive_ids.tolist())}
    region_row = {int(value): row for row, value in enumerate(region_map.anchor_ids.tolist())}
    clean = None if clean_primitive_ids is None else set(np.asarray(clean_primitive_ids, dtype=np.int64).tolist())
    per_maplet_ids: list[np.ndarray] = []
    per_maplet_weights: list[np.ndarray] = []
    per_maplet_camera_centers: list[np.ndarray] = []
    for row, maplet_id in enumerate(maplets.maplet_ids.tolist()):
        start, end = int(maplets.support_offsets[row]), int(maplets.support_offsets[row + 1])
        ids = np.asarray(maplets.support_element_ids[start:end], dtype=np.int64)
        valid = np.asarray([int(value) in geometry_row and (clean is None or int(value) in clean) for value in ids], dtype=bool)
        ids = ids[valid]
        if ids.size == 0:
            raise ValueError(f"maplet {maplet_id} has no declared clean physical primitive")
        rrow = region_row.get(int(maplet_id))
        weights_by_id: dict[int, float] = {}
        camera_centers: list[np.ndarray] = []
        if rrow is not None:
            rstart, rend = int(region_map.support_offsets[rrow]), int(region_map.support_offsets[rrow + 1])
            weights_by_id = {
                int(pid): float(weight)
                for pid, weight in zip(
                    region_map.support_element_ids[rstart:rend].tolist(),
                    region_map.support_weights[rstart:rend].tolist(),
                )
            }
            for image_id in region_map.observed_view_ids[rrow]:
                if image_id in pose_w2c_by_image:
                    camera_centers.append(_camera_center(pose_w2c_by_image[image_id]))
        weights = np.asarray([max(weights_by_id.get(int(value), 1.0), 1e-6) for value in ids], dtype=np.float64)
        per_maplet_ids.append(ids)
        per_maplet_weights.append(weights)
        per_maplet_camera_centers.append(np.asarray(camera_centers, dtype=np.float64).reshape(-1, 3))

    # Keep the complete declared clean scene as an occluder table, not merely
    # the primitives owned by maplets.  Membership remains sparse below.  This
    # is what lets later visibility use a full-scene z-buffer instead of a
    # self-occlusion-only approximation.
    if clean_primitive_ids is None:
        unique_ids = np.unique(np.concatenate(per_maplet_ids)).astype(np.int64)
    else:
        unique_ids = np.asarray(
            sorted(set(np.asarray(clean_primitive_ids, dtype=np.int64).tolist()).intersection(geometry_row)),
            dtype=np.int64,
        )
    source_rows = np.asarray([geometry_row[int(value)] for value in unique_ids], dtype=np.int64)
    compact_row = {int(value): row for row, value in enumerate(unique_ids.tolist())}
    primitive_normals = np.asarray(geometry.normals[source_rows], dtype=np.float64).copy()
    primitive_tangent1 = np.asarray(geometry.tangent1[source_rows], dtype=np.float64).copy()
    primitive_tangent2 = np.asarray(geometry.tangent2[source_rows], dtype=np.float64).copy()

    oriented_maplet_normals: list[np.ndarray] = []
    maplet_sidedness: list[int] = []
    maplet_confidence: list[float] = []
    for row, cameras in enumerate(per_maplet_camera_centers):
        normal = np.asarray(maplets.normals[row], dtype=np.float64)
        if cameras.size == 0:
            score = 0.0
        else:
            direction = _unit(cameras - np.asarray(maplets.centers[row], dtype=np.float64))
            signed = direction @ normal
            score = float(np.mean(signed))
        if score < 0.0:
            normal = -normal
        confidence = abs(score)
        oriented_maplet_normals.append(_unit(normal.reshape(1, 3))[0])
        maplet_confidence.append(confidence)
        maplet_sidedness.append(int(SINGLE_SIDED if confidence >= minimum_orientation_confidence else DOUBLE_SIDED))

    primitive_votes: list[list[tuple[np.ndarray, float, int]]] = [[] for _ in range(unique_ids.size)]
    for maplet_row, (ids, weights) in enumerate(zip(per_maplet_ids, per_maplet_weights)):
        for pid, weight in zip(ids.tolist(), weights.tolist()):
            primitive_votes[compact_row[int(pid)]].append((oriented_maplet_normals[maplet_row], float(weight), maplet_sidedness[maplet_row]))
    primitive_confidence = np.zeros((unique_ids.size,), dtype=np.float32)
    primitive_sidedness = np.full((unique_ids.size,), int(DOUBLE_SIDED), dtype=np.uint8)
    for row, votes in enumerate(primitive_votes):
        if not votes:
            # Clean scene primitives outside any maplet are retained only as
            # occluders.  Their PLY normal sign is unknown, so they stay
            # explicitly double-sided rather than being guessed.
            continue
        direction = np.sum([normal * weight for normal, weight, _ in votes], axis=0)
        confidence = float(np.linalg.norm(direction) / max(sum(weight for _, weight, _ in votes), 1e-12))
        if np.dot(primitive_normals[row], direction) < 0.0:
            primitive_normals[row] *= -1.0
            primitive_tangent2[row] *= -1.0
        primitive_confidence[row] = confidence
        if confidence >= minimum_orientation_confidence and all(side == int(SINGLE_SIDED) for _, _, side in votes):
            primitive_sidedness[row] = SINGLE_SIDED

    membership_offsets = [0]
    membership_rows: list[int] = []
    membership_weights: list[float] = []
    membership_boundary: list[float] = []
    maplet_centers: list[np.ndarray] = []
    maplet_normals: list[np.ndarray] = []
    maplet_frames: list[np.ndarray] = []
    maplet_extents: list[np.ndarray] = []
    maplet_child_offsets = [0]
    child_parent_rows: list[int] = []
    child_centers: list[np.ndarray] = []
    child_normals: list[np.ndarray] = []
    child_frames: list[np.ndarray] = []
    child_extents: list[np.ndarray] = []
    child_member_offsets = [0]
    child_member_rows: list[int] = []
    child_member_weights: list[float] = []
    child_member_local_uv: list[np.ndarray] = []
    primitive_centers = np.asarray(geometry.centers[source_rows], dtype=np.float64)
    primitive_scale1 = np.asarray(geometry.scale1[source_rows], dtype=np.float64)
    primitive_scale2 = np.asarray(geometry.scale2[source_rows], dtype=np.float64)
    for maplet_row, (ids, weights) in enumerate(zip(per_maplet_ids, per_maplet_weights)):
        rows = np.asarray([compact_row[int(value)] for value in ids], dtype=np.int64)
        points = primitive_centers[rows]
        area_weight = weights * np.pi * primitive_scale1[rows] * primitive_scale2[rows]
        center = np.average(points, axis=0, weights=np.maximum(area_weight, 1e-12))
        normal = _unit(np.sum(primitive_normals[rows] * area_weight[:, None], axis=0).reshape(1, 3))[0]
        if float(np.dot(normal, oriented_maplet_normals[maplet_row])) < 0.0:
            normal *= -1.0
        frame = _oriented_frame(normal, maplets.tangent_frames[maplet_row], points, area_weight)
        local = (points - center) @ frame.T
        radius = np.maximum(primitive_scale1[rows], primitive_scale2[rows])
        extent = np.max(np.abs(local) + radius[:, None], axis=0)
        boundary = _boundary_weights(local[:, :2])
        maplet_centers.append(center)
        maplet_normals.append(normal)
        maplet_frames.append(frame)
        maplet_extents.append(extent)
        membership_rows.extend(rows.tolist())
        membership_weights.extend(weights.tolist())
        membership_boundary.extend(boundary.tolist())
        membership_offsets.append(len(membership_rows))
        target = int(np.clip(round(np.sqrt(rows.size)), minimum_child_count, maximum_child_count))
        target = min(target, max(1, rows.size // 2))
        hard_groups = _hard_surface_partition(
            local,
            primitive_normals[rows],
            maximum_groups=max(target, 1),
        )
        groups = _refine_surface_groups_to_target(local, hard_groups, target)
        for group in groups:
            group_rows = rows[group]
            group_weights = area_weight[group]
            group_points = primitive_centers[group_rows]
            child_center = np.average(group_points, axis=0, weights=np.maximum(group_weights, 1e-12))
            local_normals = primitive_normals[group_rows].copy()
            local_double = primitive_sidedness[group_rows] == DOUBLE_SIDED
            flip = local_double & ((local_normals @ normal) < 0.0)
            local_normals[flip] *= -1.0
            child_normal = _unit(np.sum(local_normals * group_weights[:, None], axis=0).reshape(1, 3))[0]
            if np.dot(child_normal, normal) < 0.0:
                child_normal *= -1.0
            child_frame = _oriented_frame(child_normal, frame, group_points, group_weights)
            child_local = (group_points - child_center) @ child_frame.T
            child_radius = np.maximum(primitive_scale1[group_rows], primitive_scale2[group_rows])
            child_extent = np.max(np.abs(child_local) + child_radius[:, None], axis=0)
            child_parent_rows.append(maplet_row)
            child_centers.append(child_center)
            child_normals.append(child_normal)
            child_frames.append(child_frame)
            child_extents.append(child_extent)
            child_member_rows.extend(group_rows.tolist())
            child_member_weights.extend(weights[group].tolist())
            child_member_local_uv.extend(child_local[:, :2].astype(np.float32))
            child_member_offsets.append(len(child_member_rows))
        maplet_child_offsets.append(len(child_parent_rows))

    output_metadata = {
        "artifact_type": SCHEMA,
        "representation": "exact_2dgs_primitive_child_tile_maplet_hierarchy",
        "vfm_layer": "radio_final",
        "stores_mapping_rgb": False,
        "stores_mapping_image_paths": False,
        "stores_mapping_image_ids": False,
        "uses_alike_descriptors": False,
        "uses_radio_intermediate": False,
        "uses_sfm_points": False,
        "uses_sfm_tracks": False,
        "uses_point_correspondences": False,
        "normal_orientation": "mapping_camera_consensus_with_explicit_double_sided_fallback",
        "minimum_orientation_confidence": float(minimum_orientation_confidence),
        "child_partition": "recursive_metric_surface_partition",
        **dict(metadata or {}),
    }
    return GoalMapletPhysicalMap(
        maplet_ids=np.asarray(maplets.maplet_ids, dtype=np.int64),
        maplet_centers=np.asarray(maplet_centers, dtype=np.float64),
        maplet_normals=np.asarray(maplet_normals, dtype=np.float64),
        maplet_frames=np.asarray(maplet_frames, dtype=np.float64),
        maplet_extents=np.asarray(maplet_extents, dtype=np.float64),
        maplet_sidedness=np.asarray(maplet_sidedness, dtype=np.uint8),
        maplet_orientation_confidence=np.asarray(maplet_confidence, dtype=np.float32),
        primitive_ids=unique_ids,
        primitive_centers=primitive_centers,
        primitive_tangent1=primitive_tangent1,
        primitive_tangent2=primitive_tangent2,
        primitive_normals=primitive_normals,
        primitive_scale1=primitive_scale1,
        primitive_scale2=primitive_scale2,
        primitive_opacity=np.asarray(geometry.opacity[source_rows], dtype=np.float64),
        primitive_sidedness=primitive_sidedness,
        primitive_orientation_confidence=primitive_confidence,
        membership_offsets=np.asarray(membership_offsets, dtype=np.int64),
        membership_primitive_rows=np.asarray(membership_rows, dtype=np.int64),
        membership_weights=np.asarray(membership_weights, dtype=np.float32),
        membership_boundary_weights=np.asarray(membership_boundary, dtype=np.float32),
        maplet_child_offsets=np.asarray(maplet_child_offsets, dtype=np.int64),
        child_parent_rows=np.asarray(child_parent_rows, dtype=np.int64),
        child_centers=np.asarray(child_centers, dtype=np.float64).reshape(-1, 3),
        child_normals=np.asarray(child_normals, dtype=np.float64).reshape(-1, 3),
        child_frames=np.asarray(child_frames, dtype=np.float64).reshape(-1, 3, 3),
        child_extents=np.asarray(child_extents, dtype=np.float64).reshape(-1, 3),
        child_member_offsets=np.asarray(child_member_offsets, dtype=np.int64),
        child_member_primitive_rows=np.asarray(child_member_rows, dtype=np.int64),
        child_member_weights=np.asarray(child_member_weights, dtype=np.float32),
        child_member_local_uv=np.asarray(child_member_local_uv, dtype=np.float32).reshape(-1, 2),
        metadata=output_metadata,
    )
