"""Source-only projective exact-face seam authority.

This module intentionally does not use a world-space nearest surface query.
For every directed overlap sample it projects a source M0 vertex into the
target source camera, selects the front-most *exact v3 face* at that ray, and
freezes the target face plus perspective-correct barycentric coordinates.

The authority is built before an aligned arm or held geometry is opened.  A
separate evaluator may later replay the frozen material correspondences.  Its
hard statistics are directional; opposite directions are never pooled before
quantiles are computed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from .chart_comparison_domain import (
    BASE_ARRAY_NAMES,
    source_tree_sha256,
    topology_array_names,
    validate_exact_topology_arrays,
)
from .chart_submap_selection import ChartSubmapPlan
from .lineage import arrays_sha256, canonical_json_sha256, file_sha256


AUTHORITY_SCHEMA = "goal_maplet_projective_exact_face_seam_authority_v2"
REPORT_SCHEMA = "goal_maplet_projective_exact_face_seam_geometry_gate_v3"
AUTHORITY_SEMANTICS_VERSION = (
    "source_ray_projected_exact_face_visibility_same_side_v1"
)
PHYSICAL_DOMAIN_SCHEMA = "goal_maplet_chart_comparison_domain_v3"
DISJOINT_AUTHORITY_SCHEMA = "goal_maplet_disjoint_chart_upstream_authority_v2"
EDGE_FORMAL_VALID_DEFINITION = (
    "AND of both direction formal-valid rows; no bidirectional pooling"
)
MATERIAL_WELD_SEMANTICS_VERSION = (
    "frozen_M0_material_pair_euclidean_distance_excess_directional_v1"
)
MATERIAL_WELD_EDGE_VALID_DEFINITION = (
    "AND of both direction material-weld-valid rows; no bidirectional pooling"
)
MAX_PROJECTIVE_RAY_FACE_PAIRS_PER_CHUNK = 250_000


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _replay_metadata(metadata: Mapping[str, object], label: str) -> str:
    replay = dict(metadata)
    claimed = replay.pop("content_sha256", None)
    if not _is_sha256(claimed) or claimed != canonical_json_sha256(replay):
        raise ValueError(f"{label} metadata content hash differs")
    return str(claimed)


def _unit_rows(offsets: np.ndarray) -> np.ndarray:
    offsets = np.asarray(offsets, np.int64)
    return np.concatenate(
        [
            np.full(int(offsets[row + 1] - offsets[row]), row, np.int32)
            for row in range(len(offsets) - 1)
        ]
    )


def _surface_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, np.float64)
    faces = np.asarray(faces, np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("projective seam vertices must have shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("projective seam faces must have shape (F, 3)")
    if np.any((faces < 0) | (faces >= len(vertices))):
        raise ValueError("projective seam face leaves the vertex inventory")
    triangle = vertices[faces]
    face_normal = np.cross(
        triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
    )
    normal = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normal, faces[:, corner], face_normal)
    length = np.linalg.norm(normal, axis=1)
    if np.any(~np.isfinite(length)) or np.any(length <= 1e-10):
        raise ValueError("projective seam topology has a degenerate vertex normal")
    return normal / length[:, None]


def _common_valid_normal_stencil(valid: np.ndarray) -> np.ndarray:
    """Five-pixel stencil used by the source-only root-cause denominator."""

    valid = np.asarray(valid, bool)
    if valid.ndim != 2:
        raise ValueError("projective seam common-valid mask must be two-dimensional")
    output = np.zeros_like(valid)
    output[1:-1, 1:-1] = (
        valid[1:-1, 1:-1]
        & valid[1:-1, :-2]
        & valid[1:-1, 2:]
        & valid[:-2, 1:-1]
        & valid[2:, 1:-1]
    )
    return output


def _dense_world_normals(
    points_world: np.ndarray, eligible: np.ndarray
) -> np.ndarray:
    points_world = np.asarray(points_world, np.float64)
    eligible = np.asarray(eligible, bool)
    if points_world.shape[:2] != eligible.shape or points_world.shape[2:] != (3,):
        raise ValueError("projective seam dense normal inputs differ")
    normal = np.zeros_like(points_world)
    normal[1:-1, 1:-1] = np.cross(
        points_world[1:-1, 2:] - points_world[1:-1, :-2],
        points_world[2:, 1:-1] - points_world[:-2, 1:-1],
    )
    length = np.linalg.norm(normal, axis=2)
    good = eligible & np.isfinite(length) & (length > 1e-10)
    normal[good] /= length[good, None]
    normal[~good] = 0.0
    return normal


def _project_world(
    points_world: np.ndarray,
    camera_to_world: np.ndarray,
    focal_px: float,
    principal_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points_world = np.asarray(points_world, np.float64)
    camera_to_world = np.asarray(camera_to_world, np.float64)
    camera = (points_world - camera_to_world[:3, 3]) @ camera_to_world[:3, :3]
    z = camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.column_stack(
            (
                focal_px * camera[:, 0] / z + principal_xy[0],
                focal_px * camera[:, 1] / z + principal_xy[1],
            )
        )
    return uv, z


def _screen_barycentric(
    points_uv: np.ndarray, triangles_uv: np.ndarray, *, epsilon: float = 1e-9
) -> tuple[np.ndarray, np.ndarray]:
    """Return inclusive containment and screen-space barycentrics."""

    points_uv = np.asarray(points_uv, np.float64)
    triangles_uv = np.asarray(triangles_uv, np.float64)
    a = triangles_uv[:, 0]
    first = triangles_uv[:, 1] - a
    second = triangles_uv[:, 2] - a
    denominator = first[:, 0] * second[:, 1] - second[:, 0] * first[:, 1]
    displacement = points_uv[:, None, :] - a[None, :, :]
    beta = (
        displacement[:, :, 0] * second[None, :, 1]
        - second[None, :, 0] * displacement[:, :, 1]
    ) / denominator[None, :]
    gamma = (
        first[None, :, 0] * displacement[:, :, 1]
        - displacement[:, :, 0] * first[None, :, 1]
    ) / denominator[None, :]
    alpha = 1.0 - beta - gamma
    barycentric = np.stack((alpha, beta, gamma), axis=2)
    finite_triangle = np.isfinite(triangles_uv).all((1, 2)) & (
        np.abs(denominator) > 1e-12
    )
    inside = (
        finite_triangle[None, :]
        & np.isfinite(barycentric).all(2)
        & np.all(barycentric >= -epsilon, axis=2)
        & np.all(barycentric <= 1.0 + epsilon, axis=2)
    )
    return inside, barycentric


def _perspective_correct_barycentric(
    screen_barycentric: np.ndarray, corner_depth: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    screen_barycentric = np.asarray(screen_barycentric, np.float64)
    corner_depth = np.asarray(corner_depth, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        weighted = screen_barycentric / corner_depth
        denominator = np.sum(weighted, axis=-1, keepdims=True)
        material = weighted / denominator
        depth = 1.0 / denominator[..., 0]
    return material, depth


def _frontmost_exact_face_at_rays(
    points_uv: np.ndarray,
    target_face_indices: np.ndarray,
    faces: np.ndarray,
    target_vertices_uv: np.ndarray,
    target_vertices_depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose the front-most exact target face at every continuous ray.

    The front-most rule is the visibility rule.  Selecting the face whose
    depth is closest to the source depth would permit a hidden, parallel
    surface to bypass the occlusion gate.
    """

    points_uv = np.asarray(points_uv, np.float64)
    target_face_indices = np.asarray(target_face_indices, np.int64)
    face_corners = np.asarray(faces, np.int64)[target_face_indices]
    corner_uv = np.asarray(target_vertices_uv, np.float64)[face_corners]
    corner_depth = np.asarray(target_vertices_depth, np.float64)[face_corners]
    inside, screen = _screen_barycentric(points_uv, corner_uv)
    valid_depth = np.isfinite(corner_depth).all(1) & np.all(corner_depth > 0, axis=1)
    material, interpolated_depth = _perspective_correct_barycentric(
        screen, corner_depth[None, :, :]
    )
    visible_candidate = (
        inside
        & valid_depth[None, :]
        & np.isfinite(interpolated_depth)
        & (interpolated_depth > 0)
        & np.isfinite(material).all(2)
    )
    candidate_depth = np.where(visible_candidate, interpolated_depth, np.inf)
    local_face = np.argmin(candidate_depth, axis=1)
    row = np.arange(len(points_uv))
    best_depth = candidate_depth[row, local_face]
    found = np.isfinite(best_depth)
    output_face = np.full(len(points_uv), -1, np.int64)
    output_screen = np.zeros((len(points_uv), 3), np.float64)
    output_material = np.zeros((len(points_uv), 3), np.float64)
    output_depth = np.full(len(points_uv), np.nan, np.float64)
    output_face[found] = target_face_indices[local_face[found]]
    output_screen[found] = screen[row[found], local_face[found]]
    output_material[found] = material[row[found], local_face[found]]
    output_depth[found] = best_depth[found]
    return output_face, output_screen, output_material, output_depth


def _chunked_frontmost_exact_face_at_rays(
    points_uv: np.ndarray,
    target_face_indices: np.ndarray,
    faces: np.ndarray,
    target_vertices_uv: np.ndarray,
    target_vertices_depth: np.ndarray,
    *,
    maximum_ray_face_pairs: int = MAX_PROJECTIVE_RAY_FACE_PAIRS_PER_CHUNK,
    reverse_chunk_order: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bound projection workspace without changing per-ray face semantics.

    Rays are independent.  Chunk results are scattered back to their original
    rows, so forward and reverse chunk traversal must be bit-identical.
    """

    points_uv = np.asarray(points_uv, np.float64)
    target_face_indices = np.asarray(target_face_indices, np.int64)
    if maximum_ray_face_pairs < 1:
        raise ValueError("projective seam ray/face chunk budget must be positive")
    if not len(points_uv):
        return (
            np.zeros((0,), np.int64),
            np.zeros((0, 3), np.float64),
            np.zeros((0, 3), np.float64),
            np.zeros((0,), np.float64),
        )
    face_count = max(len(target_face_indices), 1)
    rays_per_chunk = max(1, maximum_ray_face_pairs // face_count)
    chunks = [
        np.arange(lo, min(lo + rays_per_chunk, len(points_uv)), dtype=np.int64)
        for lo in range(0, len(points_uv), rays_per_chunk)
    ]
    if reverse_chunk_order:
        chunks.reverse()
    output_face = np.full(len(points_uv), -1, np.int64)
    output_screen = np.zeros((len(points_uv), 3), np.float64)
    output_material = np.zeros((len(points_uv), 3), np.float64)
    output_depth = np.full(len(points_uv), np.nan, np.float64)
    for rows in chunks:
        values = _frontmost_exact_face_at_rays(
            points_uv[rows],
            target_face_indices,
            faces,
            target_vertices_uv,
            target_vertices_depth,
        )
        output_face[rows] = values[0]
        output_screen[rows] = values[1]
        output_material[rows] = values[2]
        output_depth[rows] = values[3]
    return output_face, output_screen, output_material, output_depth


@dataclass(frozen=True)
class ProjectiveSeamConfig:
    topology_stride: int = 4
    absolute_depth_tolerance_m: float = 0.30
    relative_depth_tolerance: float = 0.025
    maximum_reference_unsigned_normal_angle_deg: float = 35.0
    minimum_correspondences_per_direction: int = 4
    minimum_supported_fraction: float = 0.05
    frozen_overlap_support_fraction_multiplier: float = 0.50
    maximum_point_to_plane_p50_m: float = 0.10
    maximum_point_to_plane_p90_m: float = 0.30
    maximum_unsigned_normal_p90_deg: float = 30.0
    maximum_oriented_normal_p90_deg: float = 60.0
    maximum_opposed_normal_fraction: float = 0.05
    maximum_selected_chart_self_reprojection_p90_px: float = 2.0

    def validated(self) -> "ProjectiveSeamConfig":
        if self.topology_stride not in (2, 4):
            raise ValueError(
                "projective seam authority supports only sealed stride-2 "
                "diagnostics or the formal stride-4 topology"
            )
        if self.absolute_depth_tolerance_m <= 0 or not 0 < self.relative_depth_tolerance < 1:
            raise ValueError("projective seam depth tolerance is invalid")
        if not 0 < self.maximum_reference_unsigned_normal_angle_deg <= 90:
            raise ValueError("projective seam reference normal threshold is invalid")
        if self.minimum_correspondences_per_direction < 1:
            raise ValueError("projective seam correspondence count must be positive")
        if not 0 < self.minimum_supported_fraction <= 1:
            raise ValueError("projective seam minimum support is invalid")
        if not 0 < self.frozen_overlap_support_fraction_multiplier <= 1:
            raise ValueError("projective seam overlap multiplier is invalid")
        if not 0 < self.maximum_point_to_plane_p50_m <= self.maximum_point_to_plane_p90_m:
            raise ValueError("projective seam point-to-plane thresholds are invalid")
        if not 0 < self.maximum_unsigned_normal_p90_deg <= 90:
            raise ValueError("projective seam unsigned-normal threshold is invalid")
        if not self.maximum_unsigned_normal_p90_deg <= self.maximum_oriented_normal_p90_deg <= 180:
            raise ValueError("projective seam oriented-normal threshold is invalid")
        if not 0 <= self.maximum_opposed_normal_fraction <= 1:
            raise ValueError("projective seam opposed-normal threshold is invalid")
        if self.maximum_selected_chart_self_reprojection_p90_px <= 0:
            raise ValueError("projective seam self-reprojection ceiling is invalid")
        return self

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class MaterialWeldConfig:
    """Evaluator-only gate for tangential drift of frozen M0 material pairs.

    This is deliberately not part of :class:`ProjectiveSeamConfig`: changing
    an aligned-arm audit must not mutate, invalidate, or reclassify the frozen
    source graph authority.
    """

    maximum_euclidean_distance_excess_p50_m: float = 0.10
    maximum_euclidean_distance_excess_p90_m: float = 0.30

    def validated(self) -> "MaterialWeldConfig":
        if not (
            0
            < self.maximum_euclidean_distance_excess_p50_m
            <= self.maximum_euclidean_distance_excess_p90_m
        ):
            raise ValueError("projective seam material-weld thresholds are invalid")
        return self

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _directed_projective_correspondences(
    *,
    source_chart: int,
    target_chart: int,
    direction: int,
    vertices: np.ndarray,
    association_normal: np.ndarray,
    target_surface_normal: np.ndarray,
    eligible: np.ndarray,
    vertex_offsets: np.ndarray,
    face_offsets: np.ndarray,
    faces: np.ndarray,
    camera_to_world: np.ndarray,
    focal_px: np.ndarray,
    principal_xy: np.ndarray,
    projected_vertex_uv: list[np.ndarray],
    projected_vertex_depth: list[np.ndarray],
    height: int,
    width: int,
    config: ProjectiveSeamConfig,
) -> dict[str, np.ndarray | int | float]:
    """Deterministically materialize one directed source-only seam row."""

    source_lo, source_hi = map(
        int, vertex_offsets[source_chart : source_chart + 2]
    )
    source_inventory = np.arange(source_lo, source_hi, dtype=np.int64)
    source_inventory = source_inventory[eligible[source_inventory]]
    target_face_lo, target_face_hi = map(
        int, face_offsets[target_chart : target_chart + 2]
    )
    target_face_inventory = np.arange(
        target_face_lo, target_face_hi, dtype=np.int64
    )
    projected_uv, source_depth = _project_world(
        vertices[source_inventory],
        camera_to_world[target_chart],
        focal_px[target_chart],
        principal_xy,
    )
    in_frame = (
        np.isfinite(projected_uv).all(1)
        & np.isfinite(source_depth)
        & (source_depth > 0)
        & (projected_uv[:, 0] >= 0)
        & (projected_uv[:, 0] <= width - 1)
        & (projected_uv[:, 1] >= 0)
        & (projected_uv[:, 1] <= height - 1)
    )
    candidate_rows = np.flatnonzero(in_frame)
    face = np.full(len(source_inventory), -1, np.int64)
    screen = np.zeros((len(source_inventory), 3), np.float64)
    material = np.zeros((len(source_inventory), 3), np.float64)
    target_depth = np.full(len(source_inventory), np.nan, np.float64)
    if len(candidate_rows):
        values = _chunked_frontmost_exact_face_at_rays(
            projected_uv[candidate_rows],
            target_face_inventory,
            faces,
            projected_vertex_uv[target_chart],
            projected_vertex_depth[target_chart],
        )
        (
            face[candidate_rows],
            screen[candidate_rows],
            material[candidate_rows],
            target_depth[candidate_rows],
        ) = values
    rows = np.flatnonzero(face >= 0)
    corners = faces[face[rows]]
    target_point = np.sum(vertices[corners] * material[rows, :, None], axis=1)
    target_normal = np.sum(
        target_surface_normal[corners] * material[rows, :, None], axis=1
    )
    target_corner_normal_length = np.linalg.norm(
        target_surface_normal[corners], axis=2
    )
    target_normal_length = np.linalg.norm(target_normal, axis=1)
    normal_valid = np.isfinite(target_normal_length) & (
        target_normal_length > 1e-10
    ) & np.all(
        np.isfinite(target_corner_normal_length)
        & (target_corner_normal_length > 0.5),
        axis=1,
    )
    target_normal /= np.maximum(target_normal_length[:, None], 1e-12)
    source_normal = association_normal[source_inventory[rows]]
    dot = np.sum(source_normal * target_normal, axis=1)
    angle = np.degrees(np.arccos(np.clip(np.abs(dot), -1.0, 1.0)))
    tolerance = np.maximum(
        config.absolute_depth_tolerance_m,
        config.relative_depth_tolerance
        * np.minimum(source_depth[rows], target_depth[rows]),
    )
    source_side = np.sum(
        (camera_to_world[source_chart, :3, 3] - target_point) * target_normal,
        axis=1,
    )
    target_side = np.sum(
        (camera_to_world[target_chart, :3, 3] - target_point) * target_normal,
        axis=1,
    )
    same_side = source_side * target_side
    accepted_local = (
        normal_valid
        & np.isfinite(target_depth[rows])
        & (target_depth[rows] > 0)
        & (np.abs(source_depth[rows] - target_depth[rows]) <= tolerance)
        & (angle <= config.maximum_reference_unsigned_normal_angle_deg)
        & np.isfinite(same_side)
        & (same_side > 0)
    )
    accepted_rows = rows[accepted_local]
    accepted_source = source_inventory[accepted_rows]
    accepted_face = face[accepted_rows]
    accepted_material = material[accepted_rows]
    accepted_target_point = np.sum(
        vertices[faces[accepted_face]] * accepted_material[..., None], axis=1
    )
    accepted_count = len(accepted_rows)
    return {
        "source_vertex_indices": accepted_source,
        "target_face_indices": accepted_face,
        "target_screen_barycentric": screen[accepted_rows],
        "target_barycentric": accepted_material,
        "correspondence_direction": np.full(accepted_count, direction, np.int8),
        "source_projected_uv": projected_uv[accepted_rows],
        "source_projected_depth_m": source_depth[accepted_rows],
        "target_reference_depth_m": target_depth[accepted_rows],
        "reference_euclidean_distance_m": np.linalg.norm(
            vertices[accepted_source] - accepted_target_point, axis=1
        ),
        "reference_unsigned_normal_angle_deg": angle[accepted_local],
        "reference_same_side_product": same_side[accepted_local],
        "eligible_count": len(source_inventory),
        "accepted_count": accepted_count,
        "supported_fraction": float(accepted_count / max(len(source_inventory), 1)),
    }


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability)) if len(values) else float("nan")


def _direction_geometry_metrics(
    vertices_world: np.ndarray,
    surface_normals_world: np.ndarray,
    faces: np.ndarray,
    source_vertex_indices: np.ndarray,
    target_face_indices: np.ndarray,
    target_barycentric: np.ndarray,
    config: ProjectiveSeamConfig,
) -> dict[str, object]:
    source = np.asarray(source_vertex_indices, np.int64)
    target_face = np.asarray(target_face_indices, np.int64)
    barycentric = np.asarray(target_barycentric, np.float64)
    if not len(source):
        return {
            "point_to_plane_source_p50_m": float("nan"),
            "point_to_plane_source_p90_m": float("nan"),
            "point_to_plane_target_p50_m": float("nan"),
            "point_to_plane_target_p90_m": float("nan"),
            "symmetric_max_point_to_plane_p50_m": float("nan"),
            "symmetric_max_point_to_plane_p90_m": float("nan"),
            "unsigned_normal_p90_deg": float("nan"),
            "oriented_normal_p90_deg": float("nan"),
            "opposed_normal_fraction": float("nan"),
            "geometry_valid": False,
        }
    corners = faces[target_face]
    target_point = np.sum(
        vertices_world[corners] * barycentric[..., None], axis=1
    )
    target_normal = np.sum(
        surface_normals_world[corners] * barycentric[..., None], axis=1
    )
    length = np.linalg.norm(target_normal, axis=1)
    if np.any(~np.isfinite(length)) or np.any(length <= 1e-10):
        raise ValueError("projective seam target surface normal is degenerate")
    target_normal /= length[:, None]
    source_normal = surface_normals_world[source]
    delta = vertices_world[source] - target_point
    source_plane = np.abs(np.sum(delta * source_normal, axis=1))
    target_plane = np.abs(np.sum(delta * target_normal, axis=1))
    symmetric_max = np.maximum(source_plane, target_plane)
    signed_dot = np.sum(source_normal * target_normal, axis=1)
    unsigned = np.degrees(np.arccos(np.clip(np.abs(signed_dot), -1.0, 1.0)))
    oriented = np.degrees(np.arccos(np.clip(signed_dot, -1.0, 1.0)))
    opposed = signed_dot < 0
    values = {
        "point_to_plane_source_p50_m": _quantile(source_plane, 0.5),
        "point_to_plane_source_p90_m": _quantile(source_plane, 0.9),
        "point_to_plane_target_p50_m": _quantile(target_plane, 0.5),
        "point_to_plane_target_p90_m": _quantile(target_plane, 0.9),
        "symmetric_max_point_to_plane_p50_m": _quantile(symmetric_max, 0.5),
        "symmetric_max_point_to_plane_p90_m": _quantile(symmetric_max, 0.9),
        "unsigned_normal_p90_deg": _quantile(unsigned, 0.9),
        "oriented_normal_p90_deg": _quantile(oriented, 0.9),
        "opposed_normal_fraction": float(np.mean(opposed)),
    }
    values["geometry_valid"] = bool(
        values["symmetric_max_point_to_plane_p50_m"]
        <= config.maximum_point_to_plane_p50_m
        and values["symmetric_max_point_to_plane_p90_m"]
        <= config.maximum_point_to_plane_p90_m
        and values["unsigned_normal_p90_deg"]
        <= config.maximum_unsigned_normal_p90_deg
        and values["oriented_normal_p90_deg"]
        <= config.maximum_oriented_normal_p90_deg
        and values["opposed_normal_fraction"]
        <= config.maximum_opposed_normal_fraction
    )
    return values


def _direction_material_weld_metrics(
    reference_vertices_world: np.ndarray,
    evaluated_vertices_world: np.ndarray,
    faces: np.ndarray,
    source_vertex_indices: np.ndarray,
    target_face_indices: np.ndarray,
    target_barycentric: np.ndarray,
    config: MaterialWeldConfig = MaterialWeldConfig(),
) -> dict[str, object]:
    """Measure only additional separation of a frozen M0 material pair.

    ``d0`` is the Euclidean separation already present in the source M0
    association and ``da`` is the separation after evaluating an aligned
    atlas arm.  The hard residual is ``max(0, da - d0)``.  Subtracting the
    frozen source separation avoids treating source sampling/association
    thickness as optimizer damage, while the Euclidean norm exposes
    tangential slides that point-to-plane residuals cannot see.
    """

    config = config.validated()
    reference = np.asarray(reference_vertices_world, np.float64)
    evaluated = np.asarray(evaluated_vertices_world, np.float64)
    faces = np.asarray(faces, np.int64)
    source = np.asarray(source_vertex_indices, np.int64)
    target_face = np.asarray(target_face_indices, np.int64)
    barycentric = np.asarray(target_barycentric, np.float64)
    if reference.ndim != 2 or reference.shape[1:] != (3,):
        raise ValueError("projective seam material-weld reference geometry is invalid")
    if evaluated.shape != reference.shape:
        raise ValueError("projective seam material-weld evaluated geometry differs")
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError("projective seam material-weld topology is invalid")
    if source.ndim != 1 or target_face.shape != source.shape:
        raise ValueError("projective seam material-weld correspondence vectors differ")
    if barycentric.shape != (len(source), 3):
        raise ValueError("projective seam material-weld barycentric inventory differs")
    if (
        not np.isfinite(reference).all()
        or not np.isfinite(evaluated).all()
        or not np.isfinite(barycentric).all()
    ):
        raise ValueError("projective seam material-weld geometry is nonfinite")
    if np.any((source < 0) | (source >= len(reference))):
        raise ValueError("projective seam material-weld source leaves the inventory")
    if np.any((target_face < 0) | (target_face >= len(faces))):
        raise ValueError("projective seam material-weld face leaves the inventory")
    if np.any((faces < 0) | (faces >= len(reference))):
        raise ValueError("projective seam material-weld topology leaves the inventory")
    if not len(source):
        return {
            "reference_material_distance_p50_m": float("nan"),
            "reference_material_distance_p90_m": float("nan"),
            "evaluated_material_distance_p50_m": float("nan"),
            "evaluated_material_distance_p90_m": float("nan"),
            "material_weld_excess_p50_m": float("nan"),
            "material_weld_excess_p90_m": float("nan"),
            "material_weld_excess_maximum_m": float("nan"),
            "material_pair_improved_fraction": float("nan"),
            "material_weld_valid": False,
        }

    corners = faces[target_face]
    reference_target = np.sum(
        reference[corners] * barycentric[..., None], axis=1
    )
    evaluated_target = np.sum(
        evaluated[corners] * barycentric[..., None], axis=1
    )
    reference_distance = np.linalg.norm(
        reference[source] - reference_target, axis=1
    )
    evaluated_distance = np.linalg.norm(
        evaluated[source] - evaluated_target, axis=1
    )
    excess = np.maximum(0.0, evaluated_distance - reference_distance)
    values = {
        "reference_material_distance_p50_m": _quantile(reference_distance, 0.5),
        "reference_material_distance_p90_m": _quantile(reference_distance, 0.9),
        "evaluated_material_distance_p50_m": _quantile(evaluated_distance, 0.5),
        "evaluated_material_distance_p90_m": _quantile(evaluated_distance, 0.9),
        "material_weld_excess_p50_m": _quantile(excess, 0.5),
        "material_weld_excess_p90_m": _quantile(excess, 0.9),
        "material_weld_excess_maximum_m": float(np.max(excess)),
        "material_pair_improved_fraction": float(
            np.mean(evaluated_distance < reference_distance)
        ),
    }
    values["material_weld_valid"] = bool(
        values["material_weld_excess_p50_m"]
        <= config.maximum_euclidean_distance_excess_p50_m
        and values["material_weld_excess_p90_m"]
        <= config.maximum_euclidean_distance_excess_p90_m
    )
    return values


@dataclass(frozen=True)
class ProjectiveExactFaceSeamAuthority:
    chart_names: np.ndarray
    common_valid: np.ndarray
    chart_vertex_offsets: np.ndarray
    sampled_vertex_pixel_indices: np.ndarray
    chart_face_offsets: np.ndarray
    faces: np.ndarray
    reference_points_world_dense: np.ndarray
    reference_vertices_world: np.ndarray
    reference_association_normals_world: np.ndarray
    reference_surface_normals_world: np.ndarray
    source_eligible_vertex_mask: np.ndarray
    camera_to_world: np.ndarray
    focal_px: np.ndarray
    principal_xy: np.ndarray
    self_reprojection_error_px: np.ndarray
    edge_chart_indices: np.ndarray
    edge_correspondence_offsets: np.ndarray
    source_vertex_indices: np.ndarray
    target_face_indices: np.ndarray
    target_screen_barycentric: np.ndarray
    target_barycentric: np.ndarray
    correspondence_direction: np.ndarray
    source_projected_uv: np.ndarray
    source_projected_depth_m: np.ndarray
    target_reference_depth_m: np.ndarray
    reference_euclidean_distance_m: np.ndarray
    reference_unsigned_normal_angle_deg: np.ndarray
    reference_same_side_product: np.ndarray
    source_eligible_count_by_direction: np.ndarray
    source_count_by_direction: np.ndarray
    source_supported_fraction_by_direction: np.ndarray
    edge_plan_symmetric_surface_overlap: np.ndarray
    edge_minimum_supported_fraction_required: np.ndarray
    direction_point_to_plane_source_p50_m: np.ndarray
    direction_point_to_plane_source_p90_m: np.ndarray
    direction_point_to_plane_target_p50_m: np.ndarray
    direction_point_to_plane_target_p90_m: np.ndarray
    direction_symmetric_max_point_to_plane_p50_m: np.ndarray
    direction_symmetric_max_point_to_plane_p90_m: np.ndarray
    direction_unsigned_normal_p90_deg: np.ndarray
    direction_oriented_normal_p90_deg: np.ndarray
    direction_opposed_normal_fraction: np.ndarray
    direction_support_valid: np.ndarray
    direction_geometry_valid: np.ndarray
    direction_formal_valid: np.ndarray
    edge_formal_valid: np.ndarray
    metadata: dict[str, object]

    @staticmethod
    def array_names() -> tuple[str, ...]:
        return (
            "chart_names",
            "common_valid",
            "chart_vertex_offsets",
            "sampled_vertex_pixel_indices",
            "chart_face_offsets",
            "faces",
            "reference_points_world_dense",
            "reference_vertices_world",
            "reference_association_normals_world",
            "reference_surface_normals_world",
            "source_eligible_vertex_mask",
            "camera_to_world",
            "focal_px",
            "principal_xy",
            "self_reprojection_error_px",
            "edge_chart_indices",
            "edge_correspondence_offsets",
            "source_vertex_indices",
            "target_face_indices",
            "target_screen_barycentric",
            "target_barycentric",
            "correspondence_direction",
            "source_projected_uv",
            "source_projected_depth_m",
            "target_reference_depth_m",
            "reference_euclidean_distance_m",
            "reference_unsigned_normal_angle_deg",
            "reference_same_side_product",
            "source_eligible_count_by_direction",
            "source_count_by_direction",
            "source_supported_fraction_by_direction",
            "edge_plan_symmetric_surface_overlap",
            "edge_minimum_supported_fraction_required",
            "direction_point_to_plane_source_p50_m",
            "direction_point_to_plane_source_p90_m",
            "direction_point_to_plane_target_p50_m",
            "direction_point_to_plane_target_p90_m",
            "direction_symmetric_max_point_to_plane_p50_m",
            "direction_symmetric_max_point_to_plane_p90_m",
            "direction_unsigned_normal_p90_deg",
            "direction_oriented_normal_p90_deg",
            "direction_opposed_normal_fraction",
            "direction_support_valid",
            "direction_geometry_valid",
            "direction_formal_valid",
            "edge_formal_valid",
        )

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(getattr(self, name)) for name in self.array_names()}

    def validated(self) -> "ProjectiveExactFaceSeamAuthority":
        arrays = self.arrays()
        names = arrays["chart_names"].astype(str)
        chart_count = len(names)
        if chart_count < 2 or len(set(names.tolist())) != chart_count:
            raise ValueError("projective seam chart inventory is invalid")
        valid = arrays["common_valid"].astype(bool)
        if valid.ndim != 3 or valid.shape[0] != chart_count:
            raise ValueError("projective seam common domain is invalid")
        height, width = valid.shape[1:]
        vertex_offsets = arrays["chart_vertex_offsets"].astype(np.int64)
        face_offsets = arrays["chart_face_offsets"].astype(np.int64)
        if vertex_offsets.shape != (chart_count + 1,) or face_offsets.shape != (chart_count + 1,):
            raise ValueError("projective seam topology offsets differ")
        if vertex_offsets[0] != 0 or face_offsets[0] != 0 or np.any(np.diff(vertex_offsets) <= 0) or np.any(np.diff(face_offsets) <= 0):
            raise ValueError("projective seam contains an empty topology unit")
        vertex_count = int(vertex_offsets[-1])
        face_count = int(face_offsets[-1])
        pixel = arrays["sampled_vertex_pixel_indices"].astype(np.int64)
        faces = arrays["faces"].astype(np.int64)
        if pixel.shape != (vertex_count,) or np.any((pixel < 0) | (pixel >= height * width)):
            raise ValueError("projective seam sampled pixels are invalid")
        if faces.shape != (face_count, 3) or np.any((faces < 0) | (faces >= vertex_count)):
            raise ValueError("projective seam faces are invalid")
        vertex_rows = _unit_rows(vertex_offsets)
        face_rows = _unit_rows(face_offsets)
        if np.any(vertex_rows[faces] != face_rows[:, None]):
            raise ValueError("projective seam face crosses a chart boundary")
        vertices = arrays["reference_vertices_world"].astype(np.float64)
        dense_points = arrays["reference_points_world_dense"].astype(np.float64)
        association_normal = arrays["reference_association_normals_world"].astype(np.float64)
        surface_normal = arrays["reference_surface_normals_world"].astype(np.float64)
        if dense_points.shape != (chart_count, height, width, 3) or vertices.shape != (vertex_count, 3) or association_normal.shape != (vertex_count, 3) or surface_normal.shape != (vertex_count, 3):
            raise ValueError("projective seam reference geometry differs")
        if not np.isfinite(dense_points).all() or not np.isfinite(vertices).all() or not np.isfinite(association_normal).all():
            raise ValueError("projective seam reference geometry is nonfinite")
        replayed_surface_normal = _surface_vertex_normals(vertices, faces)
        if not np.allclose(surface_normal, replayed_surface_normal, atol=1e-10, rtol=1e-10):
            raise ValueError("projective seam surface normals do not replay")
        eligible = arrays["source_eligible_vertex_mask"].astype(bool)
        if eligible.shape != (vertex_count,):
            raise ValueError("projective seam source eligibility differs")
        replayed_eligible = np.zeros(vertex_count, bool)
        replayed_vertices = np.zeros_like(vertices)
        replayed_association_normal = np.zeros_like(association_normal)
        for chart in range(chart_count):
            lo, hi = map(int, vertex_offsets[chart : chart + 2])
            stencil = _common_valid_normal_stencil(valid[chart])
            replayed_eligible[lo:hi] = stencil.ravel()[pixel[lo:hi]]
            replayed_vertices[lo:hi] = dense_points[chart].reshape(-1, 3)[pixel[lo:hi]]
            dense_normal = _dense_world_normals(dense_points[chart], stencil)
            replayed_association_normal[lo:hi] = dense_normal.reshape(-1, 3)[pixel[lo:hi]]
        if not np.array_equal(eligible, replayed_eligible):
            raise ValueError("projective seam source eligibility does not replay common mask")
        if not np.allclose(vertices, replayed_vertices, atol=1e-12, rtol=0.0):
            raise ValueError("projective seam packed reference vertices do not replay dense M0")
        if not np.allclose(association_normal, replayed_association_normal, atol=1e-12, rtol=0.0):
            raise ValueError("projective seam association normals do not replay dense M0")
        association_length = np.linalg.norm(association_normal, axis=1)
        if np.any(np.abs(association_length[eligible] - 1.0) > 1e-8):
            raise ValueError("projective seam eligible association normal is not unit")
        camera_to_world = arrays["camera_to_world"].astype(np.float64)
        focal = arrays["focal_px"].astype(np.float64)
        principal = arrays["principal_xy"].astype(np.float64)
        if camera_to_world.shape != (chart_count, 4, 4) or focal.shape != (chart_count,) or principal.shape != (2,):
            raise ValueError("projective seam camera arrays differ")
        if not np.isfinite(camera_to_world).all() or np.any(focal <= 0):
            raise ValueError("projective seam camera calibration is invalid")
        expected_principal = np.asarray([(width - 1) / 2.0, (height - 1) / 2.0])
        if not np.array_equal(principal, expected_principal):
            raise ValueError("projective seam principal point is not on pixel centers")
        self_error = arrays["self_reprojection_error_px"].astype(np.float64)
        if self_error.shape != (vertex_count,) or not np.isfinite(self_error).all():
            raise ValueError("projective seam self-reprojection inventory differs")
        for chart in range(chart_count):
            lo, hi = map(int, vertex_offsets[chart : chart + 2])
            projected, depth = _project_world(
                vertices[lo:hi], camera_to_world[chart], focal[chart], principal
            )
            nominal = np.column_stack((pixel[lo:hi] % width, pixel[lo:hi] // width))
            replay = np.linalg.norm(projected - nominal, axis=1)
            if np.any(depth <= 0) or not np.allclose(replay, self_error[lo:hi], atol=1e-10, rtol=1e-10):
                raise ValueError("projective seam self-reprojection does not replay")

        edges = arrays["edge_chart_indices"].astype(np.int64)
        edge_count = len(edges)
        if edges.shape != (edge_count, 2) or np.any(edges[:, 0] >= edges[:, 1]) or np.any((edges < 0) | (edges >= chart_count)):
            raise ValueError("projective seam edge inventory is invalid")
        if len(set(map(tuple, edges.tolist()))) != edge_count:
            raise ValueError("projective seam edge inventory is duplicated")
        offsets = arrays["edge_correspondence_offsets"].astype(np.int64)
        if offsets.shape != (edge_count + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
            raise ValueError("projective seam correspondence offsets are invalid")
        correspondence_count = int(offsets[-1])
        vector_names = (
            "source_vertex_indices",
            "target_face_indices",
            "correspondence_direction",
            "source_projected_depth_m",
            "target_reference_depth_m",
            "reference_euclidean_distance_m",
            "reference_unsigned_normal_angle_deg",
            "reference_same_side_product",
        )
        if any(arrays[name].shape != (correspondence_count,) for name in vector_names):
            raise ValueError("projective seam correspondence vectors differ")
        if arrays["target_screen_barycentric"].shape != (correspondence_count, 3) or arrays["target_barycentric"].shape != (correspondence_count, 3) or arrays["source_projected_uv"].shape != (correspondence_count, 2):
            raise ValueError("projective seam correspondence matrix differs")
        source = arrays["source_vertex_indices"].astype(np.int64)
        target_face = arrays["target_face_indices"].astype(np.int64)
        direction = arrays["correspondence_direction"].astype(np.int8)
        screen_barycentric = arrays["target_screen_barycentric"].astype(np.float64)
        barycentric = arrays["target_barycentric"].astype(np.float64)
        if np.any((source < 0) | (source >= vertex_count)) or np.any((target_face < 0) | (target_face >= face_count)) or np.any((direction < 0) | (direction > 1)):
            raise ValueError("projective seam correspondence leaves its inventory")
        for value, label in ((screen_barycentric, "screen"), (barycentric, "material")):
            if not np.isfinite(value).all() or np.any(value < -1e-9) or np.any(value > 1.0 + 1e-9) or not np.allclose(value.sum(1), 1.0, atol=1e-9, rtol=0.0):
                raise ValueError(f"projective seam {label} barycentric is invalid")
        config_payload = self.metadata.get("config")
        if not isinstance(config_payload, dict):
            raise ValueError("projective seam authority lacks frozen config")
        config = ProjectiveSeamConfig(**config_payload).validated()
        if self.metadata.get("config_content_sha256") != canonical_json_sha256(config.to_dict()):
            raise ValueError("projective seam config hash differs")

        shape_edge_direction = (edge_count, 2)
        matrix_names = (
            "source_eligible_count_by_direction",
            "source_count_by_direction",
            "source_supported_fraction_by_direction",
            "direction_point_to_plane_source_p50_m",
            "direction_point_to_plane_source_p90_m",
            "direction_point_to_plane_target_p50_m",
            "direction_point_to_plane_target_p90_m",
            "direction_symmetric_max_point_to_plane_p50_m",
            "direction_symmetric_max_point_to_plane_p90_m",
            "direction_unsigned_normal_p90_deg",
            "direction_oriented_normal_p90_deg",
            "direction_opposed_normal_fraction",
            "direction_support_valid",
            "direction_geometry_valid",
            "direction_formal_valid",
        )
        if any(arrays[name].shape != shape_edge_direction for name in matrix_names):
            raise ValueError("projective seam directional audit shape differs")
        if arrays["edge_plan_symmetric_surface_overlap"].shape != (edge_count,) or arrays["edge_minimum_supported_fraction_required"].shape != (edge_count,) or arrays["edge_formal_valid"].shape != (edge_count,):
            raise ValueError("projective seam edge audit shape differs")
        plan_overlap = arrays["edge_plan_symmetric_surface_overlap"].astype(np.float64)
        if np.any(~np.isfinite(plan_overlap)) or np.any((plan_overlap < 0) | (plan_overlap > 1)):
            raise ValueError("projective seam plan overlap is invalid")
        replay_required = np.maximum(
            config.minimum_supported_fraction,
            config.frozen_overlap_support_fraction_multiplier * plan_overlap,
        )
        if not np.allclose(
            replay_required,
            arrays["edge_minimum_supported_fraction_required"],
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError("projective seam required support does not replay plan overlap")

        replay_count = np.zeros(shape_edge_direction, np.int64)
        replay_eligible = np.zeros(shape_edge_direction, np.int64)
        replay_geometry = np.zeros(shape_edge_direction, bool)
        geometry_arrays: dict[str, np.ndarray] = {
            name: np.full(shape_edge_direction, np.nan, np.float64)
            for name in (
                "direction_point_to_plane_source_p50_m",
                "direction_point_to_plane_source_p90_m",
                "direction_point_to_plane_target_p50_m",
                "direction_point_to_plane_target_p90_m",
                "direction_symmetric_max_point_to_plane_p50_m",
                "direction_symmetric_max_point_to_plane_p90_m",
                "direction_unsigned_normal_p90_deg",
                "direction_oriented_normal_p90_deg",
                "direction_opposed_normal_fraction",
            )
        }
        projected_vertex_uv: list[np.ndarray] = []
        projected_vertex_depth: list[np.ndarray] = []
        for chart in range(chart_count):
            projected, depth = _project_world(
                vertices, camera_to_world[chart], focal[chart], principal
            )
            projected_vertex_uv.append(projected)
            projected_vertex_depth.append(depth)
        correspondence_replay_names = (
            "source_vertex_indices",
            "target_face_indices",
            "target_screen_barycentric",
            "target_barycentric",
            "correspondence_direction",
            "source_projected_uv",
            "source_projected_depth_m",
            "target_reference_depth_m",
            "reference_euclidean_distance_m",
            "reference_unsigned_normal_angle_deg",
            "reference_same_side_product",
        )
        for edge, (first, second) in enumerate(edges):
            lo, hi = map(int, offsets[edge : edge + 2])
            edge_source = source[lo:hi]
            edge_face = target_face[lo:hi]
            edge_direction = direction[lo:hi]
            source_chart = vertex_rows[edge_source]
            target_chart = face_rows[edge_face]
            expected_direction = np.where(
                (source_chart == first) & (target_chart == second), 0,
                np.where((source_chart == second) & (target_chart == first), 1, -1),
            )
            if np.any(expected_direction < 0) or not np.array_equal(expected_direction.astype(np.int8), edge_direction):
                raise ValueError("projective seam direction does not replay its edge")
            if not eligible[edge_source].all():
                raise ValueError("projective seam correspondence uses ineligible source")
            for row, source_chart_row in enumerate((first, second)):
                replay_count[edge, row] = int(np.sum(edge_direction == row))
                vlo, vhi = map(int, vertex_offsets[source_chart_row : source_chart_row + 2])
                replay_eligible[edge, row] = int(np.sum(eligible[vlo:vhi]))
                take = np.flatnonzero(edge_direction == row) + lo
                target_chart_row = second if row == 0 else first
                replay_correspondence = _directed_projective_correspondences(
                    source_chart=int(source_chart_row),
                    target_chart=int(target_chart_row),
                    direction=row,
                    vertices=vertices,
                    association_normal=association_normal,
                    target_surface_normal=surface_normal,
                    eligible=eligible,
                    vertex_offsets=vertex_offsets,
                    face_offsets=face_offsets,
                    faces=faces,
                    camera_to_world=camera_to_world,
                    focal_px=focal,
                    principal_xy=principal,
                    projected_vertex_uv=projected_vertex_uv,
                    projected_vertex_depth=projected_vertex_depth,
                    height=height,
                    width=width,
                    config=config,
                )
                if replay_correspondence["eligible_count"] != replay_eligible[edge, row]:
                    raise ValueError("projective seam directed eligible count does not replay")
                if replay_correspondence["accepted_count"] != len(take):
                    raise ValueError("projective seam directed accepted count does not replay")
                for array_name in correspondence_replay_names:
                    expected_value = np.asarray(replay_correspondence[array_name])
                    observed_value = arrays[array_name][take]
                    if expected_value.dtype.kind in "iu" or observed_value.dtype.kind in "iu":
                        equal = np.array_equal(expected_value, observed_value)
                    else:
                        equal = np.allclose(
                            expected_value,
                            observed_value,
                            atol=1e-10,
                            rtol=1e-10,
                            equal_nan=True,
                        )
                    if not equal:
                        raise ValueError(
                            f"projective seam directed {array_name} does not replay"
                        )
                metrics = _direction_geometry_metrics(
                    vertices,
                    surface_normal,
                    faces,
                    source[take],
                    target_face[take],
                    barycentric[take],
                    config,
                )
                replay_geometry[edge, row] = bool(metrics["geometry_valid"])
                mapping = {
                    "direction_point_to_plane_source_p50_m": "point_to_plane_source_p50_m",
                    "direction_point_to_plane_source_p90_m": "point_to_plane_source_p90_m",
                    "direction_point_to_plane_target_p50_m": "point_to_plane_target_p50_m",
                    "direction_point_to_plane_target_p90_m": "point_to_plane_target_p90_m",
                    "direction_symmetric_max_point_to_plane_p50_m": "symmetric_max_point_to_plane_p50_m",
                    "direction_symmetric_max_point_to_plane_p90_m": "symmetric_max_point_to_plane_p90_m",
                    "direction_unsigned_normal_p90_deg": "unsigned_normal_p90_deg",
                    "direction_oriented_normal_p90_deg": "oriented_normal_p90_deg",
                    "direction_opposed_normal_fraction": "opposed_normal_fraction",
                }
                for array_name, metric_name in mapping.items():
                    geometry_arrays[array_name][edge, row] = float(metrics[metric_name])
        if not np.array_equal(replay_count, arrays["source_count_by_direction"]):
            raise ValueError("projective seam directional counts do not replay")
        if not np.array_equal(replay_eligible, arrays["source_eligible_count_by_direction"]):
            raise ValueError("projective seam eligible counts do not replay")
        replay_support = np.divide(
            replay_count,
            replay_eligible,
            out=np.zeros_like(replay_count, dtype=np.float64),
            where=replay_eligible > 0,
        )
        if not np.allclose(replay_support, arrays["source_supported_fraction_by_direction"], atol=1e-12, rtol=0.0):
            raise ValueError("projective seam directional support does not replay")
        for name, replay in geometry_arrays.items():
            if not np.allclose(replay, arrays[name], atol=1e-10, rtol=1e-10, equal_nan=True):
                raise ValueError(f"projective seam {name} does not replay")
        if not np.array_equal(replay_geometry, arrays["direction_geometry_valid"]):
            raise ValueError("projective seam geometry decisions do not replay")
        required = arrays["edge_minimum_supported_fraction_required"][:, None]
        replay_support_valid = (replay_count >= config.minimum_correspondences_per_direction) & (replay_support >= required)
        if not np.array_equal(replay_support_valid, arrays["direction_support_valid"]):
            raise ValueError("projective seam support decisions do not replay")
        replay_formal = replay_support_valid & replay_geometry
        if not np.array_equal(replay_formal, arrays["direction_formal_valid"]):
            raise ValueError("projective seam directional decisions do not replay")
        replay_edge = np.all(replay_formal, axis=1)
        if not np.array_equal(replay_edge, arrays["edge_formal_valid"]):
            raise ValueError("projective seam edge decisions are not directional AND")

        if self.metadata.get("artifact_type") != AUTHORITY_SCHEMA or self.metadata.get("authority_semantics_version") != AUTHORITY_SEMANTICS_VERSION:
            raise ValueError("wrong projective seam authority schema")
        if self.metadata.get("edge_formal_valid_definition") != EDGE_FORMAL_VALID_DEFINITION:
            raise ValueError("projective seam edge decision semantics differ")
        if self.metadata.get("uses_query_or_ground_truth") is not False or self.metadata.get("held_geometry_consumed") is not False or self.metadata.get("aligned_arm_geometry_consumed") is not False:
            raise ValueError("projective seam authority is not source-only")
        if self.metadata.get("principal_point_semantics") != "((width-1)/2,(height-1)/2)":
            raise ValueError("projective seam authority uses the wrong pixel phase")
        if self.metadata.get("target_face_screen_domain") != "actual_projection_of_M0_target_face_vertices_under_pinned_target_camera":
            raise ValueError("projective seam authority did not use actual target screens")
        if self.metadata.get("target_3d_interpolation") != "perspective_correct_alpha=(screen_beta/z)/sum(screen_beta/z)":
            raise ValueError("projective seam authority did not use perspective correction")
        if self.metadata.get("nominal_grid_uv_not_used_for_face_containment") is not True:
            raise ValueError("projective seam authority used nominal UV for containment")
        if self.metadata.get("visibility_rule") != "frontmost_positive_depth_exact_target_face_then_source_target_depth_consistency":
            raise ValueError("projective seam visibility rule differs")
        if self.metadata.get("edge_formal_valid_sha256") != arrays_sha256({"edge_formal_valid": arrays["edge_formal_valid"]}):
            raise ValueError("projective seam edge-formal-valid hash differs")
        per_chart_self = []
        for chart in range(chart_count):
            lo, hi = map(int, vertex_offsets[chart : chart + 2])
            per_chart_self.append(float(np.quantile(self_error[lo:hi], 0.9)))
        if any(
            value > config.maximum_selected_chart_self_reprojection_p90_px
            for value in per_chart_self
        ):
            raise ValueError("projective seam selected chart exceeds self-reprojection ceiling")
        if self.metadata.get("selected_chart_self_reprojection_all_p90_pass") is not True:
            raise ValueError("projective seam self-reprojection decision differs")
        return self

    def save_npz(self, path: Path) -> dict[str, object]:
        arrays = self.validated().arrays()
        metadata = dict(self.metadata)
        metadata["arrays_sha256"] = arrays_sha256(arrays)
        metadata["content_sha256"] = canonical_json_sha256(metadata)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".temporary.npz")
        np.savez_compressed(
            temporary,
            **arrays,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        temporary.replace(path)
        return metadata

    @classmethod
    def load_npz(cls, path: Path) -> "ProjectiveExactFaceSeamAuthority":
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            arrays = {name: np.asarray(data[name]) for name in cls.array_names()}
        if metadata.get("arrays_sha256") != arrays_sha256(arrays):
            raise ValueError("projective seam authority arrays differ from lineage")
        _replay_metadata(metadata, "projective seam authority")
        return cls(metadata=metadata, **arrays).validated()


def freeze_projective_exact_face_correspondences(
    *,
    chart_names: np.ndarray,
    common_valid: np.ndarray,
    chart_vertex_offsets: np.ndarray,
    sampled_vertex_pixel_indices: np.ndarray,
    chart_face_offsets: np.ndarray,
    faces: np.ndarray,
    reference_points_world: np.ndarray,
    reference_dense_normals_world: np.ndarray,
    camera_to_world: np.ndarray,
    focal_px: np.ndarray,
    plan_chart_names: np.ndarray,
    coverage_edges: np.ndarray,
    symmetric_surface_overlap: np.ndarray,
    config: ProjectiveSeamConfig = ProjectiveSeamConfig(),
    metadata: Mapping[str, object] | None = None,
) -> ProjectiveExactFaceSeamAuthority:
    """Freeze source-ray correspondences without a world-nearest query."""

    config = config.validated()
    chart_names = np.asarray(chart_names).astype(str)
    common_valid = np.asarray(common_valid, bool)
    vertex_offsets = np.asarray(chart_vertex_offsets, np.int64)
    pixel = np.asarray(sampled_vertex_pixel_indices, np.int64)
    face_offsets = np.asarray(chart_face_offsets, np.int64)
    faces = np.asarray(faces, np.int64)
    reference_points_world = np.asarray(reference_points_world, np.float64)
    reference_dense_normals_world = np.asarray(
        reference_dense_normals_world, np.float64
    )
    camera_to_world = np.asarray(camera_to_world, np.float64)
    focal_px = np.asarray(focal_px, np.float64)
    plan_chart_names = np.asarray(plan_chart_names).astype(str)
    chart_count, height, width = common_valid.shape
    if chart_names.shape != (chart_count,) or vertex_offsets.shape != (chart_count + 1,) or face_offsets.shape != (chart_count + 1,):
        raise ValueError("projective seam topology and chart inventory differ")
    if reference_points_world.shape != (chart_count, height, width, 3) or reference_dense_normals_world.shape != reference_points_world.shape:
        raise ValueError("projective seam dense reference geometry differs")
    if camera_to_world.shape != (chart_count, 4, 4) or focal_px.shape != (chart_count,):
        raise ValueError("projective seam camera inventory differs")
    if len(set(plan_chart_names.tolist())) != len(plan_chart_names):
        raise ValueError("projective seam plan inventory is duplicated")
    plan_rows = {name: row for row, name in enumerate(plan_chart_names.tolist())}
    if any(name not in plan_rows for name in chart_names.tolist()):
        raise ValueError("projective seam chart is absent from the plan")
    selected = [plan_rows[name] for name in chart_names.tolist()]
    coverage_edges = np.asarray(coverage_edges, bool)
    symmetric_surface_overlap = np.asarray(symmetric_surface_overlap, np.float64)
    expected_pairwise = (len(plan_chart_names), len(plan_chart_names))
    if coverage_edges.shape != expected_pairwise or symmetric_surface_overlap.shape != expected_pairwise:
        raise ValueError("projective seam plan pairwise arrays differ")
    selected_coverage = coverage_edges[np.ix_(selected, selected)]
    selected_overlap = symmetric_surface_overlap[np.ix_(selected, selected)]
    principal = np.asarray([(width - 1) / 2.0, (height - 1) / 2.0])

    vertices = np.concatenate(
        [
            reference_points_world[chart].reshape(-1, 3)[
                pixel[vertex_offsets[chart] : vertex_offsets[chart + 1]]
            ]
            for chart in range(chart_count)
        ]
    )
    association_normal = np.concatenate(
        [
            reference_dense_normals_world[chart].reshape(-1, 3)[
                pixel[vertex_offsets[chart] : vertex_offsets[chart + 1]]
            ]
            for chart in range(chart_count)
        ]
    )
    surface_normal = _surface_vertex_normals(vertices, faces)
    eligible = np.zeros(len(vertices), bool)
    self_error = np.zeros(len(vertices), np.float64)
    projected_vertex_uv: list[np.ndarray] = []
    projected_vertex_depth: list[np.ndarray] = []
    self_reprojection_rows: list[dict[str, object]] = []
    for chart in range(chart_count):
        lo, hi = map(int, vertex_offsets[chart : chart + 2])
        stencil = _common_valid_normal_stencil(common_valid[chart])
        eligible[lo:hi] = stencil.ravel()[pixel[lo:hi]]
        projected, depth = _project_world(
            vertices, camera_to_world[chart], focal_px[chart], principal
        )
        projected_vertex_uv.append(projected)
        projected_vertex_depth.append(depth)
        nominal = np.column_stack((pixel[lo:hi] % width, pixel[lo:hi] // width))
        error = np.linalg.norm(projected[lo:hi] - nominal, axis=1)
        self_error[lo:hi] = error
        self_reprojection_rows.append(
            {
                "chart_name": str(chart_names[chart]),
                "vertex_count": int(hi - lo),
                "p50_px": float(np.quantile(error, 0.5)),
                "p90_px": float(np.quantile(error, 0.9)),
                "maximum_px": float(np.max(error)),
                "p90_pass": bool(
                    np.quantile(error, 0.9)
                    <= config.maximum_selected_chart_self_reprojection_p90_px
                ),
            }
        )
    projected_vertex_uv = list(projected_vertex_uv)
    projected_vertex_depth = list(projected_vertex_depth)
    if np.any(np.linalg.norm(association_normal[eligible], axis=1) < 0.5):
        raise ValueError("projective seam eligible source lacks a dense normal")

    edge_rows: list[tuple[int, int]] = []
    edge_offsets = [0]
    source_values: list[np.ndarray] = []
    face_values: list[np.ndarray] = []
    screen_values: list[np.ndarray] = []
    material_values: list[np.ndarray] = []
    direction_values: list[np.ndarray] = []
    projected_uv_values: list[np.ndarray] = []
    source_depth_values: list[np.ndarray] = []
    target_depth_values: list[np.ndarray] = []
    distance_values: list[np.ndarray] = []
    angle_values: list[np.ndarray] = []
    same_side_values: list[np.ndarray] = []
    eligible_counts: list[tuple[int, int]] = []
    accepted_counts: list[tuple[int, int]] = []
    support_values: list[tuple[float, float]] = []
    required_values: list[float] = []
    for first in range(chart_count):
        for second in range(first + 1, chart_count):
            if not bool(selected_coverage[first, second]):
                continue
            edge_source: list[np.ndarray] = []
            edge_face: list[np.ndarray] = []
            edge_screen: list[np.ndarray] = []
            edge_material: list[np.ndarray] = []
            edge_direction: list[np.ndarray] = []
            edge_projected_uv: list[np.ndarray] = []
            edge_source_depth: list[np.ndarray] = []
            edge_target_depth: list[np.ndarray] = []
            edge_distance: list[np.ndarray] = []
            edge_angle: list[np.ndarray] = []
            edge_same_side: list[np.ndarray] = []
            direction_eligible_count: list[int] = []
            direction_accepted_count: list[int] = []
            direction_support: list[float] = []
            for direction, (source_chart, target_chart) in enumerate(
                ((first, second), (second, first))
            ):
                result = _directed_projective_correspondences(
                    source_chart=source_chart,
                    target_chart=target_chart,
                    direction=direction,
                    vertices=vertices,
                    association_normal=association_normal,
                    target_surface_normal=surface_normal,
                    eligible=eligible,
                    vertex_offsets=vertex_offsets,
                    face_offsets=face_offsets,
                    faces=faces,
                    camera_to_world=camera_to_world,
                    focal_px=focal_px,
                    principal_xy=principal,
                    projected_vertex_uv=projected_vertex_uv,
                    projected_vertex_depth=projected_vertex_depth,
                    height=height,
                    width=width,
                    config=config,
                )
                edge_source.append(np.asarray(result["source_vertex_indices"]))
                edge_face.append(np.asarray(result["target_face_indices"]))
                edge_screen.append(
                    np.asarray(result["target_screen_barycentric"])
                )
                edge_material.append(np.asarray(result["target_barycentric"]))
                edge_direction.append(
                    np.asarray(result["correspondence_direction"])
                )
                edge_projected_uv.append(np.asarray(result["source_projected_uv"]))
                edge_source_depth.append(
                    np.asarray(result["source_projected_depth_m"])
                )
                edge_target_depth.append(
                    np.asarray(result["target_reference_depth_m"])
                )
                edge_distance.append(
                    np.asarray(result["reference_euclidean_distance_m"])
                )
                edge_angle.append(
                    np.asarray(result["reference_unsigned_normal_angle_deg"])
                )
                edge_same_side.append(
                    np.asarray(result["reference_same_side_product"])
                )
                direction_eligible_count.append(int(result["eligible_count"]))
                direction_accepted_count.append(int(result["accepted_count"]))
                direction_support.append(float(result["supported_fraction"]))
            joined_source = np.concatenate(edge_source)
            edge_rows.append((first, second))
            source_values.append(joined_source)
            face_values.append(np.concatenate(edge_face))
            screen_values.append(np.concatenate(edge_screen))
            material_values.append(np.concatenate(edge_material))
            direction_values.append(np.concatenate(edge_direction))
            projected_uv_values.append(np.concatenate(edge_projected_uv))
            source_depth_values.append(np.concatenate(edge_source_depth))
            target_depth_values.append(np.concatenate(edge_target_depth))
            distance_values.append(np.concatenate(edge_distance))
            angle_values.append(np.concatenate(edge_angle))
            same_side_values.append(np.concatenate(edge_same_side))
            edge_offsets.append(edge_offsets[-1] + len(joined_source))
            eligible_counts.append(tuple(direction_eligible_count))
            accepted_counts.append(tuple(direction_accepted_count))
            support_values.append(tuple(direction_support))
            required_values.append(
                max(
                    config.minimum_supported_fraction,
                    config.frozen_overlap_support_fraction_multiplier
                    * float(selected_overlap[first, second]),
                )
            )
    if not edge_rows:
        raise ValueError("projective seam selected submap has no coverage edge")

    source_all = np.concatenate(source_values).astype(np.int64, copy=False)
    face_all = np.concatenate(face_values).astype(np.int64, copy=False)
    material_all = np.concatenate(material_values).astype(np.float64, copy=False)
    direction_all = np.concatenate(direction_values).astype(np.int8, copy=False)
    edge_array = np.asarray(edge_rows, np.int32)
    offsets_array = np.asarray(edge_offsets, np.int64)
    eligible_count_array = np.asarray(eligible_counts, np.int64)
    count_array = np.asarray(accepted_counts, np.int64)
    support_array = np.asarray(support_values, np.float64)
    required_array = np.asarray(required_values, np.float64)
    edge_count = len(edge_array)
    metric_names = (
        "direction_point_to_plane_source_p50_m",
        "direction_point_to_plane_source_p90_m",
        "direction_point_to_plane_target_p50_m",
        "direction_point_to_plane_target_p90_m",
        "direction_symmetric_max_point_to_plane_p50_m",
        "direction_symmetric_max_point_to_plane_p90_m",
        "direction_unsigned_normal_p90_deg",
        "direction_oriented_normal_p90_deg",
        "direction_opposed_normal_fraction",
    )
    metric_arrays = {
        name: np.full((edge_count, 2), np.nan, np.float64)
        for name in metric_names
    }
    direction_geometry_valid = np.zeros((edge_count, 2), bool)
    mapping = {
        "direction_point_to_plane_source_p50_m": "point_to_plane_source_p50_m",
        "direction_point_to_plane_source_p90_m": "point_to_plane_source_p90_m",
        "direction_point_to_plane_target_p50_m": "point_to_plane_target_p50_m",
        "direction_point_to_plane_target_p90_m": "point_to_plane_target_p90_m",
        "direction_symmetric_max_point_to_plane_p50_m": "symmetric_max_point_to_plane_p50_m",
        "direction_symmetric_max_point_to_plane_p90_m": "symmetric_max_point_to_plane_p90_m",
        "direction_unsigned_normal_p90_deg": "unsigned_normal_p90_deg",
        "direction_oriented_normal_p90_deg": "oriented_normal_p90_deg",
        "direction_opposed_normal_fraction": "opposed_normal_fraction",
    }
    for edge in range(edge_count):
        lo, hi = map(int, offsets_array[edge : edge + 2])
        for direction in range(2):
            take = np.flatnonzero(direction_all[lo:hi] == direction) + lo
            metrics = _direction_geometry_metrics(
                vertices,
                surface_normal,
                faces,
                source_all[take],
                face_all[take],
                material_all[take],
                config,
            )
            direction_geometry_valid[edge, direction] = bool(
                metrics["geometry_valid"]
            )
            for array_name, metric_name in mapping.items():
                metric_arrays[array_name][edge, direction] = float(
                    metrics[metric_name]
                )
    direction_support_valid = (
        count_array >= config.minimum_correspondences_per_direction
    ) & (support_array >= required_array[:, None])
    direction_formal_valid = direction_support_valid & direction_geometry_valid
    edge_formal_valid = np.all(direction_formal_valid, axis=1)
    # Keep the summary explicit rather than hiding the per-chart maximum in a
    # single aggregate statistic.
    self_reprojection_pass = bool(
        all(bool(row["p90_pass"]) for row in self_reprojection_rows)
    )
    if not self_reprojection_pass:
        raise ValueError(
            "projective seam selected chart exceeds the frozen self-reprojection ceiling"
        )
    authority_metadata = {
        **dict(metadata or {}),
        "artifact_type": AUTHORITY_SCHEMA,
        "authority_semantics_version": AUTHORITY_SEMANTICS_VERSION,
        "uses_mapping_source_geometry": True,
        "uses_query_or_ground_truth": False,
        "held_geometry_consumed": False,
        "held_root_opened": False,
        "aligned_arm_geometry_consumed": False,
        "paired_initializer_geometry_encoded_upstream": bool(
            (metadata or {}).get("paired_initializer_geometry_encoded_upstream", True)
        ),
        "model_neutral_topology_claimed": False,
        "association_source": (
            "isolated_source_MASt3R_M0_on_exact_stride"
            f"{config.topology_stride}_topology"
        ),
        "edge_source": "frozen_chart_submap_plan_coverage_edges_only",
        "correspondence_directionality": "two_independent_directed_rows_per_undirected_edge",
        "edge_formal_valid_definition": EDGE_FORMAL_VALID_DEFINITION,
        "world_closest_surface_query_used": False,
        "principal_point_semantics": "((width-1)/2,(height-1)/2)",
        "target_face_screen_domain": "actual_projection_of_M0_target_face_vertices_under_pinned_target_camera",
        "target_3d_interpolation": "perspective_correct_alpha=(screen_beta/z)/sum(screen_beta/z)",
        "nominal_grid_uv_not_used_for_face_containment": True,
        "visibility_rule": "frontmost_positive_depth_exact_target_face_then_source_target_depth_consistency",
        "target_common_domain_rule": "target_exact_v3_face_all_corners_in_frozen_common_domain",
        "source_eligible_denominator": (
            "sealed_common_valid_five_pixel_central_difference_stencil_at_"
            "packed_vertices"
        ),
        "reference_normal_for_association": "source_MASt3R_dense_central_difference_normal_to_target_exact_topology_surface_normal",
        "surface_normal_for_geometry_gate": "exact_topology_area_weighted_vertex_normal",
        "hard_point_to_plane_residual": "max(abs(delta_dot_source_normal),abs(delta_dot_target_normal))",
        "direction_quantiles_pooled": False,
        "frontmost_projection_chunk_max_ray_face_pairs": (
            MAX_PROJECTIVE_RAY_FACE_PAIRS_PER_CHUNK
        ),
        "frontmost_projection_chunking_semantics": (
            "independent_source_rays_scattered_to_original_order"
        ),
        "config": config.to_dict(),
        "config_content_sha256": canonical_json_sha256(config.to_dict()),
        "chart_count": chart_count,
        "frozen_edge_count": edge_count,
        "frozen_correspondence_count": int(offsets_array[-1]),
        "edge_with_any_correspondence_count": int(np.sum(np.sum(count_array, axis=1) > 0)),
        "edge_with_both_directions_count": int(np.sum(np.all(count_array > 0, axis=1))),
        "direction_formal_valid_count": int(np.sum(direction_formal_valid)),
        "edge_formal_valid_count": int(np.sum(edge_formal_valid)),
        "edge_formal_valid_sha256": arrays_sha256(
            {"edge_formal_valid": edge_formal_valid}
        ),
        "selected_chart_self_reprojection": self_reprojection_rows,
        "selected_chart_self_reprojection_all_p90_pass": self_reprojection_pass,
        "selected_chart_self_reprojection_global_p50_px": float(
            np.quantile(self_error, 0.5)
        ),
        "selected_chart_self_reprojection_global_p90_px": float(
            np.quantile(self_error, 0.9)
        ),
        "selected_chart_self_reprojection_global_maximum_px": float(
            np.max(self_error)
        ),
        "projective_source_seam_core_file_sha256": file_sha256(Path(__file__)),
    }
    return ProjectiveExactFaceSeamAuthority(
        chart_names=chart_names,
        common_valid=common_valid,
        chart_vertex_offsets=vertex_offsets,
        sampled_vertex_pixel_indices=pixel,
        chart_face_offsets=face_offsets,
        faces=faces,
        reference_points_world_dense=reference_points_world,
        reference_vertices_world=vertices,
        reference_association_normals_world=association_normal,
        reference_surface_normals_world=surface_normal,
        source_eligible_vertex_mask=eligible,
        camera_to_world=camera_to_world,
        focal_px=focal_px,
        principal_xy=principal,
        self_reprojection_error_px=self_error,
        edge_chart_indices=edge_array,
        edge_correspondence_offsets=offsets_array,
        source_vertex_indices=source_all,
        target_face_indices=face_all,
        target_screen_barycentric=np.concatenate(screen_values).astype(
            np.float64, copy=False
        ),
        target_barycentric=material_all,
        correspondence_direction=direction_all,
        source_projected_uv=np.concatenate(projected_uv_values).astype(
            np.float64, copy=False
        ),
        source_projected_depth_m=np.concatenate(source_depth_values).astype(
            np.float64, copy=False
        ),
        target_reference_depth_m=np.concatenate(target_depth_values).astype(
            np.float64, copy=False
        ),
        reference_euclidean_distance_m=np.concatenate(distance_values).astype(
            np.float64, copy=False
        ),
        reference_unsigned_normal_angle_deg=np.concatenate(angle_values).astype(
            np.float64, copy=False
        ),
        reference_same_side_product=np.concatenate(same_side_values).astype(
            np.float64, copy=False
        ),
        source_eligible_count_by_direction=eligible_count_array,
        source_count_by_direction=count_array,
        source_supported_fraction_by_direction=support_array,
        edge_plan_symmetric_surface_overlap=np.asarray(
            [selected_overlap[first, second] for first, second in edge_rows],
            np.float64,
        ),
        edge_minimum_supported_fraction_required=required_array,
        direction_support_valid=direction_support_valid,
        direction_geometry_valid=direction_geometry_valid,
        direction_formal_valid=direction_formal_valid,
        edge_formal_valid=edge_formal_valid,
        metadata=authority_metadata,
        **metric_arrays,
    ).validated()


def _load_source_reference_dense(
    path: Path, *, output_height: int, output_width: int
) -> tuple[np.ndarray, tuple[int, int]]:
    payload = json.loads(Path(path).read_text())
    confidence = np.asarray(payload.get("confs"), np.float64)
    points = np.asarray(payload.get("points"), np.float64)
    if confidence.ndim != 2 or points.size != confidence.size * 3:
        raise ValueError("projective seam source pointmap has an invalid shape")
    points = points.reshape(*confidence.shape, 3)
    resized = cv2.resize(
        points,
        (output_width, output_height),
        interpolation=cv2.INTER_AREA,
    )
    if resized.shape != (output_height, output_width, 3) or not np.isfinite(resized).all():
        raise ValueError("projective seam source pointmap resize is invalid")
    return np.asarray(resized, np.float64), tuple(map(int, confidence.shape))


def build_projective_exact_face_seam_authority(
    comparison_domain_v3_path: Path,
    *,
    expected_comparison_domain_content_sha256: str,
    frozen_submap_plan_path: Path,
    expected_plan_content_sha256: str,
    disjoint_upstream_authority_path: Path,
    expected_disjoint_authority_content_sha256: str,
    source_root: Path,
    expected_source_tree_sha256: str,
    config: ProjectiveSeamConfig = ProjectiveSeamConfig(),
) -> ProjectiveExactFaceSeamAuthority:
    """Build a hash-pinned source-only projective authority."""

    if config.validated().topology_stride != 4:
        raise ValueError(
            "the v3 production-candidate builder is stride-4 only; use the "
            "dedicated paired stride-2 diagnostic builder for stride 2"
        )
    pins = (
        expected_comparison_domain_content_sha256,
        expected_plan_content_sha256,
        expected_disjoint_authority_content_sha256,
        expected_source_tree_sha256,
    )
    if any(not _is_sha256(value) for value in pins):
        raise ValueError("projective seam builder requires four explicit SHA-256 pins")
    comparison_domain_v3_path = Path(comparison_domain_v3_path)
    frozen_submap_plan_path = Path(frozen_submap_plan_path)
    disjoint_upstream_authority_path = Path(disjoint_upstream_authority_path)
    source_root = Path(source_root).resolve()
    base_names = list(BASE_ARRAY_NAMES)
    topology_names = list(topology_array_names())
    with np.load(comparison_domain_v3_path, allow_pickle=False) as data:
        domain_metadata = json.loads(str(data["metadata_json"].item()))
        domain_arrays = {
            name: np.asarray(data[name]) for name in base_names + topology_names
        }
    domain_hash = _replay_metadata(
        domain_metadata, "projective seam comparison domain"
    )
    if domain_hash != expected_comparison_domain_content_sha256:
        raise ValueError("projective seam comparison domain differs from pin")
    if domain_metadata.get("artifact_type") != PHYSICAL_DOMAIN_SCHEMA:
        raise ValueError("projective seam requires reference-safe v3 topology")
    if domain_metadata.get("full_submap_gate_primary_stride") != 4 or domain_metadata.get("source_reference_edge_safe") is not True:
        raise ValueError("projective seam v3 is not formal stride-4 physical topology")
    if domain_metadata.get("arrays_sha256") != arrays_sha256(domain_arrays):
        raise ValueError("projective seam comparison arrays differ from lineage")
    base_arrays = {name: domain_arrays[name] for name in base_names}
    topology_arrays = {name: domain_arrays[name] for name in topology_names}
    validate_exact_topology_arrays(
        base_arrays,
        topology_arrays,
        expected_sha256=domain_metadata.get("exact_topology_arrays_sha256"),
    )

    plan = ChartSubmapPlan.load_npz(frozen_submap_plan_path)
    if plan.metadata.get("content_sha256") != expected_plan_content_sha256:
        raise ValueError("projective seam plan differs from pin")
    chart_names = base_arrays["chart_names"].astype(str)
    if chart_names.tolist() != list(plan.selected_chart_names_in_order):
        raise ValueError("projective seam v3 order differs from frozen plan")
    if domain_metadata.get("frozen_submap_plan_content_sha256") != expected_plan_content_sha256:
        raise ValueError("projective seam v3 and plan lineage differ")
    corrected_plan_contract = bool(
        plan.metadata.get("projection_principal_point_convention")
        == "pixel_centers_cx=(W-1)/2_cy=(H-1)/2"
        and plan.metadata.get("project_support_self_reprojection_floor_pass") is True
        and plan.metadata.get("project_support_self_reprojection_phase_pass") is True
        and plan.metadata.get("project_support_self_reprojection_shape_pass") is True
    )

    disjoint = json.loads(disjoint_upstream_authority_path.read_text())
    disjoint_hash = _replay_metadata(disjoint, "projective seam disjoint authority")
    if disjoint_hash != expected_disjoint_authority_content_sha256:
        raise ValueError("projective seam disjoint authority differs from pin")
    if disjoint.get("artifact_type") != DISJOINT_AUTHORITY_SCHEMA or disjoint.get("physical_source_held_input_roots_disjoint") is not True:
        raise ValueError("projective seam lacks physical source/held isolation")
    if disjoint.get("uses_query_or_ground_truth") is not False or disjoint.get("forbidden_routes_opened") is not False:
        raise ValueError("projective seam disjoint authority violates source-only use")
    source = disjoint.get("source", {})
    if Path(source.get("root", "")).resolve() != source_root:
        raise ValueError("projective seam source root differs from authority")
    observed_tree = source_tree_sha256(source_root)
    if observed_tree != expected_source_tree_sha256 or source.get("tree_sha256") != observed_tree:
        raise ValueError("projective seam source tree differs from pin")
    for key, expected in (
        ("disjoint_upstream_authority_content_sha256", disjoint_hash),
        ("source_tree_sha256", observed_tree),
        ("frozen_submap_plan_content_sha256", expected_plan_content_sha256),
    ):
        if domain_metadata.get(key) != expected:
            raise ValueError(f"projective seam v3 {key} differs")

    cameras_path = source_root / "cameras.json"
    if file_sha256(cameras_path) != source.get("cameras_file_sha256") or domain_metadata.get("source_reference_cameras_file_sha256") != file_sha256(cameras_path):
        raise ValueError("projective seam cameras differ from sealed lineage")
    cameras = json.loads(cameras_path.read_text())
    camera_names = [Path(value).name for value in cameras.get("filepaths", [])]
    if camera_names != list(source.get("ordered_names", [])) or not (len(camera_names) == len(cameras.get("focals", [])) == len(cameras.get("cams2world", []))):
        raise ValueError("projective seam source camera inventory differs")
    camera_row = {name: row for row, name in enumerate(camera_names)}
    if any(name not in camera_row for name in chart_names.tolist()):
        raise ValueError("projective seam selected chart lacks a source camera")

    height, width = base_arrays["valid"].shape[1:]
    reference_dense = []
    pointmap_shapes = []
    selected_pointmap_inventory: dict[str, str] = {}
    for name in chart_names.tolist():
        pointmap_path = source_root / "pointmaps" / f"{Path(name).stem}.json"
        pointmap_hash = file_sha256(pointmap_path)
        selected_pointmap_inventory[name] = pointmap_hash
        expected_pointmap = domain_metadata.get(
            "source_reference_selected_pointmap_inventory", {}
        ).get(name)
        if expected_pointmap != pointmap_hash:
            raise ValueError(f"projective seam source pointmap differs for {name}")
        points, shape = _load_source_reference_dense(
            pointmap_path, output_height=height, output_width=width
        )
        reference_dense.append(points)
        pointmap_shapes.append(shape)
    reference_dense_array = np.stack(reference_dense)
    eligible_dense = np.stack(
        [_common_valid_normal_stencil(mask) for mask in base_arrays["valid"]]
    )
    reference_dense_normal = np.stack(
        [
            _dense_world_normals(reference_dense_array[row], eligible_dense[row])
            for row in range(len(chart_names))
        ]
    )
    camera_to_world = np.stack(
        [
            np.asarray(cameras["cams2world"][camera_row[name]], np.float64)
            for name in chart_names.tolist()
        ]
    )
    focal_px = np.asarray(
        [
            float(cameras["focals"][camera_row[name]])
            * width
            / pointmap_shapes[row][1]
            for row, name in enumerate(chart_names.tolist())
        ],
        np.float64,
    )
    stride = config.validated().topology_stride
    metadata = {
        "comparison_domain_v3_file_sha256": file_sha256(comparison_domain_v3_path),
        "comparison_domain_v3_content_sha256": domain_hash,
        "comparison_domain_v3_exact_topology_arrays_sha256": domain_metadata.get("exact_topology_arrays_sha256"),
        "frozen_submap_plan_file_sha256": file_sha256(frozen_submap_plan_path),
        "frozen_submap_plan_content_sha256": expected_plan_content_sha256,
        "selected_chart_names_in_order_sha256": plan.metadata.get("selected_chart_names_in_order_sha256"),
        "disjoint_upstream_authority_file_sha256": file_sha256(disjoint_upstream_authority_path),
        "disjoint_upstream_authority_content_sha256": disjoint_hash,
        "source_root": str(source_root),
        "source_tree_sha256": observed_tree,
        "source_cameras_file_sha256": file_sha256(cameras_path),
        "source_selected_pointmap_inventory": selected_pointmap_inventory,
        "source_selected_pointmap_inventory_sha256": canonical_json_sha256(selected_pointmap_inventory),
        "full_submap_gate_primary_stride": 4,
        "source_reference_edge_safe": True,
        "plan_projection_contract_corrected": corrected_plan_contract,
        "legacy_plan_phase_diagnostic_only": not corrected_plan_contract,
        "projective_correspondence_production_candidate": corrected_plan_contract,
        "final_model_neutral_map_topology_eligible": False,
        "topology_caveat": "v3_common_mask_encodes_paired_DAV2_MoGe_validity_and_is_not_model_neutral",
    }
    return freeze_projective_exact_face_correspondences(
        chart_names=chart_names,
        common_valid=base_arrays["valid"],
        chart_vertex_offsets=topology_arrays[
            f"sampled_vertex_offsets_stride{stride}"
        ],
        sampled_vertex_pixel_indices=topology_arrays[
            f"sampled_vertex_pixel_indices_stride{stride}"
        ],
        chart_face_offsets=topology_arrays[f"face_offsets_stride{stride}"],
        faces=topology_arrays[f"faces_stride{stride}"],
        reference_points_world=reference_dense_array,
        reference_dense_normals_world=reference_dense_normal,
        camera_to_world=camera_to_world,
        focal_px=focal_px,
        plan_chart_names=plan.chart_names,
        coverage_edges=plan.coverage_edges,
        symmetric_surface_overlap=plan.symmetric_surface_overlap,
        config=config,
        metadata=metadata,
    )


def _evaluate_geometry_state(
    authority: ProjectiveExactFaceSeamAuthority,
    vertices_world: np.ndarray,
    *,
    material_weld_config: MaterialWeldConfig = MaterialWeldConfig(),
) -> dict[str, object]:
    vertices_world = np.asarray(vertices_world, np.float64)
    if vertices_world.shape != authority.reference_vertices_world.shape or not np.isfinite(vertices_world).all():
        raise ValueError("projective seam evaluated geometry differs from topology")
    config = ProjectiveSeamConfig(**authority.metadata["config"]).validated()
    normals = _surface_vertex_normals(vertices_world, authority.faces)
    edge_rows = []
    direction_geometry_valid = np.zeros_like(authority.direction_geometry_valid)
    direction_formal_valid = np.zeros_like(authority.direction_formal_valid)
    direction_material_weld_valid = np.zeros_like(
        authority.direction_formal_valid
    )
    direction_composite_valid = np.zeros_like(authority.direction_formal_valid)
    for edge, (first, second) in enumerate(authority.edge_chart_indices):
        lo, hi = map(
            int, authority.edge_correspondence_offsets[edge : edge + 2]
        )
        direction_rows = []
        for direction in range(2):
            take = np.flatnonzero(
                authority.correspondence_direction[lo:hi] == direction
            ) + lo
            metrics = _direction_geometry_metrics(
                vertices_world,
                normals,
                authority.faces,
                authority.source_vertex_indices[take],
                authority.target_face_indices[take],
                authority.target_barycentric[take],
                config,
            )
            geometry_valid = bool(metrics.pop("geometry_valid"))
            material_metrics = _direction_material_weld_metrics(
                authority.reference_vertices_world,
                vertices_world,
                authority.faces,
                authority.source_vertex_indices[take],
                authority.target_face_indices[take],
                authority.target_barycentric[take],
                material_weld_config,
            )
            material_weld_valid = bool(
                material_metrics.pop("material_weld_valid")
            )
            support_valid = bool(authority.direction_support_valid[edge, direction])
            formal_valid = support_valid and geometry_valid
            composite_valid = formal_valid and material_weld_valid
            direction_geometry_valid[edge, direction] = geometry_valid
            direction_formal_valid[edge, direction] = formal_valid
            direction_material_weld_valid[edge, direction] = (
                support_valid and material_weld_valid
            )
            direction_composite_valid[edge, direction] = composite_valid
            direction_rows.append(
                {
                    "direction": "first_to_second" if direction == 0 else "second_to_first",
                    "source": str(authority.chart_names[int(first if direction == 0 else second)]),
                    "target": str(authority.chart_names[int(second if direction == 0 else first)]),
                    "eligible_source_count": int(authority.source_eligible_count_by_direction[edge, direction]),
                    "correspondence_count": int(authority.source_count_by_direction[edge, direction]),
                    "supported_fraction": float(authority.source_supported_fraction_by_direction[edge, direction]),
                    "minimum_supported_fraction_required": float(authority.edge_minimum_supported_fraction_required[edge]),
                    "support_valid": support_valid,
                    **metrics,
                    "geometry_valid": geometry_valid,
                    "direction_formal_valid": formal_valid,
                    **material_metrics,
                    "material_weld_valid": material_weld_valid,
                    "direction_material_weld_valid": bool(
                        support_valid and material_weld_valid
                    ),
                    "direction_composite_valid": composite_valid,
                }
            )
        edge_formal = bool(np.all(direction_formal_valid[edge]))
        edge_material_weld = bool(
            np.all(direction_material_weld_valid[edge])
        )
        edge_composite = bool(np.all(direction_composite_valid[edge]))
        edge_rows.append(
            {
                "first": str(authority.chart_names[int(first)]),
                "second": str(authority.chart_names[int(second)]),
                "directions": direction_rows,
                "edge_formal_valid": edge_formal,
                "edge_material_weld_valid": edge_material_weld,
                "edge_composite_valid": edge_composite,
            }
        )
    edge_formal_valid = np.all(direction_formal_valid, axis=1)
    edge_material_weld_valid = np.all(direction_material_weld_valid, axis=1)
    edge_composite_valid = np.all(direction_composite_valid, axis=1)
    return {
        "edge_count": len(edge_rows),
        "direction_count": int(direction_formal_valid.size),
        "direction_support_valid_count": int(np.sum(authority.direction_support_valid)),
        "direction_geometry_valid_count": int(np.sum(direction_geometry_valid)),
        "direction_formal_valid_count": int(np.sum(direction_formal_valid)),
        "edge_formal_valid_count": int(np.sum(edge_formal_valid)),
        "all_edges_formal_valid": bool(np.all(edge_formal_valid)),
        "formal_decision": "GO" if bool(np.all(edge_formal_valid)) else "KILL",
        "direction_material_weld_valid_count": int(
            np.sum(direction_material_weld_valid)
        ),
        "edge_material_weld_valid_count": int(
            np.sum(edge_material_weld_valid)
        ),
        "all_edges_material_weld_valid": bool(
            np.all(edge_material_weld_valid)
        ),
        "material_weld_decision": (
            "GO" if bool(np.all(edge_material_weld_valid)) else "KILL"
        ),
        "direction_composite_valid_count": int(
            np.sum(direction_composite_valid)
        ),
        "edge_composite_valid_count": int(np.sum(edge_composite_valid)),
        "all_edges_composite_valid": bool(np.all(edge_composite_valid)),
        "composite_decision": (
            "GO" if bool(np.all(edge_composite_valid)) else "KILL"
        ),
        "direction_quantiles_pooled": False,
        "edge_formal_valid_definition": EDGE_FORMAL_VALID_DEFINITION,
        "material_weld_semantics_version": MATERIAL_WELD_SEMANTICS_VERSION,
        "material_weld_config": material_weld_config.validated().to_dict(),
        "material_weld_is_evaluator_only": True,
        "per_edge": edge_rows,
    }


def evaluate_projective_exact_face_seam_geometry(
    authority: ProjectiveExactFaceSeamAuthority,
    arms: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Evaluate frozen projective pairs with directional hard gates."""

    authority = authority.validated()
    m0 = _evaluate_geometry_state(authority, authority.reference_vertices_world)
    arm_rows = {
        str(name): _evaluate_geometry_state(authority, vertices)
        for name, vertices in arms.items()
    }
    m0_replays_authority = bool(
        m0["edge_formal_valid_count"] == int(np.sum(authority.edge_formal_valid))
        and m0["direction_formal_valid_count"]
        == int(np.sum(authority.direction_formal_valid))
    )
    report = {
        "artifact_type": REPORT_SCHEMA,
        "authority_semantics_version": AUTHORITY_SEMANTICS_VERSION,
        "authority_content_sha256": authority.metadata["content_sha256"],
        "authority_arrays_sha256": authority.metadata["arrays_sha256"],
        "uses_query_or_ground_truth": False,
        "held_geometry_consumed": False,
        "direction_quantiles_pooled": False,
        "hard_point_to_plane_residual": "max(abs(delta_dot_source_normal),abs(delta_dot_target_normal))",
        "material_weld_semantics_version": MATERIAL_WELD_SEMANTICS_VERSION,
        "material_weld_edge_valid_definition": (
            MATERIAL_WELD_EDGE_VALID_DEFINITION
        ),
        "hard_material_weld_residual": (
            "max(0,euclidean_pair_distance_arm-euclidean_pair_distance_M0)"
        ),
        "material_weld_quantiles_are_directional": True,
        "material_weld_cannot_revive_missing_support": True,
        "edge_formal_valid_definition": EDGE_FORMAL_VALID_DEFINITION,
        "m0_source_reference": m0,
        "m0_replays_authority": m0_replays_authority,
        "arms": arm_rows,
        "formal_arm_gate_eligible": bool(
            m0_replays_authority
            and authority.metadata.get("projective_correspondence_production_candidate")
            is True
            and m0["all_edges_formal_valid"] is True
        ),
    }
    report["content_sha256"] = canonical_json_sha256(report)
    return report
