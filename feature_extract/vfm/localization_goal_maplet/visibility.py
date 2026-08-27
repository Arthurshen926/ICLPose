"""Signed, full-scene z-buffer visibility for exact Goal-Maplet members."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feature_extract.vfm.gaussian_vfm_field import GaussianVFMFeatureView
from feature_extract.vfm.vfm_2dgs_mapping import (
    SurfaceElementMap,
    _render_surface_element_pixel_contributions_2dgs,
)

from .physical_map import DOUBLE_SIDED, GoalMapletPhysicalMap


def camera_center_from_w2c(pose_w2c: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    return -pose[:3, :3].T @ pose[:3, 3]


def signed_surface_visibility(
    centers: np.ndarray,
    normals: np.ndarray,
    sidedness: np.ndarray,
    pose_w2c: np.ndarray,
    *,
    minimum_incidence: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Return visibility and non-negative incidence without ``abs(n dot v)``."""

    camera_center = camera_center_from_w2c(pose_w2c)
    view = camera_center[None] - np.asarray(centers, dtype=np.float64)
    view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-12)
    signed = np.sum(np.asarray(normals, dtype=np.float64) * view, axis=1)
    double_sided = np.asarray(sidedness, dtype=np.uint8) == DOUBLE_SIDED
    incidence = np.where(double_sided, np.abs(signed), np.maximum(signed, 0.0))
    return incidence >= float(minimum_incidence), incidence.astype(np.float32)


def dominant_maplet_owner(physical_map: GoalMapletPhysicalMap) -> np.ndarray:
    """Resolve the rare overlapping membership by maximum physical weight."""

    owner = np.full((physical_map.primitive_ids.size,), -1, dtype=np.int64)
    weight = np.full((physical_map.primitive_ids.size,), -np.inf, dtype=np.float32)
    for maplet_row in range(physical_map.maplet_ids.size):
        link = physical_map.member_slice(maplet_row)
        rows = physical_map.membership_primitive_rows[link]
        values = physical_map.membership_weights[link]
        replace = values > weight[rows]
        owner[rows[replace]] = maplet_row
        weight[rows[replace]] = values[replace]
    return owner


@dataclass(frozen=True)
class MapletVisibilityBuffer:
    dominant_maplet_rows: np.ndarray
    dominant_primitive_ids: np.ndarray
    dominant_weights: np.ndarray
    depth: np.ndarray
    visible_maplet_rows: np.ndarray
    visible_pixel_counts: np.ndarray
    metadata: dict[str, object]


def render_exact_maplet_visibility(
    physical_map: GoalMapletPhysicalMap,
    pose_w2c: np.ndarray,
    camera,
    *,
    width: int,
    height: int,
    device: str = "cuda",
    minimum_incidence: float = 0.05,
) -> MapletVisibilityBuffer:
    """Rasterize clean scene disks, then label only exact maplet members.

    Non-maplet clean primitives stay in the rasterization and can occlude
    owned primitives.  This is therefore a full-scene z-buffer rather than a
    selected-maplet self-occlusion approximation.
    """

    front, _ = signed_surface_visibility(
        physical_map.primitive_centers,
        physical_map.primitive_normals,
        physical_map.primitive_sidedness,
        pose_w2c,
        minimum_incidence=float(minimum_incidence),
    )
    rows = np.flatnonzero(front)
    elements = SurfaceElementMap(
        element_ids=physical_map.primitive_ids[rows],
        parent_gaussian_indices=physical_map.primitive_ids[rows],
        centers=physical_map.primitive_centers[rows],
        tangent1=physical_map.primitive_tangent1[rows],
        tangent2=physical_map.primitive_tangent2[rows],
        normals=physical_map.primitive_normals[rows],
        scale1=physical_map.primitive_scale1[rows],
        scale2=physical_map.primitive_scale2[rows],
        opacity=physical_map.primitive_opacity[rows],
        area=np.pi * physical_map.primitive_scale1[rows] * physical_map.primitive_scale2[rows],
        adjacency=tuple(),
        metadata={"representation": "goal_maplet_full_clean_scene_signed_visibility"},
    )
    view = GaussianVFMFeatureView(
        image_id="goal_maplet_visibility",
        feature_map=np.zeros((1, int(height), int(width)), dtype=np.float32),
        pose_w2c=np.asarray(pose_w2c, dtype=np.float64),
        camera=camera,
    )
    pixel_ids, local_rows, contribution, primitive_depth = (
        _render_surface_element_pixel_contributions_2dgs(
            elements,
            view,
            width=int(width),
            height=int(height),
            device=str(device),
        )
    )
    pixel_count = int(width) * int(height)
    best_weight = np.zeros((pixel_count,), dtype=np.float32)
    best_compact_row = np.full((pixel_count,), -1, dtype=np.int64)
    if pixel_ids.size:
        # Contributions already include front-to-back alpha transmittance.
        order = np.lexsort((local_rows, -contribution, pixel_ids))
        ordered_pixels = pixel_ids[order]
        first = np.r_[True, ordered_pixels[1:] != ordered_pixels[:-1]]
        selected = order[first]
        best_weight[pixel_ids[selected]] = contribution[selected]
        best_compact_row[pixel_ids[selected]] = rows[local_rows[selected]]
    owner = dominant_maplet_owner(physical_map)
    labels = np.full((pixel_count,), -1, dtype=np.int64)
    primitive_ids = np.full((pixel_count,), -1, dtype=np.int64)
    depth = np.zeros((pixel_count,), dtype=np.float32)
    valid = best_compact_row >= 0
    labels[valid] = owner[best_compact_row[valid]]
    primitive_ids[valid] = physical_map.primitive_ids[best_compact_row[valid]]
    # ``primitive_depth`` is indexed in the compact front-facing ``elements``
    # inventory, whereas ``best_compact_row`` has already been remapped to the
    # full physical-map row inventory.  Expand once before the global lookup;
    # indexing the compact array by a global row can silently select the wrong
    # depth or raise when a high-index primitive is visible.
    global_primitive_depth = np.zeros((physical_map.primitive_ids.size,), dtype=np.float32)
    global_primitive_depth[rows] = primitive_depth
    depth[valid] = global_primitive_depth[best_compact_row[valid]]
    visible_labels = labels[labels >= 0]
    visible_rows, counts = np.unique(visible_labels, return_counts=True)
    return MapletVisibilityBuffer(
        dominant_maplet_rows=labels.reshape(int(height), int(width)),
        dominant_primitive_ids=primitive_ids.reshape(int(height), int(width)),
        dominant_weights=best_weight.reshape(int(height), int(width)),
        depth=depth.reshape(int(height), int(width)),
        visible_maplet_rows=visible_rows.astype(np.int64),
        visible_pixel_counts=counts.astype(np.int64),
        metadata={
            "renderer": "gsplat.rasterization_2dgs",
            "visibility": "signed_normal_full_clean_scene_zbuffer",
            "uses_abs_normal_for_single_sided": False,
            "scene_primitive_count": int(physical_map.primitive_ids.size),
            "front_facing_primitive_count": int(rows.size),
        },
    )
