"""2DGS rasterizer contributor-ID buffers for V6 atlas construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feature_extract.vfm.gaussian_vfm_field import GaussianVFMSource
from feature_extract.vfm.vfm_2dgs_mapping import (
    GaussianVFMFeatureView,
    SurfaceElementMap,
    _render_surface_element_pixel_contributions_2dgs,
    _surface_tangent_axes_and_scales,
)


@dataclass(frozen=True)
class PrimitiveContributorBuffer:
    dominant_ids: np.ndarray
    dominant_weights: np.ndarray
    topk_ids: np.ndarray
    topk_weights: np.ndarray
    primitive_depth: np.ndarray
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        ids = np.asarray(self.dominant_ids, dtype=np.int64)
        weights = np.asarray(self.dominant_weights, dtype=np.float32)
        top_ids = np.asarray(self.topk_ids, dtype=np.int64)
        top_weights = np.asarray(self.topk_weights, dtype=np.float32)
        depth = np.asarray(self.primitive_depth, dtype=np.float32)
        if ids.ndim != 2 or weights.shape != ids.shape or depth.shape != ids.shape:
            raise ValueError("dominant contributor arrays must have shape (H,W)")
        if (
            top_ids.ndim != 3
            or top_ids.shape[:2] != ids.shape
            or top_weights.shape != top_ids.shape
        ):
            raise ValueError("top-k contributor arrays must have shape (H,W,K)")
        if np.any(weights < 0.0) or np.any(top_weights < 0.0):
            raise ValueError("contributor weights must be non-negative")
        object.__setattr__(self, "dominant_ids", ids)
        object.__setattr__(self, "dominant_weights", weights)
        object.__setattr__(self, "topk_ids", top_ids)
        object.__setattr__(self, "topk_weights", top_weights)
        object.__setattr__(self, "primitive_depth", depth)


def clean_primitive_surface_elements(
    source: GaussianVFMSource, clean_source_indices: np.ndarray
) -> SurfaceElementMap:
    """Represent clean source disks without virtual splits or proxy points."""

    rows = np.asarray(clean_source_indices, dtype=np.int64).reshape(-1)
    if (
        rows.size == 0
        or np.any(rows < 0)
        or np.max(rows) >= int(source.xyz.shape[0])
        or np.unique(rows).size != rows.size
    ):
        raise ValueError("clean source indices must be unique valid primitive rows")
    if source.normal is None:
        raise ValueError("complete 2DGS source must provide normals")
    normals = np.asarray(source.normal, dtype=np.float32)[rows]
    tangent1, tangent2, scale1, scale2 = _surface_tangent_axes_and_scales(
        source, rows, normals
    )
    return SurfaceElementMap(
        element_ids=rows,
        parent_gaussian_indices=rows,
        centers=np.asarray(source.xyz, dtype=np.float64)[rows],
        tangent1=tangent1,
        tangent2=tangent2,
        normals=normals,
        scale1=scale1,
        scale2=scale2,
        opacity=np.asarray(source.opacity, dtype=np.float32)[rows],
        area=(np.pi * scale1 * scale2).astype(np.float32),
        adjacency=tuple(),
        metadata={
            "representation": "clean_complete_2dgs_disks",
            "uses_virtual_surface_cells": False,
        },
    )


def render_primitive_contributors(
    elements: SurfaceElementMap,
    view: GaussianVFMFeatureView,
    *,
    width: int,
    height: int,
    top_k: int = 4,
    device: str = "cuda",
) -> PrimitiveContributorBuffer:
    """Rasterize dominant/top-k source IDs with alpha-transmittance weights."""

    k = max(int(top_k), 1)
    pixel_ids, element_rows, weights, depth = (
        _render_surface_element_pixel_contributions_2dgs(
            elements,
            view,
            width=int(width),
            height=int(height),
            device=str(device),
        )
    )
    top_ids = np.full((int(height) * int(width), k), -1, dtype=np.int64)
    top_weights = np.zeros((int(height) * int(width), k), dtype=np.float32)
    if pixel_ids.size:
        order = np.lexsort(
            (
                np.asarray(elements.element_ids, dtype=np.int64)[element_rows],
                -weights,
                pixel_ids,
            )
        )
        pixels = pixel_ids[order]
        rows = element_rows[order]
        ordered_weights = weights[order]
        rank = np.zeros(pixels.shape, dtype=np.int32)
        if pixels.size > 1:
            boundaries = np.r_[True, pixels[1:] != pixels[:-1]]
            starts = np.maximum.accumulate(
                np.where(boundaries, np.arange(pixels.size), 0)
            )
            rank = np.arange(pixels.size) - starts
        keep = rank < k
        top_ids[pixels[keep], rank[keep]] = np.asarray(
            elements.element_ids, dtype=np.int64
        )[rows[keep]]
        top_weights[pixels[keep], rank[keep]] = ordered_weights[keep]
    top_ids = top_ids.reshape(int(height), int(width), k)
    top_weights = top_weights.reshape(int(height), int(width), k)
    dominant_ids = top_ids[..., 0]
    dominant_weights = top_weights[..., 0]
    depth_image = np.zeros((int(height), int(width)), dtype=np.float32)
    if pixel_ids.size:
        # Renderer depth is per primitive.  Use the depth of the dominant row.
        row_by_id = {
            int(value): int(row)
            for row, value in enumerate(elements.element_ids.tolist())
        }
        valid = dominant_ids >= 0
        dominant_rows = np.asarray(
            [row_by_id[int(value)] for value in dominant_ids[valid].tolist()],
            dtype=np.int64,
        )
        depth_image[valid] = np.asarray(depth, dtype=np.float32)[dominant_rows]
    return PrimitiveContributorBuffer(
        dominant_ids=dominant_ids,
        dominant_weights=dominant_weights,
        topk_ids=top_ids,
        topk_weights=top_weights,
        primitive_depth=depth_image,
        metadata={
            "renderer": "gsplat.rasterization_2dgs",
            "assignment": "alpha_transmittance_topk_source_index",
            "top_k": k,
            "uses_kdtree_fallback": False,
        },
    )
