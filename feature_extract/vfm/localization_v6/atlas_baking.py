"""Raster-contributor-verified multi-view feature baking for V6 atlases."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from feature_extract.vfm.gaussian_vfm_field import GaussianVFMFeatureView
from feature_extract.vfm.localization.continuous_surface_alignment import (
    project_world_points,
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)
from feature_extract.vfm.localization_v6.primitive_contributors import (
    PrimitiveContributorBuffer,
)


def _bilinear_sample(feature_map: np.ndarray, xy: np.ndarray) -> np.ndarray:
    features = np.asarray(feature_map, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError("feature map must have shape (C,H,W)")
    channels, height, width = features.shape
    points = np.asarray(xy, dtype=np.float64)
    x0 = np.floor(points[:, 0]).astype(np.int64)
    y0 = np.floor(points[:, 1]).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    dx = (points[:, 0] - x0).astype(np.float32)
    dy = (points[:, 1] - y0).astype(np.float32)
    flat = features.reshape(channels, -1)
    v00 = flat[:, y0 * width + x0].T
    v10 = flat[:, y0 * width + x1].T
    v01 = flat[:, y1 * width + x0].T
    v11 = flat[:, y1 * width + x1].T
    return (
        v00 * ((1.0 - dx) * (1.0 - dy))[:, None]
        + v10 * (dx * (1.0 - dy))[:, None]
        + v01 * ((1.0 - dx) * dy)[:, None]
        + v11 * (dx * dy)[:, None]
    )


def bake_feature_atlas(
    geometry: MapletFeatureAtlasBank,
    views: Sequence[GaussianVFMFeatureView],
    contributor_buffers: Sequence[PrimitiveContributorBuffer],
    *,
    minimum_contributor_weight: float = 0.01,
    minimum_support: int = 2,
    trajectory_ids: Sequence[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> tuple[MapletFeatureAtlasBank, dict[str, object]]:
    """Fuse metric features without allowing observations to move geometry.

    A projected texel is accepted only when its canonical primitive ID occurs
    in the rasterizer's top-k contributor list at that pixel.  This is the
    production assignment rule: no nearest-neighbour or depth-only fallback.
    """

    if len(views) != len(contributor_buffers) or not views:
        raise ValueError("views and contributor buffers must be equally non-empty")
    channels = int(np.asarray(views[0].feature_map).shape[0])
    flat_valid = np.asarray(geometry.valid_mask, dtype=bool).reshape(-1)
    flat_xyz = np.asarray(geometry.xyz, dtype=np.float64).reshape(-1, 3)
    flat_primitive = np.asarray(geometry.primitive_ids, dtype=np.int64).reshape(-1)
    valid_rows = np.flatnonzero(flat_valid)
    sums = np.zeros((flat_valid.size, channels), dtype=np.float64)
    squared_sums = np.zeros((flat_valid.size,), dtype=np.float64)
    weight_sums = np.zeros((flat_valid.size,), dtype=np.float64)
    support = np.zeros((flat_valid.size,), dtype=np.int32)
    accepted_by_view: list[int] = []
    for view, contributors in zip(views, contributor_buffers):
        fmap = np.asarray(view.feature_map, dtype=np.float32)
        if fmap.shape[0] != channels:
            raise ValueError("all feature maps must have the same channels")
        height, width = fmap.shape[1:]
        contributor_height, contributor_width = contributors.dominant_ids.shape
        image_xy, depth = project_world_points(
            flat_xyz[valid_rows], view.pose_w2c, view.camera
        )
        grid_xy = np.stack(
            [
                (image_xy[:, 0] + 0.5) * width / view.camera.width - 0.5,
                (image_xy[:, 1] + 0.5) * height / view.camera.height - 0.5,
            ],
            axis=1,
        )
        contributor_xy = np.stack(
            [
                (image_xy[:, 0] + 0.5)
                * contributor_width
                / view.camera.width
                - 0.5,
                (image_xy[:, 1] + 0.5)
                * contributor_height
                / view.camera.height
                - 0.5,
            ],
            axis=1,
        )
        inside = (
            np.isfinite(grid_xy).all(axis=1)
            & np.isfinite(contributor_xy).all(axis=1)
            & np.isfinite(depth)
            & (depth > 0.0)
            & (grid_xy[:, 0] >= 0.0)
            & (grid_xy[:, 0] <= width - 1)
            & (grid_xy[:, 1] >= 0.0)
            & (grid_xy[:, 1] <= height - 1)
            & (contributor_xy[:, 0] >= 0.0)
            & (contributor_xy[:, 0] <= contributor_width - 1)
            & (contributor_xy[:, 1] >= 0.0)
            & (contributor_xy[:, 1] <= contributor_height - 1)
        )
        candidate = np.flatnonzero(inside)
        if not candidate.size:
            accepted_by_view.append(0)
            continue
        nearest_x = np.clip(
            np.rint(contributor_xy[candidate, 0]).astype(np.int64),
            0,
            contributor_width - 1,
        )
        nearest_y = np.clip(
            np.rint(contributor_xy[candidate, 1]).astype(np.int64),
            0,
            contributor_height - 1,
        )
        top_ids = contributors.topk_ids[nearest_y, nearest_x]
        top_weights = contributors.topk_weights[nearest_y, nearest_x]
        expected_ids = flat_primitive[valid_rows[candidate]]
        identity_match = top_ids == expected_ids[:, None]
        identity_weight = np.max(
            np.where(identity_match, top_weights, 0.0), axis=1
        )
        accepted_local = identity_weight >= float(minimum_contributor_weight)
        candidate = candidate[accepted_local]
        if not candidate.size:
            accepted_by_view.append(0)
            continue
        rows = valid_rows[candidate]
        weights = identity_weight[accepted_local].astype(np.float64)
        samples = _bilinear_sample(fmap, grid_xy[candidate])
        sample_norm = np.linalg.norm(samples, axis=1)
        finite = np.isfinite(samples).all(axis=1) & (sample_norm > 1e-8)
        rows = rows[finite]
        weights = weights[finite]
        samples = samples[finite] / sample_norm[finite, None]
        np.add.at(sums, rows, weights[:, None] * samples)
        np.add.at(squared_sums, rows, weights * np.sum(samples * samples, axis=1))
        np.add.at(weight_sums, rows, weights)
        np.add.at(support, rows, 1)
        accepted_by_view.append(int(rows.size))
    mean = np.zeros_like(sums, dtype=np.float32)
    has_weight = weight_sums > 0.0
    mean[has_weight] = (
        sums[has_weight] / weight_sums[has_weight, None]
    ).astype(np.float32)
    mean_norm = np.linalg.norm(mean, axis=1)
    mean[mean_norm > 1e-8] /= mean_norm[mean_norm > 1e-8, None]
    # For unit input descriptors this is a stable scalar dispersion measure.
    resultant = np.zeros_like(weight_sums)
    resultant[has_weight] = (
        np.linalg.norm(sums[has_weight], axis=1) / weight_sums[has_weight]
    )
    variance = np.ones_like(weight_sums, dtype=np.float32)
    variance[has_weight] = np.clip(
        1.0 - resultant[has_weight], 0.0, 1.0
    ).astype(np.float32)
    baked_valid = flat_valid & (support >= int(minimum_support))
    shape = geometry.valid_mask.shape
    contract = dict(geometry.metadata or {})
    contract.update(metadata or {})
    contract.update(
        {
            "artifact_type": "v6_canonical_maplet_feature_atlas",
            "feature_assignment": "gsplat_topk_contributor_source_index",
            "observation_dependent_geometry": False,
            "feature_state": "baked",
            "minimum_support": int(minimum_support),
            "stores_mapping_rgb": False,
            "stores_mapping_image_paths": False,
            "uses_kdtree_fallback": False,
        }
    )
    if trajectory_ids is not None:
        unique_trajectories = sorted(set(str(value) for value in trajectory_ids))
        contract["mapping_trajectory_count"] = len(unique_trajectories)
        contract["mapping_trajectory_ids"] = unique_trajectories
    bank = replace(
        geometry,
        features=mean.reshape(*shape, channels).transpose(0, 3, 1, 2),
        variance=variance.reshape(shape),
        support_count=support.reshape(shape),
        valid_mask=baked_valid.reshape(shape),
        metadata=contract,
    )
    report = {
        "stage": "v6_canonical_feature_atlas_baking",
        "view_count": len(views),
        "accepted_texel_observations_by_view": accepted_by_view,
        "accepted_texel_observation_count": int(sum(accepted_by_view)),
        "geometry_valid_cell_count": int(np.sum(flat_valid)),
        "baked_valid_cell_count": int(np.sum(baked_valid)),
        "baked_valid_fraction_of_geometry": float(
            np.sum(baked_valid) / max(np.sum(flat_valid), 1)
        ),
        "support_quantiles": {
            "median": float(np.median(support[baked_valid]))
            if np.any(baked_valid)
            else 0.0,
            "p90": float(np.quantile(support[baked_valid], 0.9))
            if np.any(baked_valid)
            else 0.0,
        },
        "assignment": "gsplat_topk_contributor_source_index",
        "uses_kdtree_fallback": False,
        "observation_dependent_geometry": False,
    }
    return bank, report
