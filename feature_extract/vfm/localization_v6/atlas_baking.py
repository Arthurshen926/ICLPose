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


def _bilinear_contributor_mass(
    contributors: PrimitiveContributorBuffer,
    xy: np.ndarray,
    expected_ids: np.ndarray,
) -> np.ndarray:
    """Integrate exact contributor identity over the subpixel footprint."""

    points = np.asarray(xy, dtype=np.float64)
    height, width = contributors.dominant_ids.shape
    x0 = np.floor(points[:, 0]).astype(np.int64)
    y0 = np.floor(points[:, 1]).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    dx = points[:, 0] - x0
    dy = points[:, 1] - y0
    result = np.zeros((points.shape[0],), dtype=np.float64)
    for px, py, footprint in (
        (x0, y0, (1.0 - dx) * (1.0 - dy)),
        (x1, y0, dx * (1.0 - dy)),
        (x0, y1, (1.0 - dx) * dy),
        (x1, y1, dx * dy),
    ):
        top_ids = contributors.topk_ids[py, px]
        top_weights = contributors.topk_weights[py, px]
        identity_mass = np.sum(
            np.where(top_ids == expected_ids[:, None], top_weights, 0.0),
            axis=1,
        )
        result += footprint * identity_mass
    return result.astype(np.float32)


def bake_feature_atlas(
    geometry: MapletFeatureAtlasBank,
    views: Sequence[GaussianVFMFeatureView],
    contributor_buffers: Sequence[PrimitiveContributorBuffer],
    *,
    minimum_contributor_weight: float = 0.01,
    minimum_support: int = 2,
    trajectory_ids: Sequence[str] | None = None,
    metadata: dict[str, object] | None = None,
    appearance_modes: int = 1,
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
    observation_rows: list[np.ndarray] = []
    observation_features: list[np.ndarray] = []
    observation_weights: list[np.ndarray] = []
    observation_directions: list[np.ndarray] = []
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
        expected_ids = flat_primitive[valid_rows[candidate]]
        identity_weight = _bilinear_contributor_mass(
            contributors,
            contributor_xy[candidate],
            expected_ids,
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
        camera_center = (
            -np.asarray(view.pose_w2c[:3, :3], dtype=np.float64).T
            @ np.asarray(view.pose_w2c[:3, 3], dtype=np.float64)
        )
        directions = camera_center[None] - flat_xyz[rows]
        directions /= np.maximum(
            np.linalg.norm(directions, axis=1, keepdims=True), 1e-8
        )
        observation_rows.append(rows.copy())
        observation_features.append(samples.astype(np.float32))
        observation_weights.append(weights.astype(np.float32))
        observation_directions.append(directions.astype(np.float32))
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
            "feature_assignment": "bilinear_gsplat_topk_contributor_source_index",
            "observation_dependent_geometry": False,
            "feature_state": "baked",
            "minimum_support": int(minimum_support),
            "stores_mapping_rgb": False,
            "stores_mapping_image_paths": False,
            "uses_kdtree_fallback": False,
            "appearance_mode_count": int(max(1, appearance_modes)),
            "view_conditioned_appearance": int(appearance_modes) > 1,
        }
    )
    if trajectory_ids is not None:
        unique_trajectories = sorted(set(str(value) for value in trajectory_ids))
        contract["mapping_trajectory_count"] = len(unique_trajectories)
        contract["mapping_trajectory_ids"] = unique_trajectories
    mode_payload: dict[str, np.ndarray | None] = {
        "mode_features": None,
        "mode_weights": None,
        "mode_view_directions": None,
        "mode_view_covariance": None,
        "mode_variance": None,
        "mode_valid_mask": None,
    }
    mode_count = int(max(1, appearance_modes))
    if mode_count > 1:
        mode_shape = (flat_valid.size, mode_count)
        mode_features = np.zeros((*mode_shape, channels), dtype=np.float32)
        mode_weights = np.zeros(mode_shape, dtype=np.float32)
        mode_directions = np.zeros((*mode_shape, 3), dtype=np.float32)
        mode_covariance = np.zeros((*mode_shape, 3, 3), dtype=np.float32)
        mode_variance = np.ones(mode_shape, dtype=np.float32)
        mode_valid = np.zeros(mode_shape, dtype=bool)
        all_rows = np.concatenate(observation_rows)
        all_features = np.concatenate(observation_features)
        all_weights = np.concatenate(observation_weights)
        all_directions = np.concatenate(observation_directions)
        order = np.argsort(all_rows, kind="stable")
        all_rows = all_rows[order]
        all_features = all_features[order]
        all_weights = all_weights[order]
        all_directions = all_directions[order]
        unique, starts = np.unique(all_rows, return_index=True)
        ends = np.r_[starts[1:], all_rows.size]
        for row, begin, end in zip(unique.tolist(), starts.tolist(), ends.tolist()):
            descriptors = all_features[begin:end]
            weights = all_weights[begin:end].astype(np.float64)
            directions = all_directions[begin:end]
            clusters = min(mode_count, descriptors.shape[0])
            # Deterministic farthest-first spherical clustering.  This avoids
            # storing per-view descriptors while preserving distinct modes.
            centres = [int(np.argmax(weights))]
            while len(centres) < clusters:
                similarity = descriptors @ descriptors[centres].T
                distance = 1.0 - np.max(similarity, axis=1)
                distance[centres] = -1.0
                centres.append(int(np.argmax(distance * weights)))
            centroids = descriptors[centres].copy()
            labels = np.zeros(descriptors.shape[0], dtype=np.int64)
            for _iteration in range(6):
                labels = np.argmax(descriptors @ centroids.T, axis=1)
                for mode in range(clusters):
                    selected = labels == mode
                    if not np.any(selected):
                        continue
                    vector = np.sum(
                        descriptors[selected] * weights[selected, None], axis=0
                    )
                    centroids[mode] = vector / max(np.linalg.norm(vector), 1e-8)
            total_weight = max(float(np.sum(weights)), 1e-8)
            for mode in range(clusters):
                selected = labels == mode
                if not np.any(selected):
                    continue
                local_weight = weights[selected]
                weight_sum = max(float(np.sum(local_weight)), 1e-8)
                mode_features[row, mode] = centroids[mode]
                mode_weights[row, mode] = weight_sum / total_weight
                direction_mean = np.sum(
                    directions[selected] * local_weight[:, None], axis=0
                ) / weight_sum
                mode_directions[row, mode] = direction_mean / max(
                    np.linalg.norm(direction_mean), 1e-8
                )
                residual = directions[selected] - direction_mean[None]
                mode_covariance[row, mode] = np.einsum(
                    "n,ni,nj->ij", local_weight, residual, residual
                ) / weight_sum
                resultant = np.linalg.norm(
                    np.sum(descriptors[selected] * local_weight[:, None], axis=0)
                ) / weight_sum
                mode_variance[row, mode] = np.clip(1.0 - resultant, 0.0, 1.0)
                mode_valid[row, mode] = True
        atlas_shape = geometry.valid_mask.shape
        mode_payload = {
            "mode_features": mode_features.reshape(
                *atlas_shape, mode_count, channels
            ).transpose(0, 3, 4, 1, 2),
            "mode_weights": mode_weights.reshape(*atlas_shape, mode_count).transpose(
                0, 3, 1, 2
            ),
            "mode_view_directions": mode_directions.reshape(
                *atlas_shape, mode_count, 3
            ).transpose(0, 3, 1, 2, 4),
            "mode_view_covariance": mode_covariance.reshape(
                *atlas_shape, mode_count, 3, 3
            ).transpose(0, 3, 1, 2, 4, 5),
            "mode_variance": mode_variance.reshape(
                *atlas_shape, mode_count
            ).transpose(0, 3, 1, 2),
            "mode_valid_mask": mode_valid.reshape(
                *atlas_shape, mode_count
            ).transpose(0, 3, 1, 2),
        }
    bank = replace(
        geometry,
        features=mean.reshape(*shape, channels).transpose(0, 3, 1, 2),
        variance=variance.reshape(shape),
        support_count=support.reshape(shape),
        valid_mask=baked_valid.reshape(shape),
        metadata=contract,
        **mode_payload,
    )
    geometry_per_maplet = np.sum(flat_valid.reshape(shape), axis=(1, 2))
    baked_per_maplet = np.sum(baked_valid.reshape(shape), axis=(1, 2))
    per_maplet_coverage = baked_per_maplet / np.maximum(
        geometry_per_maplet, 1
    )
    texel_size = np.sqrt(
        np.maximum(
            (2.0 * np.asarray(geometry.extents[:, 0]) / geometry.width)
            * (2.0 * np.asarray(geometry.extents[:, 1]) / geometry.height),
            0.0,
        )
    )
    direction_span = np.zeros((len(geometry),), dtype=np.float32)
    if observation_rows:
        all_observation_rows = np.concatenate(observation_rows)
        all_observation_directions = np.concatenate(observation_directions)
        observation_maplets = all_observation_rows // (geometry.height * geometry.width)
        for maplet_row in np.unique(observation_maplets):
            local = observation_maplets == maplet_row
            direction_span[maplet_row] = 1.0 - np.linalg.norm(
                np.mean(all_observation_directions[local], axis=0)
            )
    supported_maplets = geometry_per_maplet > 0
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
        "per_maplet_coverage": {
            "geometry_maplet_count": int(np.sum(supported_maplets)),
            "baked_maplet_count": int(
                np.sum(supported_maplets & (baked_per_maplet > 0))
            ),
            "zero_baked_maplet_count": int(
                np.sum(supported_maplets & (baked_per_maplet == 0))
            ),
            "median": float(np.median(per_maplet_coverage[supported_maplets]))
            if np.any(supported_maplets)
            else 0.0,
            "p10": float(np.quantile(per_maplet_coverage[supported_maplets], 0.1))
            if np.any(supported_maplets)
            else 0.0,
            "p90": float(np.quantile(per_maplet_coverage[supported_maplets], 0.9))
            if np.any(supported_maplets)
            else 0.0,
            "median_among_baked": float(
                np.median(per_maplet_coverage[baked_per_maplet > 0])
            )
            if np.any(baked_per_maplet > 0)
            else 0.0,
        },
        "physical_texel_size_m": {
            "median": float(np.median(texel_size[supported_maplets]))
            if np.any(supported_maplets)
            else 0.0,
            "p10": float(np.quantile(texel_size[supported_maplets], 0.1))
            if np.any(supported_maplets)
            else 0.0,
            "p90": float(np.quantile(texel_size[supported_maplets], 0.9))
            if np.any(supported_maplets)
            else 0.0,
        },
        "maplet_view_direction_dispersion": {
            "median": float(np.median(direction_span[baked_per_maplet > 0]))
            if np.any(baked_per_maplet > 0)
            else 0.0,
            "p90": float(np.quantile(direction_span[baked_per_maplet > 0], 0.9))
            if np.any(baked_per_maplet > 0)
            else 0.0,
            "computed_over_baked_maplets": True,
        },
        "assignment": "bilinear_gsplat_topk_contributor_source_index",
        "uses_kdtree_fallback": False,
        "observation_dependent_geometry": False,
        "appearance_mode_count": mode_count,
        "view_conditioned_appearance": mode_count > 1,
    }
    return bank, report
