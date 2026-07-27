"""Area rasterization of selected canonical maplet feature atlases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.continuous_surface_alignment import (
    project_world_points,
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)


@dataclass(frozen=True)
class RenderedMapletAtlases:
    feature: np.ndarray
    xyz: np.ndarray
    normal: np.ndarray
    uncertainty: np.ndarray
    maplet_id: np.ndarray
    mask: np.ndarray
    depth: np.ndarray


def _scaled_pixels(
    pixels: np.ndarray,
    camera: ColmapCamera,
    width: int,
    height: int,
) -> np.ndarray:
    value = np.asarray(pixels, dtype=np.float64).copy()
    value[:, 0] = (value[:, 0] + 0.5) * int(width) / int(camera.width) - 0.5
    value[:, 1] = (value[:, 1] + 0.5) * int(height) / int(camera.height) - 0.5
    return value


def _triangle_barycentric(
    triangle: np.ndarray, points: np.ndarray
) -> np.ndarray:
    a, b, c = np.asarray(triangle, dtype=np.float64)
    denominator = (
        (b[1] - c[1]) * (a[0] - c[0])
        + (c[0] - b[0]) * (a[1] - c[1])
    )
    if abs(float(denominator)) < 1e-10:
        return np.full((points.shape[0], 3), np.nan, dtype=np.float64)
    first = (
        (b[1] - c[1]) * (points[:, 0] - c[0])
        + (c[0] - b[0]) * (points[:, 1] - c[1])
    ) / denominator
    second = (
        (c[1] - a[1]) * (points[:, 0] - c[0])
        + (a[0] - c[0]) * (points[:, 1] - c[1])
    ) / denominator
    return np.stack([first, second, 1.0 - first - second], axis=1)


def render_selected_maplet_atlases(
    atlas: MapletFeatureAtlasBank,
    selected_maplet_ids: np.ndarray,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    *,
    width: int,
    height: int,
    full_scene_depth: np.ndarray | None = None,
    occlusion_epsilon: float = 0.03,
) -> RenderedMapletAtlases:
    """Rasterize atlas quads with perspective-correct feature interpolation."""

    output_feature = np.zeros(
        (atlas.feature_dim, int(height), int(width)), dtype=np.float32
    )
    output_xyz = np.zeros((int(height), int(width), 3), dtype=np.float32)
    output_normal = np.zeros((int(height), int(width), 3), dtype=np.float32)
    output_uncertainty = np.ones((int(height), int(width)), dtype=np.float32)
    output_maplet = np.full((int(height), int(width)), -1, dtype=np.int64)
    output_depth = np.full((int(height), int(width)), np.inf, dtype=np.float32)
    id_to_row = {
        int(value): int(row) for row, value in enumerate(atlas.maplet_ids.tolist())
    }
    selected_rows = [
        id_to_row[int(value)]
        for value in np.asarray(selected_maplet_ids, dtype=np.int64).tolist()
        if int(value) in id_to_row
    ]
    triangles = ((0, 1, 2), (0, 2, 3))
    corner_offsets = ((0, 0), (0, 1), (1, 1), (1, 0))
    for maplet_row in selected_rows:
        valid = atlas.valid_mask[maplet_row] & (
            atlas.support_count[maplet_row] > 0
        )
        for y in range(atlas.height - 1):
            for x in range(atlas.width - 1):
                corners = [(y + dy, x + dx) for dy, dx in corner_offsets]
                if not all(valid[yy, xx] for yy, xx in corners):
                    continue
                world = np.asarray(
                    [atlas.xyz[maplet_row, yy, xx] for yy, xx in corners],
                    dtype=np.float64,
                )
                pixels, depth = project_world_points(world, pose_w2c, camera)
                if np.any(depth <= 0.01) or not np.all(np.isfinite(pixels)):
                    continue
                pixels = _scaled_pixels(pixels, camera, int(width), int(height))
                features = np.asarray(
                    [
                        atlas.features[maplet_row, :, yy, xx]
                        for yy, xx in corners
                    ],
                    dtype=np.float32,
                )
                uncertainties = np.asarray(
                    [
                        atlas.variance[maplet_row, yy, xx]
                        for yy, xx in corners
                    ],
                    dtype=np.float32,
                )
                normal = np.asarray(
                    atlas.frames[maplet_row, 2], dtype=np.float32
                )
                for triangle_rows in triangles:
                    indices = np.asarray(triangle_rows, dtype=np.int64)
                    screen_triangle = pixels[indices]
                    minimum = np.floor(np.min(screen_triangle, axis=0)).astype(int)
                    maximum = np.ceil(np.max(screen_triangle, axis=0)).astype(int)
                    x0, y0 = max(minimum[0], 0), max(minimum[1], 0)
                    x1 = min(maximum[0], int(width) - 1)
                    y1 = min(maximum[1], int(height) - 1)
                    if x1 < x0 or y1 < y0:
                        continue
                    yy, xx = np.meshgrid(
                        np.arange(y0, y1 + 1),
                        np.arange(x0, x1 + 1),
                        indexing="ij",
                    )
                    sample_xy = np.stack(
                        [xx.reshape(-1), yy.reshape(-1)], axis=1
                    )
                    barycentric = _triangle_barycentric(
                        screen_triangle, sample_xy
                    )
                    inside = np.all(barycentric >= -1e-6, axis=1)
                    if not np.any(inside):
                        # Conservative subpixel coverage: a projected atlas
                        # triangle smaller than one output pixel still covers
                        # area and must not disappear at stride-8/16.  Use its
                        # centroid in the intersected pixel; this is area
                        # rasterization, not point-center projection.
                        centroid = np.mean(screen_triangle, axis=0)
                        if (
                            centroid[0] < -0.5
                            or centroid[0] >= int(width) - 0.5
                            or centroid[1] < -0.5
                            or centroid[1] >= int(height) - 0.5
                        ):
                            continue
                        sample_xy = np.asarray(
                            [
                                [
                                    np.clip(
                                        np.rint(centroid[0]), 0, int(width) - 1
                                    ),
                                    np.clip(
                                        np.rint(centroid[1]), 0, int(height) - 1
                                    ),
                                ]
                            ],
                            dtype=np.float64,
                        )
                        barycentric = np.full((1, 3), 1.0 / 3.0, dtype=np.float64)
                    else:
                        sample_xy = sample_xy[inside]
                        barycentric = barycentric[inside]
                    inverse_depth = 1.0 / depth[indices]
                    perspective = barycentric * inverse_depth[None]
                    perspective /= np.maximum(
                        np.sum(perspective, axis=1, keepdims=True), 1e-12
                    )
                    sample_depth = 1.0 / np.maximum(
                        np.sum(barycentric * inverse_depth[None], axis=1),
                        1e-12,
                    )
                    sample_x = np.rint(sample_xy[:, 0]).astype(np.int64)
                    sample_y = np.rint(sample_xy[:, 1]).astype(np.int64)
                    closer = sample_depth < output_depth[sample_y, sample_x]
                    if full_scene_depth is not None:
                        scene_depth = np.asarray(full_scene_depth, dtype=np.float32)
                        if scene_depth.shape != (int(height), int(width)):
                            raise ValueError(
                                "full_scene_depth must match render resolution"
                            )
                        reference = scene_depth[sample_y, sample_x]
                        closer &= (reference <= 0.0) | (
                            sample_depth <= reference + float(occlusion_epsilon)
                        )
                    if not np.any(closer):
                        continue
                    sample_x = sample_x[closer]
                    sample_y = sample_y[closer]
                    weights = perspective[closer]
                    output_depth[sample_y, sample_x] = sample_depth[closer]
                    output_xyz[sample_y, sample_x] = (
                        weights @ world[indices]
                    ).astype(np.float32)
                    interpolated = weights @ features[indices]
                    interpolated /= np.maximum(
                        np.linalg.norm(interpolated, axis=1, keepdims=True), 1e-8
                    )
                    output_feature[:, sample_y, sample_x] = interpolated.T
                    output_uncertainty[sample_y, sample_x] = (
                        weights @ uncertainties[indices]
                    )
                    output_normal[sample_y, sample_x] = normal
                    output_maplet[sample_y, sample_x] = atlas.maplet_ids[
                        maplet_row
                    ]
    mask = np.isfinite(output_depth)
    output_depth[~mask] = 0.0
    return RenderedMapletAtlases(
        feature=output_feature,
        xyz=output_xyz,
        normal=output_normal,
        uncertainty=output_uncertainty,
        maplet_id=output_maplet,
        mask=mask,
        depth=output_depth,
    )
