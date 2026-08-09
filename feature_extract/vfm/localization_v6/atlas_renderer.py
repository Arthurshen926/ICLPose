"""Area rasterization of selected canonical maplet feature atlases."""

from __future__ import annotations

from dataclasses import dataclass
import math
import weakref

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - production environment has numba.
    njit = None

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.continuous_surface_alignment import (
    project_world_points,
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)


MINIMUM_MODE_VIEW_DIRECTION_COSINE = 0.25
VIEW_DIRECTION_ANGULAR_STD_FLOOR_DEG = 1.0


@dataclass(frozen=True)
class _PreparedViewModeDensity:
    mean: np.ndarray
    tangent_first: np.ndarray
    tangent: np.ndarray
    eigenvalue: np.ndarray
    eigenvector: np.ndarray
    log_weight: np.ndarray
    valid: np.ndarray


def _prepare_view_mode_density(
    means: np.ndarray,
    covariances: np.ndarray,
    weights: np.ndarray,
    valid: np.ndarray,
    *,
    angular_std_floor_deg: float = VIEW_DIRECTION_ANGULAR_STD_FLOOR_DEG,
) -> _PreparedViewModeDensity:
    mean = np.array(means, dtype=np.float64, copy=True)
    covariance = np.asarray(covariances, dtype=np.float64)
    mode_weight = np.asarray(weights, dtype=np.float64)
    mode_valid = np.asarray(valid, dtype=bool)
    if mean.ndim < 2 or mean.shape[-1] != 3:
        raise ValueError("view-mode means must have shape (K, ..., 3)")
    if covariance.shape != mean.shape + (3,):
        raise ValueError("view-mode covariance shape differs from means")
    if mode_weight.shape != mean.shape[:-1]:
        raise ValueError("view-mode weights shape differs from means")
    if mode_valid.shape != mode_weight.shape:
        raise ValueError("view-mode validity shape differs from weights")
    sigma = np.deg2rad(float(angular_std_floor_deg))
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("view-direction angular floor must be positive")

    mean /= np.maximum(np.linalg.norm(mean, axis=-1, keepdims=True), 1e-12)
    normalized_weight = np.where(
        mode_valid, np.maximum(mode_weight, 0.0), 0.0
    )
    normalized_weight /= np.maximum(
        np.sum(normalized_weight, axis=0, keepdims=True), 1e-12
    )

    reference = np.zeros_like(mean)
    use_x = np.abs(mean[..., 0]) < 0.8
    reference[..., 0] = use_x
    reference[..., 1] = ~use_x
    tangent_first = np.cross(mean, reference)
    tangent_first /= np.maximum(
        np.linalg.norm(tangent_first, axis=-1, keepdims=True), 1e-12
    )
    tangent_second = np.cross(mean, tangent_first)
    tangent = np.stack([tangent_first, tangent_second], axis=-1)
    covariance_2d = np.einsum(
        "...ia,...ij,...jb->...ab",
        tangent,
        covariance,
        tangent,
    )
    covariance_2d = 0.5 * (
        covariance_2d + np.swapaxes(covariance_2d, -1, -2)
    )
    eigenvalue, eigenvector = np.linalg.eigh(covariance_2d)
    eigenvalue = np.maximum(eigenvalue, sigma * sigma)
    return _PreparedViewModeDensity(
        mean=mean,
        tangent_first=tangent_first,
        tangent=tangent,
        eigenvalue=eigenvalue,
        eigenvector=eigenvector,
        log_weight=np.log(np.maximum(normalized_weight, 1e-12)),
        valid=mode_valid,
    )


def _evaluate_prepared_view_mode_density(
    prepared: _PreparedViewModeDensity,
    query_direction: np.ndarray,
) -> np.ndarray:
    direction = np.array(query_direction, dtype=np.float64, copy=True)
    if direction.shape != prepared.mean.shape[1:]:
        raise ValueError("query view direction shape differs from atlas cells")
    direction /= np.maximum(
        np.linalg.norm(direction, axis=-1, keepdims=True), 1e-12
    )

    cosine = np.clip(
        np.einsum(
            "k...i,...i->k...", prepared.mean, direction
        ),
        -1.0,
        1.0,
    )
    angle = np.arccos(cosine)
    tangent_component = (
        direction[None] - cosine[..., None] * prepared.mean
    )
    tangent_norm = np.linalg.norm(
        tangent_component, axis=-1, keepdims=True
    )
    residual_3d = tangent_component * (
        angle[..., None] / np.maximum(tangent_norm, 1e-12)
    )
    antipodal = (tangent_norm[..., 0] < 1e-8) & (cosine < 0.0)
    residual_3d[antipodal] = (
        np.pi * prepared.tangent_first[antipodal]
    )
    residual = np.einsum(
        "...ia,...i->...a", prepared.tangent, residual_3d
    )
    rotated = np.einsum(
        "...ab,...a->...b",
        np.swapaxes(prepared.eigenvector, -1, -2),
        residual,
    )
    log_probability = (
        prepared.log_weight
        - 0.5
        * np.sum(rotated * rotated / prepared.eigenvalue, axis=-1)
        - 0.5 * np.sum(np.log(prepared.eigenvalue), axis=-1)
        - np.log(2.0 * np.pi)
    )
    log_probability[~prepared.valid] = -np.inf
    return log_probability


def _view_mode_log_densities(
    means: np.ndarray,
    covariances: np.ndarray,
    weights: np.ndarray,
    valid: np.ndarray,
    query_direction: np.ndarray,
    *,
    angular_std_floor_deg: float = VIEW_DIRECTION_ANGULAR_STD_FLOOR_DEG,
) -> np.ndarray:
    """Evaluate anonymous view modes with a tangent Gaussian on S2."""

    return _evaluate_prepared_view_mode_density(
        _prepare_view_mode_density(
            means,
            covariances,
            weights,
            valid,
            angular_std_floor_deg=float(angular_std_floor_deg),
        ),
        query_direction,
    )


_VIEW_MODE_CELL_CACHE: dict[
    tuple[int, int, int, int, float],
    tuple[weakref.ReferenceType[MapletFeatureAtlasBank], _PreparedViewModeDensity],
] = {}

_VIEW_MODE_CHART_CACHE: dict[
    tuple[int, int, float],
    tuple[weakref.ReferenceType[MapletFeatureAtlasBank], _PreparedViewModeDensity],
] = {}


def _atlas_cell_view_mode_log_densities(
    atlas: MapletFeatureAtlasBank,
    maplet_row: int,
    yy: int,
    xx: int,
    query_direction: np.ndarray,
    *,
    angular_std_floor_deg: float = VIEW_DIRECTION_ANGULAR_STD_FLOOR_DEG,
) -> np.ndarray:
    """Evaluate one atlas cell while caching its pose-invariant eigensystem."""

    key = (
        id(atlas),
        int(maplet_row),
        int(yy),
        int(xx),
        float(angular_std_floor_deg),
    )
    cached = _VIEW_MODE_CELL_CACHE.get(key)
    if cached is None or cached[0]() is not atlas:
        prepared = _prepare_view_mode_density(
            atlas.mode_view_directions[maplet_row, :, yy, xx],
            atlas.mode_view_covariance[maplet_row, :, yy, xx],
            atlas.mode_weights[maplet_row, :, yy, xx],
            atlas.mode_valid_mask[maplet_row, :, yy, xx],
            angular_std_floor_deg=float(angular_std_floor_deg),
        )
        cached = (weakref.ref(atlas), prepared)
        _VIEW_MODE_CELL_CACHE[key] = cached
    return _evaluate_prepared_view_mode_density(
        cached[1], query_direction
    )


def _atlas_chart_view_mode_log_densities(
    atlas: MapletFeatureAtlasBank,
    maplet_row: int,
    query_direction: np.ndarray,
    *,
    angular_std_floor_deg: float = VIEW_DIRECTION_ANGULAR_STD_FLOOR_DEG,
) -> np.ndarray:
    """Vectorized cell-equivalent view density for one complete chart."""

    key = (
        id(atlas),
        int(maplet_row),
        float(angular_std_floor_deg),
    )
    cached = _VIEW_MODE_CHART_CACHE.get(key)
    if cached is None or cached[0]() is not atlas:
        prepared = _prepare_view_mode_density(
            atlas.mode_view_directions[int(maplet_row)],
            atlas.mode_view_covariance[int(maplet_row)],
            atlas.mode_weights[int(maplet_row)],
            atlas.mode_valid_mask[int(maplet_row)],
            angular_std_floor_deg=float(angular_std_floor_deg),
        )
        cached = (weakref.ref(atlas), prepared)
        _VIEW_MODE_CHART_CACHE[key] = cached
    return _evaluate_prepared_view_mode_density(
        cached[1], np.asarray(query_direction, dtype=np.float64)
    )


def atlas_view_direction_log_likelihood(
    atlas: MapletFeatureAtlasBank,
    selected_maplet_ids: np.ndarray,
    pose_w2c: np.ndarray,
    *,
    angular_std_floor_deg: float = VIEW_DIRECTION_ANGULAR_STD_FLOOR_DEG,
) -> float:
    """Score camera direction under anonymous per-texel map view modes.

    The baked atlas already stores a mean direction and covariance for each
    appearance mode.  Treat those as a two-dimensional tangent-plane mixture
    instead of the historical arbitrary ``2*cos(theta)`` prior.  Cells are
    averaged within charts and charts are averaged equally, so map raster area
    and repeated texels cannot become extra pose evidence.
    """

    if atlas.mode_view_directions is None or atlas.mode_view_covariance is None:
        return 0.0
    id_to_row = {
        int(value): row
        for row, value in enumerate(atlas.maplet_ids.tolist())
    }
    camera_center = (
        -np.asarray(pose_w2c[:3, :3], dtype=np.float64).T
        @ np.asarray(pose_w2c[:3, 3], dtype=np.float64)
    )
    chart_scores = []
    for maplet_id in np.asarray(
        selected_maplet_ids, dtype=np.int64
    ).tolist():
        if int(maplet_id) not in id_to_row:
            continue
        row = id_to_row[int(maplet_id)]
        mode_valid = np.asarray(
            atlas.mode_valid_mask[row], dtype=bool
        )
        cell_mask = np.asarray(atlas.valid_mask[row], dtype=bool) & np.any(
            mode_valid, axis=0
        )
        if not np.any(cell_mask):
            continue
        position = np.asarray(
            atlas.xyz[row][cell_mask], dtype=np.float64
        )
        query_direction = camera_center[None] - position
        query_direction /= np.maximum(
            np.linalg.norm(query_direction, axis=1, keepdims=True), 1e-8
        )
        means = np.asarray(
            atlas.mode_view_directions[row][:, cell_mask],
            dtype=np.float64,
        )
        covariance = np.asarray(
            atlas.mode_view_covariance[row][:, cell_mask],
            dtype=np.float64,
        )
        valid = mode_valid[:, cell_mask]
        log_probability = _view_mode_log_densities(
            means,
            covariance,
            np.asarray(
                atlas.mode_weights[row][:, cell_mask], dtype=np.float64
            ),
            valid,
            query_direction,
            angular_std_floor_deg=float(angular_std_floor_deg),
        )
        maximum = np.max(log_probability, axis=0)
        cell_score = maximum + np.log(
            np.sum(np.exp(log_probability - maximum[None]), axis=0)
        )
        chart_scores.append(float(np.mean(cell_score)))
    return (
        float(np.mean(chart_scores))
        if chart_scores
        else float("-inf")
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
    mode_feature: np.ndarray | None = None
    mode_log_prior: np.ndarray | None = None
    surface_id: np.ndarray | None = None
    primitive_id: np.ndarray | None = None
    child_id: np.ndarray | None = None
    atlas_xy: np.ndarray | None = None
    # Optional exact-surface evidence.  Legacy atlas renderers leave these
    # unset; the Goal-Maplet full-scene renderer fills them from the very same
    # compositing pass used for ``feature``.  Keeping them pose-conditioned and
    # ephemeral avoids introducing a second stored map representation.
    visibility: np.ndarray | None = None
    field_missing: np.ndarray | None = None
    incidence: np.ndarray | None = None
    projected_scale: np.ndarray | None = None


def render_sampled_maplet_atlases(
    atlas: MapletFeatureAtlasBank,
    selected_maplet_ids: np.ndarray,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    *,
    width: int,
    height: int,
    maximum_samples: int = 192,
    minimum_depth: float = 0.01,
    minimum_mode_view_direction_cosine: float | None = (
        MINIMUM_MODE_VIEW_DIRECTION_COSINE
    ),
) -> RenderedMapletAtlases:
    """Fast point-sampled atlas render for candidate broad screening.

    The final likelihood and every accepted pose update still use the full
    perspective-correct area renderer below.  A broad screen only needs a
    conservative feature observation from every requested chart; rasterizing
    all 32x32 chart triangles for thousands of Stage-B poses wastes minutes
    on candidates that will immediately be discarded.  This function takes a
    deterministic, chart-balanced subset of canonical atlas texels, projects
    them in one vectorized operation and keeps the nearest sample per feature
    cell.  It retains anonymous appearance modes and their view-conditioned
    priors and never accesses an image-side map artifact.
    """

    sample_budget = max(int(maximum_samples), 1)
    output_feature = np.zeros(
        (atlas.feature_dim, int(height), int(width)), dtype=np.float32
    )
    output_xyz = np.zeros((int(height), int(width), 3), dtype=np.float32)
    output_normal = np.zeros(
        (int(height), int(width), 3), dtype=np.float32
    )
    output_uncertainty = np.ones(
        (int(height), int(width)), dtype=np.float32
    )
    output_maplet = np.full(
        (int(height), int(width)), -1, dtype=np.int64
    )
    output_depth = np.full(
        (int(height), int(width)), np.inf, dtype=np.float32
    )
    output_surface_id = np.full(
        (int(height), int(width)), -1, dtype=np.int64
    )
    output_primitive_id = np.full(
        (int(height), int(width)), -1, dtype=np.int64
    )
    output_atlas_xy = np.full(
        (int(height), int(width), 2), -1, dtype=np.int32
    )
    output_mode_feature = (
        np.zeros(
            (
                atlas.appearance_mode_count,
                atlas.feature_dim,
                int(height),
                int(width),
            ),
            dtype=np.float32,
        )
        if atlas.mode_features is not None
        else None
    )
    output_mode_log_prior = (
        np.full(
            (atlas.appearance_mode_count, int(height), int(width)),
            -np.inf,
            dtype=np.float32,
        )
        if atlas.mode_features is not None
        else None
    )

    id_to_row = {
        int(value): int(row)
        for row, value in enumerate(atlas.maplet_ids.tolist())
    }
    selected_rows = []
    for value in np.asarray(selected_maplet_ids, dtype=np.int64).tolist():
        row = id_to_row.get(int(value))
        if row is not None and row not in selected_rows:
            selected_rows.append(row)
    if not selected_rows:
        output_depth.fill(0.0)
        return RenderedMapletAtlases(
            feature=output_feature,
            xyz=output_xyz,
            normal=output_normal,
            uncertainty=output_uncertainty,
            maplet_id=output_maplet,
            mask=np.zeros((int(height), int(width)), dtype=bool),
            depth=output_depth,
            mode_feature=output_mode_feature,
            mode_log_prior=output_mode_log_prior,
            surface_id=output_surface_id,
            primitive_id=output_primitive_id,
            atlas_xy=output_atlas_xy,
        )

    per_chart = max(
        sample_budget // max(len(selected_rows), 1), 1
    )
    rows_out = []
    local_out = []
    # Give every chart an equal deterministic quota. Any remainder is filled
    # in query-ranked chart order without allowing a large facade to dominate.
    for row in selected_rows:
        valid = np.flatnonzero(
            np.asarray(atlas.valid_mask[row], dtype=bool).reshape(-1)
            & (
                np.asarray(atlas.support_count[row], dtype=np.int64).reshape(-1)
                > 0
            )
        )
        if valid.size == 0:
            continue
        count = min(int(per_chart), int(valid.size))
        choice = np.linspace(0, valid.size - 1, count, dtype=np.int64)
        rows_out.extend([int(row)] * count)
        local_out.extend(valid[choice].tolist())
    if len(rows_out) < sample_budget:
        selected_keys = set(zip(rows_out, local_out))
        for row in selected_rows:
            valid = np.flatnonzero(
                np.asarray(atlas.valid_mask[row], dtype=bool).reshape(-1)
                & (
                    np.asarray(
                        atlas.support_count[row], dtype=np.int64
                    ).reshape(-1)
                    > 0
                )
            )
            for local in valid.tolist():
                key = (int(row), int(local))
                if key in selected_keys:
                    continue
                rows_out.append(key[0])
                local_out.append(key[1])
                selected_keys.add(key)
                if len(rows_out) >= sample_budget:
                    break
            if len(rows_out) >= sample_budget:
                break
    if not rows_out:
        output_depth.fill(0.0)
        return RenderedMapletAtlases(
            feature=output_feature,
            xyz=output_xyz,
            normal=output_normal,
            uncertainty=output_uncertainty,
            maplet_id=output_maplet,
            mask=np.zeros((int(height), int(width)), dtype=bool),
            depth=output_depth,
            mode_feature=output_mode_feature,
            mode_log_prior=output_mode_log_prior,
            surface_id=output_surface_id,
            primitive_id=output_primitive_id,
            atlas_xy=output_atlas_xy,
        )

    maplet_rows = np.asarray(rows_out[:sample_budget], dtype=np.int64)
    local = np.asarray(local_out[:sample_budget], dtype=np.int64)
    yy = local // int(atlas.width)
    xx = local % int(atlas.width)
    world = np.asarray(
        atlas.xyz[maplet_rows, yy, xx], dtype=np.float32
    )
    camera_center = (
        -np.asarray(pose_w2c[:3, :3], dtype=np.float64).T
        @ np.asarray(pose_w2c[:3, 3], dtype=np.float64)
    )
    view_direction = camera_center[None] - np.asarray(world, dtype=np.float64)
    view_direction /= np.maximum(
        np.linalg.norm(view_direction, axis=1, keepdims=True), 1e-8
    )
    mode_log_density = None
    if atlas.mode_features is not None:
        means = np.asarray(
            atlas.mode_view_directions[maplet_rows, :, yy, xx],
            dtype=np.float64,
        ).transpose(1, 0, 2)
        covariance = np.asarray(
            atlas.mode_view_covariance[maplet_rows, :, yy, xx],
            dtype=np.float64,
        ).transpose(1, 0, 2, 3)
        weights = np.asarray(
            atlas.mode_weights[maplet_rows, :, yy, xx], dtype=np.float64
        ).T
        mode_valid = np.asarray(
            atlas.mode_valid_mask[maplet_rows, :, yy, xx], dtype=bool
        ).T
        cosine = np.einsum("kni,ni->kn", means, view_direction)
        supported = np.any(mode_valid, axis=0)
        if minimum_mode_view_direction_cosine is not None:
            supported &= np.max(
                np.where(mode_valid, cosine, -np.inf), axis=0
            ) >= float(minimum_mode_view_direction_cosine)
        mode_log_density = _view_mode_log_densities(
            means,
            covariance,
            weights,
            mode_valid,
            view_direction,
        )
    else:
        supported = np.ones(world.shape[0], dtype=bool)

    pixels, depth = project_world_points(world, pose_w2c, camera)
    pixels = _scaled_pixels(pixels, camera, int(width), int(height))
    finite = (
        supported
        & np.isfinite(depth)
        & (depth > float(minimum_depth))
        & np.all(np.isfinite(pixels), axis=1)
        & (pixels[:, 0] >= -0.5)
        & (pixels[:, 0] < int(width) - 0.5)
        & (pixels[:, 1] >= -0.5)
        & (pixels[:, 1] < int(height) - 0.5)
    )
    sample_x = np.zeros(world.shape[0], dtype=np.int64)
    sample_y = np.zeros(world.shape[0], dtype=np.int64)
    sample_x[finite] = np.rint(pixels[finite, 0]).astype(np.int64)
    sample_y[finite] = np.rint(pixels[finite, 1]).astype(np.int64)
    candidates = np.flatnonzero(finite)
    if candidates.size:
        linear = sample_y[candidates] * int(width) + sample_x[candidates]
        order = np.lexsort((depth[candidates], linear))
        ordered = candidates[order]
        ordered_linear = linear[order]
        keep = np.r_[True, ordered_linear[1:] != ordered_linear[:-1]]
        chosen = ordered[keep]
        px = sample_x[chosen]
        py = sample_y[chosen]
        output_depth[py, px] = depth[chosen].astype(np.float32)
        output_xyz[py, px] = world[chosen]
        output_maplet[py, px] = atlas.maplet_ids[maplet_rows[chosen]]
        output_surface_id[py, px] = (
            maplet_rows[chosen] * int(atlas.height) * int(atlas.width)
            + local[chosen]
        )
        output_primitive_id[py, px] = atlas.primitive_ids[
            maplet_rows[chosen], yy[chosen], xx[chosen]
        ]
        output_atlas_xy[py, px] = np.stack(
            [xx[chosen], yy[chosen]], axis=1
        ).astype(np.int32)
        output_feature[:, py, px] = np.asarray(
            atlas.features[maplet_rows[chosen], :, yy[chosen], xx[chosen]],
            dtype=np.float32,
        ).T
        output_uncertainty[py, px] = np.asarray(
            atlas.variance[maplet_rows[chosen], yy[chosen], xx[chosen]],
            dtype=np.float32,
        )
        if output_mode_feature is not None and mode_log_density is not None:
            output_mode_feature[:, :, py, px] = np.asarray(
                atlas.mode_features[
                    maplet_rows[chosen], :, :, yy[chosen], xx[chosen]
                ],
                dtype=np.float32,
            ).transpose(1, 2, 0)
            selected_log_density = mode_log_density[:, chosen]
            maximum = np.max(selected_log_density, axis=0)
            normalizer = maximum + np.log(
                np.sum(
                    np.exp(selected_log_density - maximum[None]), axis=0
                )
            )
            output_mode_log_prior[:, py, px] = (
                selected_log_density - normalizer[None]
            ).astype(np.float32)
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
        mode_feature=output_mode_feature,
        mode_log_prior=output_mode_log_prior,
        surface_id=output_surface_id,
        primitive_id=output_primitive_id,
        atlas_xy=output_atlas_xy,
    )


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


def atlas_scene_geometry_points(
    atlas: MapletFeatureAtlasBank,
) -> np.ndarray:
    """Return the map-only geometry used for full-scene occlusion.

    Feature atlases are rendered only for retrieved charts, but geometry from
    the other charts still has to occlude them.  Otherwise a selected rear
    facade remains visible through the actual front surface and contributes
    a pose-dependent false match.  The points below contain no RGB or mapping
    image identity: they are canonical samples of the stored 2DGS charts.
    """

    valid = (
        np.asarray(atlas.valid_mask, dtype=bool)
        & (np.asarray(atlas.primitive_ids, dtype=np.int64) >= 0)
    )
    points = np.asarray(atlas.xyz[valid], dtype=np.float32).reshape(-1, 3)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    finite = np.all(np.isfinite(points), axis=1)
    return np.ascontiguousarray(points[finite], dtype=np.float32)


def render_scene_depth_from_points(
    xyz_world: np.ndarray,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    *,
    width: int,
    height: int,
    minimum_depth: float = 0.01,
    oversample: int | None = None,
) -> np.ndarray:
    """Project canonical 2DGS samples into a ray-local feature z-buffer.

    A direct minimum over an entire stride-16 cell is too conservative: a
    foreground point near one corner can incorrectly hide a selected surface
    near the opposite corner.  Build depth at approximately original
    stride-4 resolution, then sample only the central subcells of each feature
    cell.  This retains occlusion while avoiding cell-wide false shadows.
    """

    if oversample is None:
        resolved_oversample = int(
            np.clip(
                np.rint(
                    int(camera.width)
                    / max(4.0 * int(width), 1.0)
                ),
                1,
                4,
            )
        )
    else:
        resolved_oversample = int(oversample)
    if resolved_oversample <= 0:
        raise ValueError("scene-depth oversample must be positive")
    high_width = int(width) * resolved_oversample
    high_height = int(height) * resolved_oversample
    output = np.full(
        (high_height, high_width), np.inf, dtype=np.float32
    )
    points = np.asarray(xyz_world, dtype=np.float32).reshape(-1, 3)
    if points.size == 0:
        return np.zeros((int(height), int(width)), dtype=np.float32)
    pixels, depth = project_world_points(points, pose_w2c, camera)
    pixels = _scaled_pixels(
        pixels, camera, high_width, high_height
    )
    finite_pixels = (
        np.all(np.isfinite(pixels), axis=1)
        & (pixels[:, 0] >= -0.5)
        & (pixels[:, 0] < high_width - 0.5)
        & (pixels[:, 1] >= -0.5)
        & (pixels[:, 1] < high_height - 0.5)
    )
    x = np.zeros((pixels.shape[0],), dtype=np.int64)
    y = np.zeros((pixels.shape[0],), dtype=np.int64)
    x[finite_pixels] = np.rint(pixels[finite_pixels, 0]).astype(np.int64)
    y[finite_pixels] = np.rint(pixels[finite_pixels, 1]).astype(np.int64)
    valid = (
        np.isfinite(depth)
        & finite_pixels
        & (depth > float(minimum_depth))
    )
    if np.any(valid):
        linear = y[valid] * high_width + x[valid]
        np.minimum.at(output.reshape(-1), linear, depth[valid])
    if resolved_oversample == 1:
        output[~np.isfinite(output)] = 0.0
        return output
    cells = output.reshape(
        int(height),
        resolved_oversample,
        int(width),
        resolved_oversample,
    )
    if resolved_oversample % 2 == 0:
        centre = np.asarray(
            [
                resolved_oversample // 2 - 1,
                resolved_oversample // 2,
            ],
            dtype=np.int64,
        )
    else:
        centre = np.asarray(
            [resolved_oversample // 2], dtype=np.int64
        )
    sampled = np.min(
        cells[:, centre][:, :, :, centre], axis=(1, 3)
    )
    sampled[~np.isfinite(sampled)] = 0.0
    return sampled.astype(np.float32, copy=False)


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


if njit is not None:

    @njit(cache=True)
    def _rasterize_projected_triangles(
        screen: np.ndarray,
        depth: np.ndarray,
        scene_depth: np.ndarray,
        use_scene_depth: bool,
        output_width: int,
        output_height: int,
        occlusion_epsilon: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compiled equivalent of the per-triangle area/z-buffer loop."""

        output_depth = np.full(
            (output_height, output_width), np.inf, dtype=np.float32
        )
        output_triangle = np.full(
            (output_height, output_width), -1, dtype=np.int64
        )
        output_weights = np.zeros(
            (output_height, output_width, 3), dtype=np.float32
        )
        for triangle_index in range(screen.shape[0]):
            triangle = screen[triangle_index]
            triangle_depth = depth[triangle_index]
            valid_triangle = True
            for corner in range(3):
                valid_triangle = valid_triangle and (
                    np.isfinite(triangle[corner, 0])
                    and np.isfinite(triangle[corner, 1])
                    and np.isfinite(triangle_depth[corner])
                    and triangle_depth[corner] > 0.01
                )
            if not valid_triangle:
                continue
            denominator = (
                (triangle[1, 1] - triangle[2, 1])
                * (triangle[0, 0] - triangle[2, 0])
                + (triangle[2, 0] - triangle[1, 0])
                * (triangle[0, 1] - triangle[2, 1])
            )
            if abs(denominator) < 1e-10:
                continue
            minimum_x = max(
                int(math.floor(np.min(triangle[:, 0]))), 0
            )
            maximum_x = min(
                int(math.ceil(np.max(triangle[:, 0]))), output_width - 1
            )
            minimum_y = max(
                int(math.floor(np.min(triangle[:, 1]))), 0
            )
            maximum_y = min(
                int(math.ceil(np.max(triangle[:, 1]))), output_height - 1
            )
            if maximum_x < minimum_x or maximum_y < minimum_y:
                continue
            any_inside = False
            for py in range(minimum_y, maximum_y + 1):
                for px in range(minimum_x, maximum_x + 1):
                    first = (
                        (triangle[1, 1] - triangle[2, 1])
                        * (px - triangle[2, 0])
                        + (triangle[2, 0] - triangle[1, 0])
                        * (py - triangle[2, 1])
                    ) / denominator
                    second = (
                        (triangle[2, 1] - triangle[0, 1])
                        * (px - triangle[2, 0])
                        + (triangle[0, 0] - triangle[2, 0])
                        * (py - triangle[2, 1])
                    ) / denominator
                    third = 1.0 - first - second
                    if first < -1e-6 or second < -1e-6 or third < -1e-6:
                        continue
                    any_inside = True
                    inverse_depth = (
                        first / triangle_depth[0]
                        + second / triangle_depth[1]
                        + third / triangle_depth[2]
                    )
                    if inverse_depth <= 1e-12:
                        continue
                    sample_depth = 1.0 / inverse_depth
                    if sample_depth >= output_depth[py, px]:
                        continue
                    if use_scene_depth:
                        reference = scene_depth[py, px]
                        if (
                            reference > 0.0
                            and sample_depth
                            > reference + occlusion_epsilon
                        ):
                            continue
                    first_weight = (first / triangle_depth[0]) / inverse_depth
                    second_weight = (second / triangle_depth[1]) / inverse_depth
                    third_weight = (third / triangle_depth[2]) / inverse_depth
                    output_depth[py, px] = sample_depth
                    output_triangle[py, px] = triangle_index
                    output_weights[py, px, 0] = first_weight
                    output_weights[py, px, 1] = second_weight
                    output_weights[py, px, 2] = third_weight
            if any_inside:
                continue
            centroid_x = np.mean(triangle[:, 0])
            centroid_y = np.mean(triangle[:, 1])
            if (
                centroid_x < -0.5
                or centroid_x >= output_width - 0.5
                or centroid_y < -0.5
                or centroid_y >= output_height - 0.5
            ):
                continue
            px = int(np.rint(centroid_x))
            py = int(np.rint(centroid_y))
            inverse_depth = (
                1.0 / triangle_depth[0]
                + 1.0 / triangle_depth[1]
                + 1.0 / triangle_depth[2]
            ) / 3.0
            if inverse_depth <= 1e-12:
                continue
            sample_depth = 1.0 / inverse_depth
            if sample_depth >= output_depth[py, px]:
                continue
            if use_scene_depth:
                reference = scene_depth[py, px]
                if (
                    reference > 0.0
                    and sample_depth > reference + occlusion_epsilon
                ):
                    continue
            output_depth[py, px] = sample_depth
            output_triangle[py, px] = triangle_index
            for corner in range(3):
                output_weights[py, px, corner] = (
                    (1.0 / triangle_depth[corner])
                    / (3.0 * inverse_depth)
                )
        return output_depth, output_triangle, output_weights


def render_selected_maplet_atlases_fast(
    atlas: MapletFeatureAtlasBank,
    selected_maplet_ids: np.ndarray,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    *,
    width: int,
    height: int,
    full_scene_depth: np.ndarray | None = None,
    occlusion_epsilon: float = 0.03,
    maximum_edge_stretch: float = 4.0,
    minimum_triangle_normal_cosine: float = 0.10,
    minimum_mode_view_direction_cosine: float | None = (
        MINIMUM_MODE_VIEW_DIRECTION_COSINE
    ),
) -> RenderedMapletAtlases:
    """Vectorized/compiled equivalent of the canonical area renderer.

    Geometry preparation, projection, and feature interpolation are batched;
    only the exact triangle z-buffer is a compiled loop.  If numba is not
    available, the reference renderer remains the compatibility fallback.
    """

    if njit is None:  # pragma: no cover
        return render_selected_maplet_atlases(
            atlas,
            selected_maplet_ids,
            pose_w2c,
            camera,
            width=width,
            height=height,
            full_scene_depth=full_scene_depth,
            occlusion_epsilon=occlusion_epsilon,
            maximum_edge_stretch=maximum_edge_stretch,
            minimum_triangle_normal_cosine=minimum_triangle_normal_cosine,
            minimum_mode_view_direction_cosine=(
                minimum_mode_view_direction_cosine
            ),
        )
    id_to_row = {
        int(value): int(row)
        for row, value in enumerate(atlas.maplet_ids.tolist())
    }
    selected_rows = []
    for value in np.asarray(selected_maplet_ids, dtype=np.int64).tolist():
        row = id_to_row.get(int(value))
        if row is not None and row not in selected_rows:
            selected_rows.append(row)
    output_shape = (int(height), int(width))
    mode_count = int(atlas.appearance_mode_count)
    output_mode_feature = (
        np.zeros(
            (mode_count, atlas.feature_dim, *output_shape),
            dtype=np.float32,
        )
        if atlas.mode_features is not None
        else None
    )
    output_mode_log_prior = (
        np.full((mode_count, *output_shape), -np.inf, dtype=np.float32)
        if atlas.mode_features is not None
        else None
    )
    if not selected_rows:
        return RenderedMapletAtlases(
            feature=np.zeros(
                (atlas.feature_dim, *output_shape), dtype=np.float32
            ),
            xyz=np.zeros((*output_shape, 3), dtype=np.float32),
            normal=np.zeros((*output_shape, 3), dtype=np.float32),
            uncertainty=np.ones(output_shape, dtype=np.float32),
            maplet_id=np.full(output_shape, -1, dtype=np.int64),
            mask=np.zeros(output_shape, dtype=bool),
            depth=np.zeros(output_shape, dtype=np.float32),
            mode_feature=output_mode_feature,
            mode_log_prior=output_mode_log_prior,
            surface_id=np.full(output_shape, -1, dtype=np.int64),
            primitive_id=np.full(output_shape, -1, dtype=np.int64),
            atlas_xy=np.full((*output_shape, 2), -1, dtype=np.int32),
        )

    rows = np.asarray(selected_rows, dtype=np.int64)
    chart_count = int(rows.size)
    cell_count = int(atlas.height * atlas.width)
    world_grid = np.asarray(atlas.xyz[rows], dtype=np.float32)
    camera_center = (
        -np.asarray(pose_w2c[:3, :3], dtype=np.float64).T
        @ np.asarray(pose_w2c[:3, 3], dtype=np.float64)
    )
    valid_grids = []
    feature_grids = []
    uncertainty_grids = []
    mode_density_grids = []
    for local_row, maplet_row in enumerate(rows.tolist()):
        valid = np.asarray(atlas.valid_mask[maplet_row], dtype=bool) & (
            np.asarray(atlas.support_count[maplet_row]) > 0
        )
        if atlas.mode_features is None:
            feature_grid = np.asarray(
                atlas.features[maplet_row], dtype=np.float32
            )
            uncertainty_grid = np.asarray(
                atlas.variance[maplet_row], dtype=np.float32
            )
            mode_log_density = None
        else:
            view_direction = (
                camera_center[None, None]
                - np.asarray(world_grid[local_row], dtype=np.float64)
            )
            view_direction /= np.maximum(
                np.linalg.norm(view_direction, axis=2, keepdims=True),
                1e-8,
            )
            direction_cosine = np.einsum(
                "khwc,hwc->khw",
                np.asarray(
                    atlas.mode_view_directions[maplet_row],
                    dtype=np.float64,
                ),
                view_direction,
            )
            direction_cosine = np.where(
                atlas.mode_valid_mask[maplet_row],
                direction_cosine,
                -np.inf,
            )
            if minimum_mode_view_direction_cosine is not None:
                valid &= np.max(direction_cosine, axis=0) >= float(
                    minimum_mode_view_direction_cosine
                )
            mode_log_density = _atlas_chart_view_mode_log_densities(
                atlas, maplet_row, view_direction
            )
            selected_mode = np.argmax(mode_log_density, axis=0)
            mode_feature_grid = np.transpose(
                np.asarray(
                    atlas.mode_features[maplet_row], dtype=np.float32
                ),
                (2, 3, 0, 1),
            )
            feature_grid = np.take_along_axis(
                mode_feature_grid,
                selected_mode[:, :, None, None],
                axis=2,
            )[:, :, 0, :].transpose(2, 0, 1)
            mode_uncertainty = np.asarray(
                atlas.mode_variance[maplet_row], dtype=np.float32
            ) + np.trace(
                np.asarray(
                    atlas.mode_view_covariance[maplet_row],
                    dtype=np.float32,
                ),
                axis1=-2,
                axis2=-1,
            ) / 3.0
            uncertainty_grid = np.take_along_axis(
                mode_uncertainty, selected_mode[None], axis=0
            )[0]
        valid_grids.append(valid)
        feature_grids.append(feature_grid)
        uncertainty_grids.append(uncertainty_grid)
        if mode_log_density is not None:
            mode_density_grids.append(mode_log_density)

    triangle_parts = []
    for local_row, valid in enumerate(valid_grids):
        quad_valid = (
            valid[:-1, :-1]
            & valid[:-1, 1:]
            & valid[1:, 1:]
            & valid[1:, :-1]
        )
        yy, xx = np.nonzero(quad_valid)
        if yy.size == 0:
            continue
        local_corners = np.stack(
            [
                yy * atlas.width + xx,
                yy * atlas.width + xx + 1,
                (yy + 1) * atlas.width + xx + 1,
                (yy + 1) * atlas.width + xx,
            ],
            axis=1,
        ).astype(np.int64)
        world = world_grid[local_row].reshape(-1, 3)[local_corners]
        expected_u = max(
            2.0 * float(atlas.extents[rows[local_row], 0]) / atlas.width,
            1e-5,
        )
        expected_v = max(
            2.0 * float(atlas.extents[rows[local_row], 1]) / atlas.height,
            1e-5,
        )
        edge_valid = (
            np.maximum(
                np.linalg.norm(world[:, 1] - world[:, 0], axis=1),
                np.linalg.norm(world[:, 2] - world[:, 3], axis=1),
            )
            <= float(maximum_edge_stretch) * expected_u
        ) & (
            np.maximum(
                np.linalg.norm(world[:, 3] - world[:, 0], axis=1),
                np.linalg.norm(world[:, 2] - world[:, 1], axis=1),
            )
            <= float(maximum_edge_stretch) * expected_v
        )
        geometric_normal = np.cross(
            world[:, 1] - world[:, 0], world[:, 3] - world[:, 0]
        )
        geometric_normal /= np.maximum(
            np.linalg.norm(geometric_normal, axis=1, keepdims=True), 1e-8
        )
        edge_valid &= (
            geometric_normal
            @ np.asarray(atlas.frames[rows[local_row], 2], dtype=np.float64)
        ) >= float(minimum_triangle_normal_cosine)
        local_corners = local_corners[edge_valid]
        if local_corners.size == 0:
            continue
        selected_corners = local_corners + local_row * cell_count
        triangle_parts.append(
            np.stack(
                [
                    selected_corners[:, (0, 1, 2)],
                    selected_corners[:, (0, 2, 3)],
                ],
                axis=1,
            ).reshape(-1, 3)
        )
    if not triangle_parts:
        return render_selected_maplet_atlases_fast(
            atlas,
            np.zeros((0,), dtype=np.int64),
            pose_w2c,
            camera,
            width=width,
            height=height,
        )
    triangle_corners = np.ascontiguousarray(
        np.concatenate(triangle_parts, axis=0), dtype=np.int64
    )
    world_flat = world_grid.reshape(-1, 3)
    pixels, depths = project_world_points(world_flat, pose_w2c, camera)
    pixels = _scaled_pixels(pixels, camera, int(width), int(height))
    screen_triangles = np.ascontiguousarray(
        pixels[triangle_corners], dtype=np.float64
    )
    depth_triangles = np.ascontiguousarray(
        depths[triangle_corners], dtype=np.float64
    )
    if full_scene_depth is None:
        scene_depth = np.zeros(output_shape, dtype=np.float32)
        use_scene_depth = False
    else:
        scene_depth = np.asarray(full_scene_depth, dtype=np.float32)
        if scene_depth.shape != output_shape:
            raise ValueError("full_scene_depth must match render resolution")
        scene_depth = np.ascontiguousarray(scene_depth)
        use_scene_depth = True
    output_depth, output_triangle, output_weights = (
        _rasterize_projected_triangles(
            screen_triangles,
            depth_triangles,
            scene_depth,
            use_scene_depth,
            int(width),
            int(height),
            float(occlusion_epsilon),
        )
    )
    mask = output_triangle >= 0
    output_depth[~mask] = 0.0
    output_feature = np.zeros(
        (atlas.feature_dim, *output_shape), dtype=np.float32
    )
    output_xyz = np.zeros((*output_shape, 3), dtype=np.float32)
    output_normal = np.zeros((*output_shape, 3), dtype=np.float32)
    output_uncertainty = np.ones(output_shape, dtype=np.float32)
    output_maplet = np.full(output_shape, -1, dtype=np.int64)
    output_surface_id = np.full(output_shape, -1, dtype=np.int64)
    output_primitive_id = np.full(output_shape, -1, dtype=np.int64)
    output_atlas_xy = np.full((*output_shape, 2), -1, dtype=np.int32)
    if np.any(mask):
        py, px = np.nonzero(mask)
        triangle_index = output_triangle[py, px]
        corners = triangle_corners[triangle_index]
        weights = np.asarray(output_weights[py, px], dtype=np.float32)
        output_xyz[py, px] = np.einsum(
            "ni,nij->nj", weights, world_flat[corners]
        )
        feature_flat = np.transpose(
            np.asarray(feature_grids, dtype=np.float32), (0, 2, 3, 1)
        ).reshape(-1, atlas.feature_dim)
        interpolated = np.einsum(
            "ni,nic->nc", weights, feature_flat[corners]
        )
        interpolated /= np.maximum(
            np.linalg.norm(interpolated, axis=1, keepdims=True), 1e-8
        )
        output_feature[:, py, px] = interpolated.T
        uncertainty_flat = np.asarray(
            uncertainty_grids, dtype=np.float32
        ).reshape(-1)
        output_uncertainty[py, px] = np.einsum(
            "ni,ni->n", weights, uncertainty_flat[corners]
        )
        dominant = corners[
            np.arange(corners.shape[0]), np.argmax(weights, axis=1)
        ]
        selected_row = dominant // cell_count
        local = dominant % cell_count
        maplet_row = rows[selected_row]
        yy = local // atlas.width
        xx = local % atlas.width
        output_maplet[py, px] = atlas.maplet_ids[maplet_row]
        output_surface_id[py, px] = maplet_row * cell_count + local
        output_primitive_id[py, px] = atlas.primitive_ids[
            maplet_row, yy, xx
        ]
        output_atlas_xy[py, px] = np.stack([xx, yy], axis=1).astype(
            np.int32
        )
        output_normal[py, px] = atlas.frames[maplet_row, 2]
        if output_mode_feature is not None:
            output_mode_feature[:, :, py, px] = np.asarray(
                atlas.mode_features[maplet_row, :, :, yy, xx],
                dtype=np.float32,
            ).transpose(1, 2, 0)
            density = np.asarray(mode_density_grids, dtype=np.float64)[
                selected_row, :, yy, xx
            ]
            maximum = np.max(density, axis=1)
            normalizer = maximum + np.log(
                np.sum(np.exp(density - maximum[:, None]), axis=1)
            )
            output_mode_log_prior[:, py, px] = (
                density - normalizer[:, None]
            ).T.astype(np.float32)
    return RenderedMapletAtlases(
        feature=output_feature,
        xyz=output_xyz,
        normal=output_normal,
        uncertainty=output_uncertainty,
        maplet_id=output_maplet,
        mask=mask,
        depth=output_depth,
        mode_feature=output_mode_feature,
        mode_log_prior=output_mode_log_prior,
        surface_id=output_surface_id,
        primitive_id=output_primitive_id,
        atlas_xy=output_atlas_xy,
    )


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
    maximum_edge_stretch: float = 4.0,
    minimum_triangle_normal_cosine: float = 0.10,
    minimum_mode_view_direction_cosine: float | None = (
        MINIMUM_MODE_VIEW_DIRECTION_COSINE
    ),
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
    output_surface_id = np.full(
        (int(height), int(width)), -1, dtype=np.int64
    )
    output_primitive_id = np.full(
        (int(height), int(width)), -1, dtype=np.int64
    )
    output_atlas_xy = np.full(
        (int(height), int(width), 2), -1, dtype=np.int32
    )
    output_mode_feature = (
        np.zeros(
            (
                atlas.appearance_mode_count,
                atlas.feature_dim,
                int(height),
                int(width),
            ),
            dtype=np.float32,
        )
        if atlas.mode_features is not None
        else None
    )
    output_mode_log_prior = (
        np.full(
            (atlas.appearance_mode_count, int(height), int(width)),
            -np.inf,
            dtype=np.float32,
        )
        if atlas.mode_features is not None
        else None
    )
    id_to_row = {
        int(value): int(row) for row, value in enumerate(atlas.maplet_ids.tolist())
    }
    selected_rows = [
        id_to_row[int(value)]
        for value in np.asarray(selected_maplet_ids, dtype=np.int64).tolist()
        if int(value) in id_to_row
    ]
    camera_center = (
        -np.asarray(pose_w2c[:3, :3], dtype=np.float64).T
        @ np.asarray(pose_w2c[:3, 3], dtype=np.float64)
    )
    triangles = ((0, 1, 2), (0, 2, 3))
    corner_offsets = ((0, 0), (0, 1), (1, 1), (1, 0))
    for maplet_row in selected_rows:
        valid = atlas.valid_mask[maplet_row] & (
            atlas.support_count[maplet_row] > 0
        )
        mode_log_density = None
        if (
            atlas.mode_features is not None
            and minimum_mode_view_direction_cosine is not None
        ):
            # A view-conditioned map feature is defined only on the camera
            # directions observed while baking the map.  Merely normalizing
            # logits across the available modes erases that absolute support
            # and lets a planar mirror pose render a facade from behind.  Use
            # a deliberately broad 75.5-degree cap; it rejects extrapolation
            # to the opposite side without requiring mapping RGB or image IDs.
            view_direction = (
                camera_center[None, None]
                - np.asarray(atlas.xyz[maplet_row], dtype=np.float64)
            )
            view_direction /= np.maximum(
                np.linalg.norm(view_direction, axis=2, keepdims=True),
                1e-8,
            )
            direction_cosine = np.einsum(
                "khwc,hwc->khw",
                np.asarray(
                    atlas.mode_view_directions[maplet_row],
                    dtype=np.float64,
                ),
                view_direction,
            )
            direction_cosine = np.where(
                atlas.mode_valid_mask[maplet_row],
                direction_cosine,
                -np.inf,
            )
            valid &= np.max(direction_cosine, axis=0) >= float(
                minimum_mode_view_direction_cosine
            )
        if atlas.mode_features is None:
            feature_grid = np.asarray(
                atlas.features[maplet_row], dtype=np.float32
            )
            uncertainty_grid = np.asarray(
                atlas.variance[maplet_row], dtype=np.float32
            )
        else:
            # The camera direction is constant for every triangle incident
            # on a texel.  Evaluate its anonymous appearance-mode density once
            # for the complete chart rather than once per repeated corner.
            view_direction = (
                camera_center[None, None]
                - np.asarray(atlas.xyz[maplet_row], dtype=np.float64)
            )
            view_direction /= np.maximum(
                np.linalg.norm(view_direction, axis=2, keepdims=True),
                1e-8,
            )
            mode_log_density = _atlas_chart_view_mode_log_densities(
                atlas, maplet_row, view_direction
            )
            selected_mode = np.argmax(mode_log_density, axis=0)
            mode_feature_grid = np.transpose(
                np.asarray(
                    atlas.mode_features[maplet_row], dtype=np.float32
                ),
                (2, 3, 0, 1),
            )
            feature_grid = np.take_along_axis(
                mode_feature_grid,
                selected_mode[:, :, None, None],
                axis=2,
            )[:, :, 0, :].transpose(2, 0, 1)
            mode_uncertainty = np.asarray(
                atlas.mode_variance[maplet_row], dtype=np.float32
            ) + np.trace(
                np.asarray(
                    atlas.mode_view_covariance[maplet_row],
                    dtype=np.float32,
                ),
                axis1=-2,
                axis2=-1,
            ) / 3.0
            uncertainty_grid = np.take_along_axis(
                mode_uncertainty,
                selected_mode[None],
                axis=0,
            )[0]
        for y in range(atlas.height - 1):
            for x in range(atlas.width - 1):
                corners = [(y + dy, x + dx) for dy, dx in corner_offsets]
                if not all(valid[yy, xx] for yy, xx in corners):
                    continue
                world = np.asarray(
                    [atlas.xyz[maplet_row, yy, xx] for yy, xx in corners],
                    dtype=np.float64,
                )
                expected_u = max(
                    2.0 * float(atlas.extents[maplet_row, 0]) / atlas.width,
                    1e-5,
                )
                expected_v = max(
                    2.0 * float(atlas.extents[maplet_row, 1]) / atlas.height,
                    1e-5,
                )
                if (
                    max(
                        np.linalg.norm(world[1] - world[0]),
                        np.linalg.norm(world[2] - world[3]),
                    )
                    > float(maximum_edge_stretch) * expected_u
                    or max(
                        np.linalg.norm(world[3] - world[0]),
                        np.linalg.norm(world[2] - world[1]),
                    )
                    > float(maximum_edge_stretch) * expected_v
                ):
                    # Adjacent chart cells on different depth layers must not
                    # be stitched into a fictitious surface triangle.
                    continue
                geometric_normal = np.cross(world[1] - world[0], world[3] - world[0])
                geometric_normal /= max(np.linalg.norm(geometric_normal), 1e-8)
                if float(
                    geometric_normal
                    @ np.asarray(atlas.frames[maplet_row, 2], dtype=np.float64)
                ) < float(minimum_triangle_normal_cosine):
                    # Signed test catches folded/inverted chart patches.
                    continue
                pixels, depth = project_world_points(world, pose_w2c, camera)
                if np.any(depth <= 0.01) or not np.all(np.isfinite(pixels)):
                    continue
                pixels = _scaled_pixels(pixels, camera, int(width), int(height))
                features = np.asarray(
                    [feature_grid[:, yy, xx] for yy, xx in corners],
                    dtype=np.float32,
                )
                uncertainties = np.asarray(
                    [uncertainty_grid[yy, xx] for yy, xx in corners],
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
                    dominant_corner = indices[np.argmax(weights, axis=1)]
                    for output_index, corner_index in enumerate(
                        dominant_corner.tolist()
                    ):
                        yy_corner, xx_corner = corners[int(corner_index)]
                        px = sample_x[output_index]
                        py = sample_y[output_index]
                        output_surface_id[py, px] = (
                            maplet_row * atlas.height * atlas.width
                            + yy_corner * atlas.width
                            + xx_corner
                        )
                        output_primitive_id[py, px] = atlas.primitive_ids[
                            maplet_row, yy_corner, xx_corner
                        ]
                        output_atlas_xy[py, px] = (xx_corner, yy_corner)
                    if output_mode_feature is not None:
                        # Mode identities are local to each texel and need not
                        # align across chart neighbours.  Use the dominant
                        # perspective corner, preserve all of its modes, and
                        # let correlation marginalise them with a view prior.
                        for output_index, corner_index in enumerate(
                            dominant_corner.tolist()
                        ):
                            yy_corner, xx_corner = corners[int(corner_index)]
                            logits = mode_log_density[
                                :, yy_corner, xx_corner
                            ]
                            finite_logits = np.isfinite(logits)
                            normalizer = np.log(
                                np.sum(
                                    np.exp(
                                        logits[finite_logits]
                                        - np.max(logits[finite_logits])
                                    )
                                )
                            ) + np.max(logits[finite_logits])
                            px = sample_x[output_index]
                            py = sample_y[output_index]
                            output_mode_feature[:, :, py, px] = (
                                atlas.mode_features[
                                    maplet_row, :, :, yy_corner, xx_corner
                                ]
                            )
                            output_mode_log_prior[:, py, px] = (
                                logits - normalizer
                            )
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
        mode_feature=output_mode_feature,
        mode_log_prior=output_mode_log_prior,
        surface_id=output_surface_id,
        primitive_id=output_primitive_id,
        atlas_xy=output_atlas_xy,
    )
