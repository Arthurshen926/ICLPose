"""Exact clean-2DGS canonical-feature rendering for continuous alignment."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from feature_extract.vfm.colmap_tracks import colmap_camera_focal_lengths
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMFeatureView
from feature_extract.vfm.localization_v6.atlas_renderer import RenderedMapletAtlases
from feature_extract.vfm.vfm_2dgs_mapping import SurfaceElementMap, _render_surface_element_pixel_contributions_2dgs

from .canonical_field import CanonicalSurfaceField
from .oracle_pose import intersect_rays_with_primitive_planes
from .physical_map import GoalMapletPhysicalMap
from .retrieval_surface_metrics import inverse_simple_radial
from .visibility import dominant_maplet_owner, signed_surface_visibility


@dataclass(frozen=True)
class RenderedSurfaceIdentity:
    primitive_rows: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class RenderedSoftChildMixture:
    """Top-L child alpha and typed residual mass per token.

    Identity mass is an exclusive partition:
    ``top_l + child_tail + unassigned_geometry + background == 1``.
    Canonical-field absence and payload exclusion are separate feature-side
    channels and never change child identity.
    """

    child_rows: np.ndarray
    child_weights: np.ndarray
    child_features: np.ndarray
    child_feature_valid: np.ndarray
    parent_rows: np.ndarray
    parent_weights: np.ndarray
    parent_tail_weight: np.ndarray
    child_tail_weight: np.ndarray
    unassigned_geometry_weight: np.ndarray
    background_weight: np.ndarray
    canonical_field_missing_weight: np.ndarray
    payload_excluded_weight: np.ndarray
    null_weight: np.ndarray
    total_alpha: np.ndarray
    maximum_alpha_overflow: float
    overflow_token_fraction: float


@dataclass(frozen=True)
class _SoftChildMassPartition:
    total_alpha: np.ndarray
    child_tail_weight: np.ndarray
    unassigned_geometry_weight: np.ndarray
    background_weight: np.ndarray
    canonical_field_missing_weight: np.ndarray
    payload_excluded_weight: np.ndarray
    null_weight: np.ndarray
    maximum_alpha_overflow: float
    overflow_token_fraction: float


@dataclass(frozen=True)
class _RawToIdealTokenWarp:
    source_ideal_pixel_ids: np.ndarray
    destination_token_pixel_ids: np.ndarray


_RAW_TO_IDEAL_TOKEN_WARP_CACHE: dict[tuple[object, ...], _RawToIdealTokenWarp] = {}


def _camera_warp_cache_key(
    camera, *, token_width: int, token_height: int, supersample_factor: int,
) -> tuple[object, ...]:
    return (
        int(camera.model_id), int(camera.width), int(camera.height),
        tuple(float(value) for value in np.asarray(camera.params, dtype=np.float64)),
        int(token_width), int(token_height), int(supersample_factor),
    )


def _raw_to_ideal_token_warp(
    camera, *, token_width: int, token_height: int, supersample_factor: int,
) -> _RawToIdealTokenWarp:
    """Cache the pose-independent raw-ray to ideal-pixel/token mapping."""

    key = _camera_warp_cache_key(
        camera, token_width=int(token_width), token_height=int(token_height),
        supersample_factor=int(supersample_factor),
    )
    cached = _RAW_TO_IDEAL_TOKEN_WARP_CACHE.get(key)
    if cached is not None:
        return cached
    factor = int(supersample_factor)
    if factor <= 0:
        raise ValueError("coordinate supersample factor must be positive")
    render_width = int(token_width) * factor
    render_height = int(token_height) * factor
    params = np.asarray(camera.params, dtype=np.float64).reshape(-1)
    if int(camera.model_id) == 0 and params.size >= 3:
        fx, cx, cy = params[:3]
        fy, k1 = fx, 0.0
    elif int(camera.model_id) == 1 and params.size >= 4:
        fx, fy, cx, cy = params[:4]
        k1 = 0.0
    elif int(camera.model_id) == 2 and params.size >= 4:
        fx, cx, cy, k1 = params[:4]
        fy = fx
    else:
        raise ValueError(
            "soft renderer requires SIMPLE_PINHOLE, PINHOLE, or SIMPLE_RADIAL camera"
        )
    scale_x = render_width / float(camera.width)
    scale_y = render_height / float(camera.height)
    if float(fx) <= 0.0 or float(fy) <= 0.0:
        raise ValueError("soft renderer camera/grid scale is invalid")
    fx_grid, fy_grid = fx * scale_x, fy * scale_y
    cx_grid, cy_grid = cx * scale_x, cy * scale_y
    yy, xx = np.meshgrid(
        np.arange(render_height, dtype=np.float64) + 0.5,
        np.arange(render_width, dtype=np.float64) + 0.5,
        indexing="ij",
    )
    distorted = np.stack(
        [(xx - cx_grid) / fx_grid, (yy - cy_grid) / fy_grid], axis=-1
    )
    undistorted = inverse_simple_radial(distorted, float(k1))
    ideal_x = np.floor(fx_grid * undistorted[..., 0] + cx_grid).astype(np.int64)
    ideal_y = np.floor(fy_grid * undistorted[..., 1] + cy_grid).astype(np.int64)
    valid = (
        (ideal_x >= 0) & (ideal_x < render_width)
        & (ideal_y >= 0) & (ideal_y < render_height)
    ).reshape(-1)
    raw_rows = np.flatnonzero(valid)
    source = (
        ideal_y.reshape(-1)[raw_rows] * render_width
        + ideal_x.reshape(-1)[raw_rows]
    ).astype(np.int64)
    raw_y, raw_x = raw_rows // render_width, raw_rows % render_width
    destination = (
        (raw_y // factor) * int(token_width) + raw_x // factor
    ).astype(np.int64)
    source.setflags(write=False)
    destination.setflags(write=False)
    result = _RawToIdealTokenWarp(source, destination)
    _RAW_TO_IDEAL_TOKEN_WARP_CACHE[key] = result
    return result


def _grouped_sum_sorted(
    keys: np.ndarray, values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum already-key-sorted rows with one deterministic reduction order."""

    key = np.asarray(keys, dtype=np.int64).reshape(-1)
    value = np.asarray(values)
    if value.shape[0] != key.size:
        raise ValueError("grouped reduction keys and values differ")
    if key.size == 0:
        return key, value[:0]
    if np.any(key[1:] < key[:-1]):
        raise ValueError("grouped reduction keys must be sorted")
    starts = np.r_[0, np.flatnonzero(key[1:] != key[:-1]) + 1]
    return key[starts], np.add.reduceat(value, starts, axis=0)


def _accumulate_feature_rows_sorted(
    destination: np.ndarray,
    slots: np.ndarray,
    contribution: np.ndarray,
    codes: np.ndarray,
    *,
    channel_block: int = 16,
) -> None:
    """Deterministically reduce weighted codes without a giant hit×D buffer."""

    slot = np.asarray(slots, dtype=np.int64).reshape(-1)
    weight = np.asarray(contribution, dtype=np.float32).reshape(-1)
    vector = np.asarray(codes, dtype=np.float32)
    if slot.shape != weight.shape or vector.shape != (slot.size, destination.shape[1]):
        raise ValueError("feature reduction arrays differ")
    if slot.size == 0:
        return
    if np.any(slot[1:] < slot[:-1]):
        raise ValueError("feature reduction slots must be sorted")
    starts = np.r_[0, np.flatnonzero(slot[1:] != slot[:-1]) + 1]
    unique = slot[starts]
    for begin in range(0, destination.shape[1], int(channel_block)):
        end = min(begin + int(channel_block), destination.shape[1])
        weighted = weight[:, None] * vector[:, begin:end]
        destination[unique, begin:end] += np.add.reduceat(weighted, starts, axis=0)


def _finalize_soft_child_mass_partition(
    *,
    total_alpha: np.ndarray,
    assigned_child_alpha: np.ndarray,
    retained_child_weights: np.ndarray,
    canonical_feature_alpha: np.ndarray,
    payload_feature_alpha: np.ndarray,
    alpha_conservation_tolerance: float,
) -> _SoftChildMassPartition:
    """Validate and finalize identity and feature-side mass channels.

    The function deliberately never renormalizes renderer output.  Numerical
    overflow inside the explicitly frozen tolerance is clipped only at the
    physical [0, 1] boundary and is reported; larger overflow or any ordering
    violation fails closed.
    """

    raw_total = np.asarray(total_alpha, dtype=np.float64).reshape(-1)
    assigned = np.asarray(assigned_child_alpha, dtype=np.float64).reshape(-1)
    retained_by_child = np.asarray(retained_child_weights, dtype=np.float64)
    canonical_by_child = np.asarray(canonical_feature_alpha, dtype=np.float64)
    payload_by_child = np.asarray(payload_feature_alpha, dtype=np.float64)
    if retained_by_child.ndim != 2:
        raise ValueError("retained child weights must have shape [token,top_l]")
    if (
        canonical_by_child.shape != retained_by_child.shape
        or payload_by_child.shape != retained_by_child.shape
        or raw_total.shape != assigned.shape
        or raw_total.size != retained_by_child.shape[0]
    ):
        raise ValueError("soft child mass arrays differ")
    tolerance = float(alpha_conservation_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("alpha_conservation_tolerance must be positive")
    arrays = (raw_total, assigned, retained_by_child, canonical_by_child, payload_by_child)
    if any(np.any(~np.isfinite(value)) for value in arrays):
        raise ValueError("soft child mass is nonfinite")
    if any(np.any(value < -tolerance) for value in arrays):
        raise ValueError("soft child mass is negative")

    retained = np.sum(retained_by_child, axis=1, dtype=np.float64)
    overflow = np.maximum(raw_total - 1.0, 0.0)
    if np.any(overflow > tolerance):
        raise ValueError(
            "rendered alpha overflow exceeds the transmittance-compositing tolerance"
        )
    if np.any(assigned > raw_total + tolerance):
        raise ValueError("assigned child mass exceeds rendered total alpha")
    if np.any(retained > assigned + tolerance):
        raise ValueError("retained child mass exceeds assigned child mass")
    if np.any(canonical_by_child > retained_by_child + tolerance):
        raise ValueError("canonical feature mass exceeds retained child mass")
    if np.any(payload_by_child > canonical_by_child + tolerance):
        raise ValueError("payload feature mass exceeds canonical feature mass")

    # Only epsilon-scale floating-point excursions are clipped.  Crucially,
    # no token-dependent scale factor is applied to any contribution.
    total = np.clip(raw_total, 0.0, 1.0)
    assigned = np.minimum(np.maximum(assigned, 0.0), total)
    retained_by_child = np.minimum(
        np.maximum(retained_by_child, 0.0), assigned[:, None]
    )
    retained = np.minimum(
        np.sum(retained_by_child, axis=1, dtype=np.float64), assigned
    )
    canonical_by_child = np.minimum(
        np.maximum(canonical_by_child, 0.0), retained_by_child
    )
    payload_by_child = np.minimum(
        np.maximum(payload_by_child, 0.0), canonical_by_child
    )

    child_tail = assigned - retained
    unassigned = total - assigned
    background = 1.0 - total
    canonical_missing = np.sum(
        retained_by_child - canonical_by_child, axis=1, dtype=np.float64
    )
    payload_excluded = np.sum(
        canonical_by_child - payload_by_child, axis=1, dtype=np.float64
    )
    null = child_tail + unassigned + background
    identity = retained + null
    if np.any(np.abs(identity - 1.0) > tolerance):
        raise ValueError("soft child identity partition does not conserve unit mass")
    return _SoftChildMassPartition(
        total_alpha=total.astype(np.float32),
        child_tail_weight=child_tail.astype(np.float32),
        unassigned_geometry_weight=unassigned.astype(np.float32),
        background_weight=background.astype(np.float32),
        canonical_field_missing_weight=canonical_missing.astype(np.float32),
        payload_excluded_weight=payload_excluded.astype(np.float32),
        null_weight=null.astype(np.float32),
        maximum_alpha_overflow=float(np.max(overflow, initial=0.0)),
        overflow_token_fraction=float(np.mean(overflow > 0.0)),
    )


def _deterministic_contribution_order(
    pixel_ids: np.ndarray,
    stable_primitive_ids: np.ndarray,
    contribution: np.ndarray,
) -> np.ndarray:
    """Canonical accumulation order independent of physical-map row order."""

    pixel = np.asarray(pixel_ids, dtype=np.int64).reshape(-1)
    primitive = np.asarray(stable_primitive_ids, dtype=np.int64).reshape(-1)
    weight = np.asarray(contribution, dtype=np.float32).reshape(-1)
    if pixel.shape != primitive.shape or pixel.shape != weight.shape:
        raise ValueError("surface contribution arrays differ")
    # ``pixel`` is the primary key, then persistent primitive identity.  The
    # weight key also canonicalizes the unlikely case of duplicate splats from
    # one primitive to one pixel.
    return np.lexsort((weight, primitive, pixel))


def _remap_ideal_hits_to_raw_tokens(
    pixel_ids: np.ndarray,
    local_rows: np.ndarray,
    contribution: np.ndarray,
    camera,
    *,
    token_width: int,
    token_height: int,
    supersample_factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather ideal-pinhole hit lists at raw SIMPLE_RADIAL sample rays.

    Rendering occurs on the high-resolution ideal grid.  Each raw distorted
    sample is inverse-warped to its nearest ideal pixel, exactly matching the
    contributor/RADIO coordinate contract, then averaged into its RADIO token.
    """

    factor = int(supersample_factor)
    warp = _raw_to_ideal_token_warp(
        camera, token_width=int(token_width), token_height=int(token_height),
        supersample_factor=factor,
    )
    source = warp.source_ideal_pixel_ids
    hits = np.asarray(pixel_ids, dtype=np.int64).reshape(-1)
    local = np.asarray(local_rows, dtype=np.int64).reshape(-1)
    weight = np.asarray(contribution, dtype=np.float32).reshape(-1)
    if hits.shape != local.shape or hits.shape != weight.shape:
        raise ValueError("ideal hit arrays differ")
    order = np.argsort(hits, kind="stable")
    sorted_hits = hits[order]
    left = np.searchsorted(sorted_hits, source, side="left")
    right = np.searchsorted(sorted_hits, source, side="right")
    nonempty = right > left
    left, right = left[nonempty], right[nonempty]
    destination = warp.destination_token_pixel_ids[nonempty]
    if destination.size == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )
    counts = right - left
    token_pixel = np.repeat(destination, counts)
    # Expand variable-length hit intervals without constructing one temporary
    # NumPy array per raw sample.  This is still an exact gather in the same
    # stable source-hit order.
    expanded_left = np.repeat(left, counts)
    group_origin = np.repeat(np.cumsum(counts) - counts, counts)
    selected_ranges = expanded_left + (
        np.arange(int(np.sum(counts)), dtype=np.int64) - group_origin
    )
    selected_hits = order[selected_ranges]
    return (
        token_pixel.astype(np.int64),
        local[selected_hits],
        (weight[selected_hits] / float(factor * factor)).astype(np.float32),
    )


def render_surface_identity(
    physical: GoalMapletPhysicalMap,
    pose_w2c: np.ndarray,
    camera,
    *,
    width: int,
    height: int,
    device: str = "cuda",
    minimum_incidence: float = 0.05,
    minimum_contribution: float = 1e-4,
) -> RenderedSurfaceIdentity:
    """Render only dominant physical identity; no feature field is touched."""

    front, _ = signed_surface_visibility(
        physical.primitive_centers,
        physical.primitive_normals,
        physical.primitive_sidedness,
        pose_w2c,
        minimum_incidence=float(minimum_incidence),
    )
    scene_rows = np.flatnonzero(front)
    elements = SurfaceElementMap(
        element_ids=physical.primitive_ids[scene_rows],
        parent_gaussian_indices=physical.primitive_ids[scene_rows],
        centers=physical.primitive_centers[scene_rows],
        tangent1=physical.primitive_tangent1[scene_rows],
        tangent2=physical.primitive_tangent2[scene_rows],
        normals=physical.primitive_normals[scene_rows],
        scale1=physical.primitive_scale1[scene_rows],
        scale2=physical.primitive_scale2[scene_rows],
        opacity=physical.primitive_opacity[scene_rows],
        area=np.pi * physical.primitive_scale1[scene_rows] * physical.primitive_scale2[scene_rows],
        adjacency=tuple(),
        metadata={"representation": "goal_maplet_full_clean_scene_identity_render"},
    )
    view = GaussianVFMFeatureView(
        image_id="goal_maplet_identity_render",
        feature_map=np.zeros((1, int(height), int(width)), dtype=np.float32),
        pose_w2c=np.asarray(pose_w2c, dtype=np.float64),
        camera=camera,
    )
    pixel_ids, local_rows, contribution, _ = _render_surface_element_pixel_contributions_2dgs(
        elements, view, width=int(width), height=int(height), device=str(device)
    )
    pixel_count = int(width) * int(height)
    primitive = np.full((pixel_count,), -1, dtype=np.int64)
    mask = np.zeros((pixel_count,), dtype=bool)
    valid = contribution >= float(minimum_contribution)
    if np.any(valid):
        selected = np.flatnonzero(valid)
        scene_primitive_rows = scene_rows[local_rows[selected]]
        order = np.lexsort((
            physical.primitive_ids[scene_primitive_rows],
            -contribution[selected],
            pixel_ids[selected],
        ))
        ordered = selected[order]
        ordered_pixels = pixel_ids[ordered]
        first = np.r_[True, ordered_pixels[1:] != ordered_pixels[:-1]]
        chosen = ordered[first]
        primitive[pixel_ids[chosen]] = scene_rows[local_rows[chosen]]
        mask[pixel_ids[chosen]] = True
    shape = (int(height), int(width))
    return RenderedSurfaceIdentity(primitive.reshape(shape), mask.reshape(shape))


_DOMINANT_CHILD_OWNER_CACHE: dict[str, np.ndarray] = {}
_PRIMITIVE_PARENT_MEMBERSHIP_CACHE: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def dominant_child_owner(physical: GoalMapletPhysicalMap) -> np.ndarray:
    cache_key = str(physical.content_sha256)
    cached = _DOMINANT_CHILD_OWNER_CACHE.get(cache_key)
    if cached is not None and cached.shape == (physical.primitive_ids.size,):
        return cached
    owner = np.full((physical.primitive_ids.size,), -1, dtype=np.int64)
    weight = np.full((physical.primitive_ids.size,), -np.inf, dtype=np.float32)
    for child in range(physical.child_parent_rows.size):
        start = int(physical.child_member_offsets[child])
        end = int(physical.child_member_offsets[child + 1])
        rows = physical.child_member_primitive_rows[start:end]
        values = physical.child_member_weights[start:end]
        replace = values > weight[rows]
        owner[rows[replace]] = child
        weight[rows[replace]] = values[replace]
    owner.setflags(write=False)
    _DOMINANT_CHILD_OWNER_CACHE[cache_key] = owner
    return owner


def _primitive_parent_memberships(
    physical: GoalMapletPhysicalMap,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized primitive→parent CSR without passing through children.

    Parent maplets overlap by construction.  A primitive's composited alpha is
    therefore split by its normalized physical membership weights, preserving
    unit mass while retaining overlapping parent support.  This hierarchy is
    independent of child Top-L truncation.
    """

    key = str(physical.content_sha256)
    cached = _PRIMITIVE_PARENT_MEMBERSHIP_CACHE.get(key)
    if cached is not None:
        return cached
    primitive_count = int(physical.primitive_ids.size)
    member_count = int(physical.membership_primitive_rows.size)
    parents = np.repeat(
        np.arange(physical.maplet_ids.size, dtype=np.int64),
        np.diff(np.asarray(physical.membership_offsets, dtype=np.int64)),
    )
    primitive = np.asarray(physical.membership_primitive_rows, dtype=np.int64)
    weight = np.asarray(physical.membership_weights, dtype=np.float64)
    if parents.shape != (member_count,):
        raise ValueError("physical parent membership offsets differ")
    order = np.lexsort((parents, primitive))
    primitive, parents, weight = primitive[order], parents[order], weight[order]
    totals = np.bincount(primitive, weights=weight, minlength=primitive_count)
    weight = weight / np.maximum(totals[primitive], 1e-12)
    offsets = np.zeros((primitive_count + 1,), dtype=np.int64)
    np.add.at(offsets, primitive + 1, 1)
    np.cumsum(offsets, out=offsets)
    for value in (offsets, parents, weight):
        value.setflags(write=False)
    result = offsets, parents, weight.astype(np.float32)
    result[2].setflags(write=False)
    _PRIMITIVE_PARENT_MEMBERSHIP_CACHE[key] = result
    return result


def _reduce_direct_parent_mass(
    physical: GoalMapletPhysicalMap,
    pixel_ids: np.ndarray,
    primitive_rows: np.ndarray,
    contribution: np.ndarray,
    *,
    pixel_count: int,
    top_l: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directly segment composited primitive alpha into parent maplets."""

    rows = np.full((int(pixel_count), int(top_l)), -1, dtype=np.int64)
    mass = np.zeros((int(pixel_count), int(top_l)), dtype=np.float32)
    tail = np.zeros((int(pixel_count),), dtype=np.float32)
    if not np.asarray(contribution).size:
        return rows, mass, tail
    offsets, parent_rows, membership = _primitive_parent_memberships(physical)
    primitive = np.asarray(primitive_rows, dtype=np.int64)
    counts = offsets[primitive + 1] - offsets[primitive]
    valid_hit = counts > 0
    if not np.any(valid_hit):
        return rows, mass, tail
    hit = np.flatnonzero(valid_hit)
    counts = counts[valid_hit]
    starts = np.repeat(offsets[primitive[hit]], counts)
    origin = np.repeat(np.cumsum(counts) - counts, counts)
    membership_row = starts + np.arange(int(np.sum(counts)), dtype=np.int64) - origin
    expanded_pixel = np.repeat(np.asarray(pixel_ids, dtype=np.int64)[hit], counts)
    expanded_parent = parent_rows[membership_row]
    expanded_mass = (
        np.repeat(np.asarray(contribution, dtype=np.float32)[hit], counts)
        * membership[membership_row]
    )
    parent_count = int(physical.maplet_ids.size)
    compound = expanded_pixel * parent_count + expanded_parent
    order = np.lexsort((expanded_mass, expanded_parent, expanded_pixel))
    unique, summed = _grouped_sum_sorted(compound[order], expanded_mass[order])
    unique_pixel, unique_parent = unique // parent_count, unique % parent_count
    rank = np.lexsort((unique_parent, -summed, unique_pixel))
    ranked_pixel = unique_pixel[rank]
    starts = np.r_[0, np.flatnonzero(ranked_pixel[1:] != ranked_pixel[:-1]) + 1]
    group_count = np.diff(np.r_[starts, rank.size])
    within = np.arange(rank.size) - np.repeat(starts, group_count)
    keep = within < int(top_l)
    selected = rank[keep]
    rows[unique_pixel[selected], within[keep]] = np.asarray(
        physical.maplet_ids, dtype=np.int64
    )[unique_parent[selected]]
    mass[unique_pixel[selected], within[keep]] = summed[selected]
    total_parent = np.bincount(unique_pixel, weights=summed, minlength=int(pixel_count))
    tail[:] = np.maximum(total_parent - np.sum(mass, axis=1, dtype=np.float64), 0.0)
    return rows, mass, tail


def _reduce_soft_child_token_hits(
    physical: GoalMapletPhysicalMap,
    field: CanonicalSurfaceField,
    *,
    token_pixel_ids: np.ndarray,
    primitive_rows: np.ndarray,
    contribution: np.ndarray,
    width: int,
    height: int,
    normalized_codes: np.ndarray,
    normalized_code_override_rows: np.ndarray | None = None,
    normalized_code_override_values: np.ndarray | None = None,
    selected_child_rows: np.ndarray | None,
    top_l: int,
    minimum_feature_alpha: float,
    alpha_conservation_tolerance: float,
    timing_sink: dict[str, float] | None = None,
) -> RenderedSoftChildMixture:
    """Shared deterministic CPU reference reduction for scalar and batch rasterizers."""

    reduction_started = time.perf_counter()
    feature_loop_seconds = 0.0
    pixel_ids = np.asarray(token_pixel_ids, dtype=np.int64).reshape(-1)
    scene_primitive_rows = np.asarray(primitive_rows, dtype=np.int64).reshape(-1)
    contribution = np.asarray(contribution, dtype=np.float32).reshape(-1)
    override_rows = np.asarray(
        [] if normalized_code_override_rows is None else normalized_code_override_rows,
        dtype=np.int64,
    ).reshape(-1)
    override_values = np.asarray(
        np.zeros((0, field.feature_dim), dtype=np.float32)
        if normalized_code_override_values is None
        else normalized_code_override_values,
        dtype=np.float32,
    )
    if (
        override_values.shape != (override_rows.size, field.feature_dim)
        or np.unique(override_rows).size != override_rows.size
        or np.any((override_rows < 0) | (override_rows >= field.primitive_rows.size))
        or np.any(~np.isfinite(override_values))
    ):
        raise ValueError("invalid sparse normalized-code overrides")
    if override_rows.size:
        override_order = np.argsort(override_rows, kind="stable")
        override_rows = override_rows[override_order]
        override_values = override_values[override_order]
    if pixel_ids.shape != scene_primitive_rows.shape or pixel_ids.shape != contribution.shape:
        raise ValueError("soft child token hit arrays differ")
    if np.any((scene_primitive_rows < 0) | (scene_primitive_rows >= physical.primitive_ids.size)):
        raise ValueError("soft child primitive rows are out of bounds")
    pixel_count = int(width) * int(height)
    if np.any((pixel_ids < 0) | (pixel_ids >= pixel_count)):
        raise ValueError("soft child token pixels are out of bounds")
    child_count = int(physical.child_parent_rows.size)
    mixture_rows = np.full((pixel_count, int(top_l)), -1, dtype=np.int64)
    mixture_weight = np.zeros((pixel_count, int(top_l)), dtype=np.float32)
    mixture_feature = np.zeros(
        (pixel_count, int(top_l), field.feature_dim), dtype=np.float32
    )
    mixture_canonical_alpha = np.zeros((pixel_count, int(top_l)), dtype=np.float32)
    mixture_payload_alpha = np.zeros((pixel_count, int(top_l)), dtype=np.float32)
    total_alpha = np.zeros((pixel_count,), dtype=np.float32)
    contribution_child = np.zeros((0,), dtype=np.int64)
    if contribution.size:
        stable_order = _deterministic_contribution_order(
            pixel_ids, physical.primitive_ids[scene_primitive_rows], contribution
        )
        total_pixels, total_values = _grouped_sum_sorted(
            pixel_ids[stable_order], contribution[stable_order].astype(np.float32)
        )
        total_alpha[total_pixels] = total_values
        child_owner = dominant_child_owner(physical)
        contribution_child = child_owner[scene_primitive_rows]
        valid_child = (
            (contribution_child >= 0)
            & (contribution_child < child_count)
            & (contribution > 0.0)
        )
        selected = np.flatnonzero(valid_child)
        if selected.size:
            key = (
                pixel_ids[selected].astype(np.int64) * child_count
                + contribution_child[selected].astype(np.int64)
            )
            order = np.lexsort((
                contribution[selected].astype(np.float32),
                physical.primitive_ids[scene_primitive_rows[selected]], key,
            ))
            ordered_key = key[order]
            starts = np.r_[0, np.flatnonzero(ordered_key[1:] != ordered_key[:-1]) + 1]
            unique_key = ordered_key[starts]
            unique_mass = np.add.reduceat(
                contribution[selected[order]].astype(np.float32), starts
            )
            unique_pixel = unique_key // child_count
            unique_child = unique_key % child_count
            rank_order = np.lexsort((unique_child, -unique_mass, unique_pixel))
            ranked_pixel = unique_pixel[rank_order]
            group_starts = np.r_[
                0, np.flatnonzero(ranked_pixel[1:] != ranked_pixel[:-1]) + 1
            ]
            group_count = np.diff(np.r_[group_starts, rank_order.size])
            within = np.arange(rank_order.size) - np.repeat(group_starts, group_count)
            keep = within < int(top_l)
            chosen = rank_order[keep]
            chosen_rank = within[keep]
            mixture_rows[unique_pixel[chosen], chosen_rank] = unique_child[chosen]
            mixture_weight[unique_pixel[chosen], chosen_rank] = unique_mass[chosen]

            chosen_flat = unique_pixel[chosen] * int(top_l) + chosen_rank
            chosen_key = unique_key[chosen]
            chosen_sort = np.argsort(chosen_key, kind="stable")
            sorted_key = chosen_key[chosen_sort]
            sorted_flat = chosen_flat[chosen_sort]
            field_row_by_scene = np.full(
                (physical.primitive_ids.size,), -1, dtype=np.int64
            )
            field_row_by_scene[field.primitive_rows] = np.arange(
                field.primitive_rows.size, dtype=np.int64
            )
            field_rows = field_row_by_scene[scene_primitive_rows]
            canonical_valid = field_rows >= 0
            payload_valid = canonical_valid.copy()
            if selected_child_rows is not None:
                payload = np.zeros((child_count,), dtype=bool)
                payload_rows = np.asarray(selected_child_rows, dtype=np.int64)
                if np.any((payload_rows < 0) | (payload_rows >= child_count)):
                    raise ValueError("selected child payload rows are out of bounds")
                payload[payload_rows] = True
                payload_valid &= payload[np.maximum(contribution_child, 0)] & (
                    contribution_child >= 0
                )
            for feature_mask, accumulate_payload in (
                (canonical_valid, False), (payload_valid, True),
            ):
                feature_started = time.perf_counter()
                feature_contribution = np.flatnonzero(feature_mask)
                if not feature_contribution.size:
                    feature_loop_seconds += time.perf_counter() - feature_started
                    continue
                feature_key = (
                    pixel_ids[feature_contribution].astype(np.int64) * child_count
                    + contribution_child[feature_contribution].astype(np.int64)
                )
                position = np.searchsorted(sorted_key, feature_key)
                present = position < sorted_key.size
                present[present] &= sorted_key[position[present]] == feature_key[present]
                feature_contribution = feature_contribution[present]
                slot = sorted_flat[position[present]]
                stable = physical.primitive_ids[scene_primitive_rows[feature_contribution]]
                canonical_order = np.lexsort((
                    contribution[feature_contribution].astype(np.float32), stable, slot,
                ))
                feature_contribution = feature_contribution[canonical_order]
                slot = slot[canonical_order]
                value = contribution[feature_contribution].astype(np.float32)
                unique_slot, alpha_sum = _grouped_sum_sorted(slot, value)
                if accumulate_payload:
                    code_rows = field_rows[feature_contribution]
                    feature_codes = normalized_codes[code_rows]
                    if override_rows.size:
                        override_position = np.searchsorted(override_rows, code_rows)
                        overridden = override_position < override_rows.size
                        overridden[overridden] &= (
                            override_rows[override_position[overridden]]
                            == code_rows[overridden]
                        )
                        if np.any(overridden):
                            feature_codes = feature_codes.copy()
                            feature_codes[overridden] = override_values[
                                override_position[overridden]
                            ]
                    _accumulate_feature_rows_sorted(
                        mixture_feature.reshape(-1, field.feature_dim), slot, value,
                        feature_codes,
                    )
                    mixture_payload_alpha.reshape(-1)[unique_slot] += alpha_sum
                else:
                    mixture_canonical_alpha.reshape(-1)[unique_slot] += alpha_sum
                feature_loop_seconds += time.perf_counter() - feature_started

    child_feature_finished = time.perf_counter()
    typed_started = child_feature_finished
    assigned_child = np.zeros((pixel_count,), dtype=np.float64)
    if contribution.size:
        assigned_mask = (
            (contribution_child >= 0)
            & (contribution_child < child_count)
            & (contribution > 0.0)
        )
        if np.any(assigned_mask):
            assigned_rows = np.flatnonzero(assigned_mask)
            assigned_order = _deterministic_contribution_order(
                pixel_ids[assigned_rows],
                physical.primitive_ids[scene_primitive_rows[assigned_rows]],
                contribution[assigned_rows],
            )
            assigned_pixels, assigned_values = _grouped_sum_sorted(
                pixel_ids[assigned_rows][assigned_order],
                contribution[assigned_rows][assigned_order].astype(np.float64),
            )
            assigned_child[assigned_pixels] = assigned_values
    mass = _finalize_soft_child_mass_partition(
        total_alpha=total_alpha,
        assigned_child_alpha=assigned_child,
        retained_child_weights=mixture_weight,
        canonical_feature_alpha=mixture_canonical_alpha,
        payload_feature_alpha=mixture_payload_alpha,
        alpha_conservation_tolerance=float(alpha_conservation_tolerance),
    )
    typed_seconds = time.perf_counter() - typed_started
    feature_finalize_started = time.perf_counter()
    feature_valid = mixture_payload_alpha >= float(minimum_feature_alpha)
    mixture_feature[feature_valid] /= np.maximum(
        mixture_payload_alpha[feature_valid, None], 1e-8
    )
    feature_norm = np.linalg.norm(mixture_feature, axis=2, keepdims=True)
    mixture_feature[feature_valid] /= np.maximum(feature_norm[feature_valid], 1e-8)
    feature_seconds = feature_loop_seconds + time.perf_counter() - feature_finalize_started
    shape = (int(height), int(width))
    parent_started = time.perf_counter()
    parent_rows, parent_weights, parent_tail = _reduce_direct_parent_mass(
        physical, pixel_ids, scene_primitive_rows, contribution,
        pixel_count=pixel_count, top_l=int(top_l),
    )
    parent_seconds = time.perf_counter() - parent_started
    if timing_sink is not None:
        timing_sink["child_identity_reduction_seconds"] = (
            child_feature_finished - reduction_started - feature_loop_seconds
        )
        timing_sink["feature_reduction_seconds"] = feature_seconds
        timing_sink["typed_finalize_seconds"] = typed_seconds
        timing_sink["direct_parent_reduction_seconds"] = parent_seconds
    return RenderedSoftChildMixture(
        child_rows=mixture_rows.reshape(*shape, int(top_l)),
        child_weights=mixture_weight.reshape(*shape, int(top_l)),
        child_features=mixture_feature.reshape(*shape, int(top_l), field.feature_dim),
        child_feature_valid=feature_valid.reshape(*shape, int(top_l)),
        parent_rows=parent_rows.reshape(*shape, int(top_l)),
        parent_weights=parent_weights.reshape(*shape, int(top_l)),
        parent_tail_weight=parent_tail.reshape(shape),
        child_tail_weight=mass.child_tail_weight.reshape(shape),
        unassigned_geometry_weight=mass.unassigned_geometry_weight.reshape(shape),
        background_weight=mass.background_weight.reshape(shape),
        canonical_field_missing_weight=mass.canonical_field_missing_weight.reshape(shape),
        payload_excluded_weight=mass.payload_excluded_weight.reshape(shape),
        null_weight=mass.null_weight.reshape(shape),
        total_alpha=mass.total_alpha.reshape(shape),
        maximum_alpha_overflow=mass.maximum_alpha_overflow,
        overflow_token_fraction=mass.overflow_token_fraction,
    )


def render_soft_child_surface_field(
    physical: GoalMapletPhysicalMap,
    field: CanonicalSurfaceField,
    pose_w2c: np.ndarray,
    camera,
    *,
    width: int,
    height: int,
    selected_child_rows: np.ndarray | None = None,
    feature_codes: np.ndarray | None = None,
    top_l: int = 4,
    device: str = "cuda",
    minimum_incidence: float = 0.05,
    minimum_feature_alpha: float = 1e-4,
    coordinate_supersample_factor: int = 4,
    alpha_conservation_tolerance: float = 2e-5,
) -> RenderedSoftChildMixture:
    """Render a true map-side soft child distribution with coupled features.

    All front-facing geometry participates in compositing/occlusion.  The
    area-selected payload only controls which child-specific canonical feature
    mixtures are materialized; it never changes the rendered child mass.
    """

    if field.physical_map_sha256 != physical.content_sha256:
        raise ValueError("canonical field and physical map lineage differ")
    if int(top_l) <= 0:
        raise ValueError("top_l must be positive")
    codes = field.codes if feature_codes is None else np.asarray(feature_codes, dtype=np.float32)
    if codes.shape != field.codes.shape:
        raise ValueError("feature_codes must align with the canonical field")
    codes = np.asarray(codes, dtype=np.float32)
    codes = codes / np.maximum(np.linalg.norm(codes, axis=1, keepdims=True), 1e-8)
    front, _ = signed_surface_visibility(
        physical.primitive_centers,
        physical.primitive_normals,
        physical.primitive_sidedness,
        pose_w2c,
        minimum_incidence=float(minimum_incidence),
    )
    scene_rows = np.flatnonzero(front)
    elements = SurfaceElementMap(
        element_ids=physical.primitive_ids[scene_rows],
        parent_gaussian_indices=physical.primitive_ids[scene_rows],
        centers=physical.primitive_centers[scene_rows],
        tangent1=physical.primitive_tangent1[scene_rows],
        tangent2=physical.primitive_tangent2[scene_rows],
        normals=physical.primitive_normals[scene_rows],
        scale1=physical.primitive_scale1[scene_rows],
        scale2=physical.primitive_scale2[scene_rows],
        opacity=physical.primitive_opacity[scene_rows],
        area=np.pi * physical.primitive_scale1[scene_rows] * physical.primitive_scale2[scene_rows],
        adjacency=tuple(),
        metadata={"representation": "goal_maplet_soft_child_feature_render_v2"},
    )
    factor = int(coordinate_supersample_factor)
    if factor <= 0:
        raise ValueError("coordinate_supersample_factor must be positive")
    render_width, render_height = int(width) * factor, int(height) * factor
    view = GaussianVFMFeatureView(
        image_id="goal_maplet_soft_child_render",
        feature_map=np.zeros((1, render_height, render_width), dtype=np.float32),
        pose_w2c=np.asarray(pose_w2c, dtype=np.float64),
        camera=camera,
    )
    pixel_ids, local_rows, contribution, _ = _render_surface_element_pixel_contributions_2dgs(
        elements, view, width=render_width, height=render_height, device=str(device)
    )
    pixel_ids, local_rows, contribution = _remap_ideal_hits_to_raw_tokens(
        pixel_ids, local_rows, contribution, camera,
        token_width=int(width), token_height=int(height),
        supersample_factor=factor,
    )
    return _reduce_soft_child_token_hits(
        physical, field,
        token_pixel_ids=pixel_ids,
        primitive_rows=(
            scene_rows[local_rows] if local_rows.size
            else np.zeros((0,), dtype=np.int64)
        ),
        contribution=contribution,
        width=int(width), height=int(height), normalized_codes=codes,
        selected_child_rows=selected_child_rows, top_l=int(top_l),
        minimum_feature_alpha=float(minimum_feature_alpha),
        alpha_conservation_tolerance=float(alpha_conservation_tolerance),
    )

def render_canonical_surface_field(
    physical: GoalMapletPhysicalMap,
    field: CanonicalSurfaceField,
    pose_w2c: np.ndarray,
    camera,
    *,
    width: int,
    height: int,
    selected_child_rows: np.ndarray | None = None,
    feature_codes: np.ndarray | None = None,
    device: str = "cuda",
    minimum_incidence: float = 0.05,
    minimum_feature_alpha: float = 1e-4,
    minimum_feature_fraction: float = 0.10,
    supersample_factor: int = 1,
) -> RenderedMapletAtlases:
    """Render one canonical feature field with full-scene 2DGS occlusion."""

    factor = int(supersample_factor)
    if factor <= 0:
        raise ValueError("supersample_factor must be positive")
    if factor > 1:
        high = render_canonical_surface_field(
            physical,
            field,
            pose_w2c,
            camera,
            width=int(width) * factor,
            height=int(height) * factor,
            selected_child_rows=selected_child_rows,
            feature_codes=feature_codes,
            device=str(device),
            minimum_incidence=float(minimum_incidence),
            minimum_feature_alpha=float(minimum_feature_alpha),
            minimum_feature_fraction=float(minimum_feature_fraction),
            supersample_factor=1,
        )
        return _pool_rendered_surface(high, factor)

    if field.physical_map_sha256 != physical.content_sha256:
        raise ValueError("canonical field and physical map lineage differ")
    codes = field.codes if feature_codes is None else np.asarray(feature_codes, dtype=np.float32)
    if codes.shape != field.codes.shape:
        raise ValueError("feature_codes must align with the canonical field")
    code_norm = np.linalg.norm(codes, axis=1, keepdims=True)
    codes = codes / np.maximum(code_norm, 1e-8)
    front, primitive_incidence = signed_surface_visibility(
        physical.primitive_centers,
        physical.primitive_normals,
        physical.primitive_sidedness,
        pose_w2c,
        minimum_incidence=float(minimum_incidence),
    )
    scene_rows = np.flatnonzero(front)
    elements = SurfaceElementMap(
        element_ids=physical.primitive_ids[scene_rows],
        parent_gaussian_indices=physical.primitive_ids[scene_rows],
        centers=physical.primitive_centers[scene_rows],
        tangent1=physical.primitive_tangent1[scene_rows],
        tangent2=physical.primitive_tangent2[scene_rows],
        normals=physical.primitive_normals[scene_rows],
        scale1=physical.primitive_scale1[scene_rows],
        scale2=physical.primitive_scale2[scene_rows],
        opacity=physical.primitive_opacity[scene_rows],
        area=np.pi * physical.primitive_scale1[scene_rows] * physical.primitive_scale2[scene_rows],
        adjacency=tuple(),
        metadata={"representation": "goal_maplet_full_clean_scene_canonical_feature_render"},
    )
    view = GaussianVFMFeatureView(
        image_id="goal_maplet_canonical_render",
        feature_map=np.zeros((1, int(height), int(width)), dtype=np.float32),
        pose_w2c=np.asarray(pose_w2c, dtype=np.float64),
        camera=camera,
    )
    pixel_ids, local_rows, contribution, _ = _render_surface_element_pixel_contributions_2dgs(
        elements, view, width=int(width), height=int(height), device=str(device)
    )
    pixel_count = int(width) * int(height)
    field_row_by_scene = np.full((physical.primitive_ids.size,), -1, dtype=np.int64)
    field_row_by_scene[field.primitive_rows] = np.arange(field.primitive_rows.size, dtype=np.int64)
    child_owner = dominant_child_owner(physical)
    selected_child = None if selected_child_rows is None else set(np.asarray(selected_child_rows, dtype=np.int64).tolist())
    scene_primitive_rows = scene_rows[local_rows] if local_rows.size else np.zeros((0,), dtype=np.int64)
    field_rows = field_row_by_scene[scene_primitive_rows] if local_rows.size else np.zeros((0,), dtype=np.int64)
    feature_valid = field_rows >= 0
    if selected_child is not None and feature_valid.size:
        feature_valid &= np.asarray(
            [int(child_owner[row]) in selected_child for row in scene_primitive_rows], dtype=bool
        )
    feature_sum = np.zeros((pixel_count, field.feature_dim), dtype=np.float32)
    feature_alpha = np.zeros((pixel_count,), dtype=np.float32)
    uncertainty_sum = np.zeros((pixel_count,), dtype=np.float32)
    if np.any(feature_valid):
        selected = np.flatnonzero(feature_valid)
        selected = selected[_deterministic_contribution_order(
            pixel_ids[selected],
            physical.primitive_ids[scene_primitive_rows[selected]],
            contribution[selected],
        )]
        for offset in range(0, selected.size, 250_000):
            rows = selected[offset : offset + 250_000]
            value = contribution[rows].astype(np.float32)
            np.add.at(feature_sum, pixel_ids[rows], value[:, None] * codes[field_rows[rows]])
            np.add.at(feature_alpha, pixel_ids[rows], value)
            np.add.at(uncertainty_sum, pixel_ids[rows], value * field.uncertainty[field_rows[rows]])
    total_alpha = np.zeros((pixel_count,), dtype=np.float32)
    if pixel_ids.size:
        accumulation_order = _deterministic_contribution_order(
            pixel_ids,
            physical.primitive_ids[scene_primitive_rows],
            contribution,
        )
        np.add.at(
            total_alpha,
            pixel_ids[accumulation_order],
            contribution[accumulation_order].astype(np.float32),
        )
    mask = (feature_alpha >= float(minimum_feature_alpha)) & (
        feature_alpha / np.maximum(total_alpha, 1e-8) >= float(minimum_feature_fraction)
    )
    visibility = total_alpha >= float(minimum_feature_alpha)
    field_missing = visibility & ~mask
    normalized_feature = np.zeros_like(feature_sum)
    normalized_feature[mask] = feature_sum[mask] / np.maximum(feature_alpha[mask, None], 1e-8)
    norm = np.linalg.norm(normalized_feature, axis=1, keepdims=True)
    normalized_feature[mask] /= np.maximum(norm[mask], 1e-8)
    uncertainty = np.ones((pixel_count,), dtype=np.float32)
    uncertainty[mask] = uncertainty_sum[mask] / np.maximum(feature_alpha[mask], 1e-8)
    dominant_scene_row = np.full((pixel_count,), -1, dtype=np.int64)
    # Geometry/null evidence must remain available even when the dominant
    # primitive has no canonical code.  Selecting identity only from
    # ``feature_valid`` silently collapsed field-missing into background.
    if contribution.size:
        selected = np.arange(contribution.size, dtype=np.int64)
        order = np.lexsort((
            physical.primitive_ids[scene_primitive_rows[selected]],
            -contribution[selected],
            pixel_ids[selected],
        ))
        ordered = selected[order]
        ordered_pixels = pixel_ids[ordered]
        first = np.r_[True, ordered_pixels[1:] != ordered_pixels[:-1]]
        chosen = ordered[first]
        dominant_scene_row[pixel_ids[chosen]] = scene_primitive_rows[chosen]
    valid_pixels = np.flatnonzero(visibility & (dominant_scene_row >= 0))
    xyz = np.zeros((pixel_count, 3), dtype=np.float32)
    normal = np.zeros((pixel_count, 3), dtype=np.float32)
    depth = np.zeros((pixel_count,), dtype=np.float32)
    maplet_id = np.full((pixel_count,), -1, dtype=np.int64)
    surface_id = np.full((pixel_count,), -1, dtype=np.int64)
    primitive_id = np.full((pixel_count,), -1, dtype=np.int64)
    child_id = np.full((pixel_count,), -1, dtype=np.int64)
    incidence = np.zeros((pixel_count,), dtype=np.float32)
    projected_scale = np.zeros((pixel_count,), dtype=np.float32)
    maplet_owner = dominant_maplet_owner(physical)
    if valid_pixels.size:
        px = valid_pixels % int(width)
        py = valid_pixels // int(width)
        xy_original = np.stack(
            [
                (px + 0.5) * float(camera.width) / float(width),
                (py + 0.5) * float(camera.height) / float(height),
            ],
            axis=1,
        ) - 0.5
        primitive_rows = dominant_scene_row[valid_pixels]
        point, intersection_valid = intersect_rays_with_primitive_planes(
            xy_original, primitive_rows, physical, pose_w2c, camera
        )
        invalid_pixels = valid_pixels[~intersection_valid]
        mask[invalid_pixels] = False
        visibility[invalid_pixels] = False
        field_missing[invalid_pixels] = False
        valid_pixels = valid_pixels[intersection_valid]
        primitive_rows = primitive_rows[intersection_valid]
        point = point[intersection_valid]
        xyz[valid_pixels] = point.astype(np.float32)
        normal[valid_pixels] = physical.primitive_normals[primitive_rows].astype(np.float32)
        pose = np.asarray(pose_w2c, dtype=np.float64)
        camera_xyz = point @ pose[:3, :3].T + pose[:3, 3]
        depth[valid_pixels] = camera_xyz[:, 2].astype(np.float32)
        owner_rows = maplet_owner[primitive_rows]
        owner_valid = owner_rows >= 0
        maplet_id[valid_pixels[owner_valid]] = physical.maplet_ids[owner_rows[owner_valid]]
        surface_id[valid_pixels] = primitive_rows
        primitive_id[valid_pixels] = physical.primitive_ids[primitive_rows]
        child_rows = child_owner[primitive_rows]
        child_valid = child_rows >= 0
        child_id[valid_pixels[child_valid]] = child_rows[child_valid]
        incidence[valid_pixels] = np.abs(primitive_incidence[primitive_rows]).astype(np.float32)
        fx, fy = colmap_camera_focal_lengths(camera)
        focal = 0.5 * (abs(float(fx)) + abs(float(fy)))
        primitive_radius = np.sqrt(
            np.maximum(
                physical.primitive_scale1[primitive_rows]
                * physical.primitive_scale2[primitive_rows],
                0.0,
            )
        )
        projected_scale[valid_pixels] = (
            primitive_radius * focal / np.maximum(camera_xyz[:, 2], 1.0e-6)
        ).astype(np.float32)
    shape = (int(height), int(width))
    return RenderedMapletAtlases(
        feature=normalized_feature.reshape(int(height), int(width), field.feature_dim).transpose(2, 0, 1),
        xyz=xyz.reshape(int(height), int(width), 3),
        normal=normal.reshape(int(height), int(width), 3),
        uncertainty=uncertainty.reshape(shape),
        maplet_id=maplet_id.reshape(shape),
        mask=mask.reshape(shape),
        depth=depth.reshape(shape),
        surface_id=surface_id.reshape(shape),
        primitive_id=primitive_id.reshape(shape),
        child_id=child_id.reshape(shape),
        visibility=visibility.reshape(shape),
        field_missing=field_missing.reshape(shape),
        incidence=incidence.reshape(shape),
        projected_scale=projected_scale.reshape(shape),
        feature_fraction=mask.reshape(shape).astype(np.float32),
        visibility_fraction=visibility.reshape(shape).astype(np.float32),
        missing_fraction=field_missing.reshape(shape).astype(np.float32),
        background_fraction=(~visibility.reshape(shape)).astype(np.float32),
        dominant_surface_fraction=visibility.reshape(shape).astype(np.float32),
        mixed_surface=np.zeros(shape, dtype=bool),
    )


def _dominant_subpixel_choice(
    component_blocks: np.ndarray,
    depth_blocks: np.ndarray,
    visibility_blocks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose frontmost sample of the dominant physical component per token."""

    component = np.asarray(component_blocks, dtype=np.int64)
    depth = np.asarray(depth_blocks, dtype=np.float32)
    visible = np.asarray(visibility_blocks, dtype=bool)
    if component.shape != depth.shape or component.shape != visible.shape or component.ndim != 4:
        raise ValueError("subpixel component/depth/visibility blocks differ")
    height, width, factor_y, factor_x = component.shape
    flat_component = component.reshape(height, width, factor_y * factor_x)
    flat_depth = depth.reshape(height, width, factor_y * factor_x)
    flat_visible = visible.reshape(height, width, factor_y * factor_x)
    sample_count = factor_y * factor_x
    component_2d = flat_component.reshape(-1, sample_count)
    depth_2d = flat_depth.reshape(-1, sample_count)
    eligible = flat_visible.reshape(-1, sample_count) & (depth_2d > 0.0)
    identified = eligible & (component_2d >= 0)
    same_component = component_2d[:, :, None] == component_2d[:, None, :]
    component_count = np.sum(
        same_component & identified[:, None, :], axis=2,
    ).astype(np.int64)
    component_count[~identified] = 0
    maximum_count = np.max(component_count, axis=1)
    candidate_component = identified & (
        component_count == maximum_count[:, None]
    )
    component_front_depth = np.min(
        np.where(
            same_component & eligible[:, None, :],
            depth_2d[:, None, :],
            np.inf,
        ),
        axis=2,
    )
    tied_front = np.min(
        np.where(candidate_component, component_front_depth, np.inf), axis=1,
    )
    selected = candidate_component & (
        component_front_depth == tied_front[:, None]
    )
    # If all visible samples lack a component identity, retain the frontmost
    # physical sample and declare the visible footprint pure.
    has_identity = maximum_count > 0
    selected[~has_identity] = eligible[~has_identity]
    choice_flat = np.argmin(
        np.where(selected, depth_2d, np.inf), axis=1,
    ).astype(np.int64)
    visible_count = np.sum(eligible, axis=1)
    purity_flat = np.zeros(component_2d.shape[0], dtype=np.float32)
    valid = visible_count > 0
    purity_flat[valid & has_identity] = (
        maximum_count[valid & has_identity] / visible_count[valid & has_identity]
    ).astype(np.float32)
    purity_flat[valid & ~has_identity] = 1.0
    different = (
        component_2d[:, :, None] != component_2d[:, None, :]
    ) & identified[:, :, None] & identified[:, None, :]
    mixed_flat = np.any(different, axis=(1, 2))
    return (
        choice_flat.reshape(height, width),
        purity_flat.reshape(height, width),
        mixed_flat.reshape(height, width),
    )


def _pool_rendered_surface(rendered: RenderedMapletAtlases, factor: int) -> RenderedMapletAtlases:
    """Mask-aware pooling after high-resolution 2DGS compositing."""

    feature = np.asarray(rendered.feature, dtype=np.float32)
    channels, high_height, high_width = feature.shape
    if high_height % int(factor) or high_width % int(factor):
        raise ValueError("render dimensions are not divisible by supersample factor")
    height, width = high_height // int(factor), high_width // int(factor)
    mask_blocks = np.asarray(rendered.mask, dtype=bool).reshape(
        height, factor, width, factor
    ).transpose(0, 2, 1, 3)
    count = np.sum(mask_blocks, axis=(2, 3)).astype(np.float32)
    mask = count > 0.0
    footprint = float(int(factor) * int(factor))
    feature_fraction = count / footprint
    feature_blocks = feature.reshape(
        channels, height, factor, width, factor
    ).transpose(0, 1, 3, 2, 4)
    pooled_feature = np.sum(feature_blocks * mask_blocks[None], axis=(3, 4))
    pooled_feature /= np.maximum(count[None], 1.0)
    pooled_feature /= np.maximum(np.linalg.norm(pooled_feature, axis=0, keepdims=True), 1e-8)
    pooled_feature[:, ~mask] = 0.0

    visibility_blocks = (
        mask_blocks
        if rendered.visibility is None
        else np.asarray(rendered.visibility, dtype=bool).reshape(
            height, factor, width, factor
        ).transpose(0, 2, 1, 3)
    )
    visibility_count = np.sum(visibility_blocks, axis=(2, 3)).astype(np.float32)
    visibility_fraction = visibility_count / footprint
    missing_blocks = (
        visibility_blocks & ~mask_blocks
        if rendered.field_missing is None
        else np.asarray(rendered.field_missing, dtype=bool).reshape(
            height, factor, width, factor
        ).transpose(0, 2, 1, 3)
    )
    missing_fraction = np.sum(missing_blocks, axis=(2, 3)).astype(np.float32) / footprint
    background_fraction = 1.0 - visibility_fraction

    visibility = (
        mask
        if rendered.visibility is None
        else np.any(visibility_blocks, axis=(2, 3))
    )
    field_missing = missing_fraction > 0.0

    # Identity and geometry are categorical.  Select one real dominant
    # physical component in every token footprint.
    ids = {}
    depth_blocks = np.asarray(rendered.depth, dtype=np.float32).reshape(
        height, factor, width, factor
    ).transpose(0, 2, 1, 3)
    component_source = (
        np.asarray(rendered.child_id, dtype=np.int64)
        if rendered.child_id is not None
        else np.asarray(rendered.surface_id, dtype=np.int64)
    )
    if rendered.child_id is not None and rendered.surface_id is not None:
        component_source = component_source.copy()
        missing_component = component_source < 0
        component_source[missing_component] = np.asarray(
            rendered.surface_id, dtype=np.int64,
        )[missing_component]
    component_blocks = component_source.reshape(
        height, factor, width, factor
    ).transpose(0, 2, 1, 3)
    choice, dominant_fraction, mixed_surface = _dominant_subpixel_choice(
        component_blocks, depth_blocks, visibility_blocks,
    )
    yy = (choice // factor) + np.arange(height)[:, None] * factor
    xx = (choice % factor) + np.arange(width)[None, :] * factor
    identity_names = ["maplet_id", "surface_id", "primitive_id"]
    if rendered.child_id is not None:
        identity_names.append("child_id")
    for name in identity_names:
        source = np.asarray(getattr(rendered, name))
        value = source[yy, xx].copy()
        value[~visibility] = -1
        ids[name] = value
    # Feature is an area mixture over the token footprint.  Pose geometry is
    # categorical and must belong to one real physical component; use the
    # frontmost sample of the dominant child/surface instead of averaging a
    # normal that may not exist anywhere in the scene.
    xyz = np.asarray(rendered.xyz, dtype=np.float32)[yy, xx].copy()
    normal = np.asarray(rendered.normal, dtype=np.float32)[yy, xx].copy()
    depth = np.asarray(rendered.depth, dtype=np.float32)[yy, xx].copy()
    uncertainty = np.asarray(rendered.uncertainty, dtype=np.float32)[yy, xx].copy()
    incidence = (
        np.zeros((height, width), dtype=np.float32)
        if rendered.incidence is None
        else np.asarray(rendered.incidence, dtype=np.float32)[yy, xx].copy()
    )
    projected_scale = (
        np.zeros((height, width), dtype=np.float32)
        if rendered.projected_scale is None
        else np.asarray(rendered.projected_scale, dtype=np.float32)[yy, xx].copy()
    )
    for value in (xyz, normal, depth, uncertainty, incidence, projected_scale):
        value[~visibility] = 0.0
    return RenderedMapletAtlases(
        feature=pooled_feature,
        xyz=xyz,
        normal=normal,
        uncertainty=uncertainty,
        maplet_id=ids["maplet_id"],
        mask=mask,
        depth=depth,
        surface_id=ids["surface_id"],
        primitive_id=ids["primitive_id"],
        child_id=ids.get("child_id"),
        visibility=visibility,
        field_missing=field_missing,
        incidence=incidence,
        projected_scale=projected_scale,
        feature_fraction=feature_fraction,
        visibility_fraction=visibility_fraction,
        missing_fraction=missing_fraction,
        background_fraction=background_fraction,
        dominant_surface_fraction=dominant_fraction,
        mixed_surface=mixed_surface,
    )
