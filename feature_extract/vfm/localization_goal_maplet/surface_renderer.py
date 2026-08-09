"""Exact clean-2DGS canonical-feature rendering for continuous alignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feature_extract.vfm.gaussian_vfm_field import GaussianVFMFeatureView
from feature_extract.vfm.localization_v6.atlas_renderer import RenderedMapletAtlases
from feature_extract.vfm.vfm_2dgs_mapping import SurfaceElementMap, _render_surface_element_pixel_contributions_2dgs

from .canonical_field import CanonicalSurfaceField
from .oracle_pose import intersect_rays_with_primitive_planes
from .physical_map import GoalMapletPhysicalMap
from .visibility import dominant_maplet_owner, signed_surface_visibility


@dataclass(frozen=True)
class RenderedSurfaceIdentity:
    primitive_rows: np.ndarray
    mask: np.ndarray


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
        order = np.lexsort((scene_primitive_rows, -contribution[selected], pixel_ids[selected]))
        ordered = selected[order]
        ordered_pixels = pixel_ids[ordered]
        first = np.r_[True, ordered_pixels[1:] != ordered_pixels[:-1]]
        chosen = ordered[first]
        primitive[pixel_ids[chosen]] = scene_rows[local_rows[chosen]]
        mask[pixel_ids[chosen]] = True
    shape = (int(height), int(width))
    return RenderedSurfaceIdentity(primitive.reshape(shape), mask.reshape(shape))


def dominant_child_owner(physical: GoalMapletPhysicalMap) -> np.ndarray:
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
    return owner


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
        for offset in range(0, selected.size, 250_000):
            rows = selected[offset : offset + 250_000]
            value = contribution[rows].astype(np.float32)
            np.add.at(feature_sum, pixel_ids[rows], value[:, None] * codes[field_rows[rows]])
            np.add.at(feature_alpha, pixel_ids[rows], value)
            np.add.at(uncertainty_sum, pixel_ids[rows], value * field.uncertainty[field_rows[rows]])
    total_alpha = np.zeros((pixel_count,), dtype=np.float32)
    if pixel_ids.size:
        np.add.at(total_alpha, pixel_ids, contribution.astype(np.float32))
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
        order = np.lexsort((scene_primitive_rows[selected], -contribution[selected], pixel_ids[selected]))
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
        focal = 0.5 * (float(camera.params[0]) + float(camera.params[1]))
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

    def mean_values(
        values: np.ndarray,
        support_blocks: np.ndarray = mask_blocks,
        support_count: np.ndarray = count,
    ) -> np.ndarray:
        source = np.asarray(values, dtype=np.float32)
        trailing = source.shape[2:]
        blocks = source.reshape(height, factor, width, factor, *trailing).transpose(
            0, 2, 1, 3, *range(4, 4 + len(trailing))
        )
        weighted = blocks * support_blocks[(...,) + (None,) * len(trailing)]
        denominator = np.maximum(support_count[(...,) + (None,) * len(trailing)], 1.0)
        return np.sum(weighted, axis=(2, 3)) / denominator

    xyz = mean_values(rendered.xyz, visibility_blocks, visibility_count)
    normal = mean_values(rendered.normal, visibility_blocks, visibility_count)
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-8)
    depth = mean_values(
        np.asarray(rendered.depth)[..., None], visibility_blocks, visibility_count,
    )[..., 0]
    uncertainty = mean_values(np.asarray(rendered.uncertainty)[..., None])[..., 0]
    visibility = (
        mask
        if rendered.visibility is None
        else np.any(visibility_blocks, axis=(2, 3))
    )
    field_missing = (
        visibility & ~mask
        if rendered.field_missing is None
        else np.any(
            np.asarray(rendered.field_missing, dtype=bool).reshape(
                height, factor, width, factor
            ).transpose(0, 2, 1, 3),
            axis=(2, 3),
        )
    )
    incidence = (
        np.zeros((height, width), dtype=np.float32)
        if rendered.incidence is None
        else mean_values(
            np.asarray(rendered.incidence)[..., None], visibility_blocks, visibility_count,
        )[..., 0]
    )
    projected_scale = (
        np.zeros((height, width), dtype=np.float32)
        if rendered.projected_scale is None
        else mean_values(
            np.asarray(rendered.projected_scale)[..., None], visibility_blocks, visibility_count,
        )[..., 0]
    )

    # Identity is categorical.  Select the closest valid high-resolution
    # sample in each token footprint; geometry above remains area-averaged.
    ids = {}
    depth_blocks = np.asarray(rendered.depth, dtype=np.float32).reshape(
        height, factor, width, factor
    ).transpose(0, 2, 1, 3)
    choice_cost = np.where(visibility_blocks & (depth_blocks > 0.0), depth_blocks, np.inf).reshape(
        height, width, factor * factor
    )
    choice = np.argmin(choice_cost, axis=2)
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
    )
