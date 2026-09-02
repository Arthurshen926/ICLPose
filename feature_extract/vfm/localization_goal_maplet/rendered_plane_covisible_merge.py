"""Mapping-only covisibility merge for rendered finite-plane fragments.

The rendered fusion stage deliberately requires shared primitive support.  With a
dense camera inventory, the same physical plane can therefore remain split across
small reconstruction holes.  This module joins only fragments that repeatedly touch
in mapping-view contributor images, are mutually coplanar, and pass an atomic refit of
their exact primitive support.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .geometry_native_planar_map import (
    GeometryNativePlanarMap,
    PrimitiveSurfaceTable,
    SCHEMA,
    _fit_plane,
    _frame,
    _robust_group_quality,
)
from .rendered_plane_fusion import _fit_stats


def _fit_groups(
    table: PrimitiveSurfaceTable, groups: list[np.ndarray], references: list[np.ndarray],
    *, observation_groups: list[np.ndarray] | None = None,
    observation_stats: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    area = np.pi * table.scale1 * table.scale2 * table.opacity
    signs = np.asarray([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    normals = []
    offsets = []
    centers = []
    frames = []
    boundaries = []
    boundary_area = []
    support = []
    rms = []
    p95 = []
    cosine = []
    for group_index, (rows, reference) in enumerate(zip(groups, references)):
        if observation_groups is None or observation_stats is None:
            center, normal, _ = _fit_plane(table, rows, reference)
            group_rms, group_p95, group_cosine = _robust_group_quality(
                table, rows, center, normal,
            )
        else:
            observations = observation_groups[group_index]
            count = int(np.sum(observation_stats["pixel_counts"][observations]))
            total = np.sum(observation_stats["point_sum_world"][observations], axis=0)
            second = np.sum(
                observation_stats["point_second_moment_world"][observations], axis=0,
            )
            center, normal, group_rms = _fit_stats(count, total, second, reference)
            group_p95 = float(np.max(observation_stats["residual_p95_m"][observations]))
            group_cosine = float(np.quantile(np.abs(table.normals[rows] @ normal), 0.10))
        frame = _frame(normal)
        corners = (
            table.centers[rows, None, :]
            + signs[None, :, :1] * table.scale1[rows, None, None] * table.tangent1[rows, None, :]
            + signs[None, :, 1:] * table.scale2[rows, None, None] * table.tangent2[rows, None, :]
        ).reshape(-1, 3)
        uv = (corners - center) @ frame[:2].T
        try:
            hull = ConvexHull(uv)
            boundary = uv[hull.vertices]
            hull_area = float(hull.volume)
        except QhullError:
            boundary = uv[np.unique(uv, axis=0, return_index=True)[1]]
            hull_area = 0.0
        normals.append(normal)
        offsets.append(float(normal @ center))
        centers.append(center)
        frames.append(frame)
        boundaries.append(boundary)
        boundary_area.append(hull_area)
        support.append(float(area[rows].sum()))
        rms.append(group_rms)
        p95.append(group_p95)
        cosine.append(group_cosine)
    boundary_offsets = np.r_[0, np.cumsum([len(value) for value in boundaries])].astype(np.int64)
    member_offsets = np.r_[0, np.cumsum([len(value) for value in groups])].astype(np.int64)
    return {
        "plane_ids": np.arange(len(groups), dtype=np.int64),
        "normals_world": np.asarray(normals, np.float64).reshape(-1, 3),
        "offsets_world": np.asarray(offsets, np.float64),
        "centers_world": np.asarray(centers, np.float64).reshape(-1, 3),
        "frames_world": np.asarray(frames, np.float64).reshape(-1, 3, 3),
        "boundary_offsets": boundary_offsets,
        "boundary_uv": np.concatenate(boundaries) if boundaries else np.zeros((0, 2), np.float64),
        "boundary_area_m2": np.asarray(boundary_area, np.float64),
        "member_offsets": member_offsets,
        "member_primitive_rows": np.concatenate(groups).astype(np.int64),
        "member_counts": np.diff(member_offsets),
        "support_area_m2": np.asarray(support, np.float64),
        "residual_rms_m": np.asarray(rms, np.float64),
        "residual_p95_m": np.asarray(p95, np.float64),
        "normal_cosine_p10": np.asarray(cosine, np.float64),
    }


def merge_rendered_covisible_planes(
    table: PrimitiveSurfaceTable,
    planar: GeometryNativePlanarMap,
    lineage_offsets: np.ndarray,
    lineage_rows: np.ndarray,
    contributor_paths: list[Path],
    *,
    observation_stats: dict[str, np.ndarray] | None = None,
    minimum_covisible_views: int = 2,
    normal_degrees: float = 6.0,
    reciprocal_plane_distance_m: float = 0.05,
    maximum_refit_rms_m: float = 0.05,
    maximum_refit_p95_m: float = 0.08,
    fit_normal_degrees: float = 15.0,
    maximum_aggregate_rendered_rms_m: float = 0.07,
) -> tuple[GeometryNativePlanarMap, dict[str, np.ndarray], dict[str, object]]:
    """Merge image-adjacent coplanar fragments without query or appearance data."""

    table = table.validated()
    planar = planar.validated(table.primitive_ids.size)
    plane_count = int(planar.plane_ids.size)
    lineage_offsets = np.asarray(lineage_offsets, np.int64)
    lineage_rows = np.asarray(lineage_rows, np.int64)
    if (
        lineage_offsets.shape != (plane_count + 1,)
        or lineage_offsets[0] != 0
        or lineage_offsets[-1] != len(lineage_rows)
        or np.any(np.diff(lineage_offsets) < 0)
        or len(np.unique(lineage_rows)) != len(lineage_rows)
    ):
        raise ValueError("source plane lineage is invalid")
    if observation_stats is not None:
        required = {
            "pixel_counts", "point_sum_world", "point_second_moment_world", "residual_p95_m",
        }
        if set(observation_stats) != required:
            raise ValueError("rendered observation statistics contract differs")
        observation_count = len(observation_stats["pixel_counts"])
        if observation_count <= int(np.max(lineage_rows, initial=-1)):
            raise ValueError("rendered observation statistics are incomplete")
        if (
            observation_stats["point_sum_world"].shape != (observation_count, 3)
            or observation_stats["point_second_moment_world"].shape != (observation_count, 3, 3)
            or observation_stats["residual_p95_m"].shape != (observation_count,)
            or np.any(observation_stats["pixel_counts"] <= 0)
        ):
            raise ValueError("rendered observation statistic shapes differ")

    primitive_owner = np.full(table.primitive_ids.size, -1, np.int32)
    for plane in range(plane_count):
        lo, hi = map(int, planar.member_offsets[plane:plane + 2])
        primitive_owner[planar.member_primitive_rows[lo:hi]] = plane
    maximum_id = int(np.max(table.primitive_ids))
    row_by_id = np.full(maximum_id + 1, -1, np.int32)
    row_by_id[table.primitive_ids] = np.arange(table.primitive_ids.size, dtype=np.int32)

    view_count: dict[tuple[int, int], int] = {}
    for path in sorted(map(Path, contributor_paths)):
        with np.load(path, allow_pickle=False) as data:
            ids = np.asarray(data["topk_ids"][:, :, 0], np.int64)
        valid_id = (ids >= 0) & (ids <= maximum_id)
        rows = np.full(ids.shape, -1, np.int32)
        rows[valid_id] = row_by_id[ids[valid_id]]
        owners = np.full(ids.shape, -1, np.int32)
        valid_row = rows >= 0
        owners[valid_row] = primitive_owner[rows[valid_row]]
        pairs = []
        for left, right in ((owners[:, :-1], owners[:, 1:]), (owners[:-1], owners[1:])):
            valid = (left >= 0) & (right >= 0) & (left != right)
            if np.any(valid):
                pairs.append(np.sort(np.stack((left[valid], right[valid]), axis=1), axis=1))
        if pairs:
            for left, right in np.unique(np.concatenate(pairs), axis=0).tolist():
                key = (int(left), int(right))
                view_count[key] = view_count.get(key, 0) + 1

    cosine_threshold = float(np.cos(np.deg2rad(normal_degrees)))
    edges = []
    for (left, right), views in view_count.items():
        if views < int(minimum_covisible_views):
            continue
        if abs(float(planar.normals_world[left] @ planar.normals_world[right])) < cosine_threshold:
            continue
        if abs(float(planar.normals_world[left] @ planar.centers_world[right] - planar.offsets_world[left])) > reciprocal_plane_distance_m:
            continue
        if abs(float(planar.normals_world[right] @ planar.centers_world[left] - planar.offsets_world[right])) > reciprocal_plane_distance_m:
            continue
        edges.append((-int(views), int(left), int(right)))

    owner = np.arange(plane_count, dtype=np.int64)
    groups = {
        plane: planar.member_primitive_rows[
            int(planar.member_offsets[plane]):int(planar.member_offsets[plane + 1])
        ].copy()
        for plane in range(plane_count)
    }
    source_planes = {plane: {plane} for plane in range(plane_count)}

    def find(value: int) -> int:
        while owner[value] != value:
            owner[value] = owner[owner[value]]
            value = int(owner[value])
        return value

    accepted = 0
    rejected = 0
    rejected_rendered_rms = 0
    rejected_primitive_refit = 0
    rejected_component_coplanarity = 0
    for _negative_views, left, right in sorted(edges):
        a, b = find(left), find(right)
        if a == b:
            continue
        # Single-linkage is unsafe here: a sequence of individually plausible
        # 6-degree / 5-cm edges can silently fold a curved facade into one plane.
        # Require complete linkage over the original fused-plane instances before
        # evaluating the aggregate rendered-point refit.
        planes = sorted(source_planes[a] | source_planes[b])
        source_normals = np.asarray(planar.normals_world[planes], np.float64)
        source_centers = np.asarray(planar.centers_world[planes], np.float64)
        source_offsets = np.asarray(planar.offsets_world[planes], np.float64)
        if np.any(np.abs(source_normals @ source_normals.T) < cosine_threshold):
            rejected += 1
            rejected_component_coplanarity += 1
            continue
        distance = np.abs(source_normals @ source_centers.T - source_offsets[:, None])
        if np.any(distance > float(reciprocal_plane_distance_m)):
            rejected += 1
            rejected_component_coplanarity += 1
            continue
        rows = np.unique(np.r_[groups[a], groups[b]])
        if observation_stats is None:
            center, normal, _ = _fit_plane(table, rows, planar.normals_world[a])
            rms, p95, cosine_p10 = _robust_group_quality(table, rows, center, normal)
            if (
                rms > float(maximum_refit_rms_m)
                or p95 > float(maximum_refit_p95_m)
                or cosine_p10 < np.cos(np.deg2rad(float(fit_normal_degrees)))
            ):
                rejected += 1
                rejected_primitive_refit += 1
                continue
        else:
            observations = np.unique(np.concatenate([
                lineage_rows[int(lineage_offsets[plane]):int(lineage_offsets[plane + 1])]
                for plane in planes
            ])).astype(np.int64)
            count = int(np.sum(observation_stats["pixel_counts"][observations]))
            total = np.sum(observation_stats["point_sum_world"][observations], axis=0)
            second = np.sum(
                observation_stats["point_second_moment_world"][observations], axis=0,
            )
            _, _, rendered_rms = _fit_stats(
                count, total, second, planar.normals_world[a],
            )
            if rendered_rms > float(maximum_aggregate_rendered_rms_m):
                rejected += 1
                rejected_rendered_rms += 1
                continue
        root, other = min(a, b), max(a, b)
        owner[other] = root
        groups[root] = rows
        source_planes[root].update(source_planes[other])
        del groups[other]
        del source_planes[other]
        accepted += 1

    ordered_roots = sorted(groups, key=lambda root: int(np.min(table.primitive_ids[groups[root]])))
    final_groups = [groups[root] for root in ordered_roots]
    references = [planar.normals_world[min(source_planes[root])] for root in ordered_roots]
    final_lineage = []
    final_lineage_offsets = [0]
    for root in ordered_roots:
        values = np.concatenate([
            lineage_rows[int(lineage_offsets[plane]):int(lineage_offsets[plane + 1])]
            for plane in sorted(source_planes[root])
        ])
        values = np.unique(values).astype(np.int64)
        final_lineage.append(values)
        final_lineage_offsets.append(final_lineage_offsets[-1] + len(values))
    arrays = _fit_groups(
        table, final_groups, references,
        observation_groups=final_lineage if observation_stats is not None else None,
        observation_stats=observation_stats,
    )
    lineage = {
        "plane_observation_offsets": np.asarray(final_lineage_offsets, np.int64),
        "plane_observation_rows": (
            np.concatenate(final_lineage) if final_lineage else np.zeros(0, np.int64)
        ),
    }
    metadata = dict(planar.metadata)
    metadata.update({
        "artifact_type": SCHEMA,
        "representation": "rendered_finite_planes_with_mapping_covisibility_refit_merge",
        "plane_count": len(final_groups),
        "pre_covisibility_plane_count": plane_count,
        "covisibility_candidate_pair_count": len(view_count),
        "covisibility_eligible_edge_count": len(edges),
        "covisibility_accepted_merge_count": accepted,
        "covisibility_rejected_merge_count": rejected,
        "covisibility_rejected_component_coplanarity_count": rejected_component_coplanarity,
        "covisibility_rejected_rendered_rms_count": rejected_rendered_rms,
        "covisibility_rejected_primitive_refit_count": rejected_primitive_refit,
        "contributor_view_count": len(contributor_paths),
        "minimum_covisible_views": int(minimum_covisible_views),
        "merge_normal_degrees": float(normal_degrees),
        "merge_reciprocal_plane_distance_m": float(reciprocal_plane_distance_m),
        "maximum_refit_rms_m": float(maximum_refit_rms_m),
        "maximum_refit_p95_m": float(maximum_refit_p95_m),
        "maximum_aggregate_rendered_rms_m": float(maximum_aggregate_rendered_rms_m),
        "fit_normal_degrees": float(fit_normal_degrees),
        "merge_reads_only_contributor_top1_ids": True,
        "merge_reads_mapping_pose_or_rgb": False,
        "merge_reads_query_or_ground_truth": False,
        "merge_full_component_refit_with_atomic_rollback": True,
        "merge_component_complete_linkage_coplanarity": True,
        "merge_refit_authority": (
            "mapping_rendered_depth_point_moments"
            if observation_stats is not None else "raw_2dgs_primitive_ellipse_support"
        ),
        "finite_support_authority": "exact_member_primitive_rows",
        "boundary_role": "visualization_and_broad_phase_only",
        "boundary_uv_localization_eligible": False,
    })
    result = GeometryNativePlanarMap(metadata=metadata, **arrays).validated(
        table.primitive_ids.size
    )
    audit = {
        "premerge_plane_count": plane_count,
        "merged_plane_count": len(final_groups),
        "candidate_pair_count": len(view_count),
        "eligible_edge_count": len(edges),
        "accepted_merge_count": accepted,
        "rejected_merge_count": rejected,
        "rejected_component_coplanarity_count": rejected_component_coplanarity,
        "rejected_rendered_rms_count": rejected_rendered_rms,
        "rejected_primitive_refit_count": rejected_primitive_refit,
        "observation_count_before": int(len(lineage_rows)),
        "observation_count_after": int(len(lineage["plane_observation_rows"])),
    }
    return result, lineage, audit


__all__ = ["merge_rendered_covisible_planes"]
