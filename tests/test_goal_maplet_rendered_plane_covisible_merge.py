from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import (
    GeometryNativePlanarMap,
    PrimitiveSurfaceTable,
    SCHEMA,
    _frame,
)
from feature_extract.vfm.localization_goal_maplet.rendered_plane_covisible_merge import (
    merge_rendered_covisible_planes,
)


def _fixture(tmp_path: Path):
    centers = np.asarray([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0], [0.2, 0.0, 2.0],
                          [0.3, 0.0, 2.0], [0.4, 0.0, 2.0], [0.5, 0.0, 2.0]])
    table = PrimitiveSurfaceTable(
        primitive_ids=np.arange(6), centers=centers,
        tangent1=np.tile([1.0, 0.0, 0.0], (6, 1)),
        tangent2=np.tile([0.0, 1.0, 0.0], (6, 1)),
        normals=np.tile([0.0, 0.0, 1.0], (6, 1)),
        scale1=np.full(6, 0.08), scale2=np.full(6, 0.08), opacity=np.ones(6),
    ).validated()
    groups = [np.arange(3), np.arange(3, 6)]
    boundaries = [np.asarray([[-.1, -.1], [.1, -.1], [.1, .1], [-.1, .1]])] * 2
    arrays = dict(
        plane_ids=np.arange(2), normals_world=np.tile([0., 0., 1.], (2, 1)),
        offsets_world=np.full(2, 2.), centers_world=np.asarray([[.1, 0., 2.], [.4, 0., 2.]]),
        frames_world=np.asarray([_frame(np.asarray([0., 0., 1.]))] * 2),
        boundary_offsets=np.asarray([0, 4, 8]), boundary_uv=np.concatenate(boundaries),
        boundary_area_m2=np.full(2, .04), member_offsets=np.asarray([0, 3, 6]),
        member_primitive_rows=np.arange(6), member_counts=np.asarray([3, 3]),
        support_area_m2=np.full(2, .05), residual_rms_m=np.zeros(2),
        residual_p95_m=np.zeros(2), normal_cosine_p10=np.ones(2),
    )
    planar = GeometryNativePlanarMap(
        metadata={"artifact_type": SCHEMA, "uses_parent_child_partition": False,
                  "boundary": "convex_summary_plus_exact_member_primitive_inventory"},
        **arrays,
    ).validated(6)
    contributor = tmp_path / "mapping.npz"
    ids = np.asarray([[0, 1, 2, 3, 4, 5]], np.int64)
    np.savez_compressed(contributor, topk_ids=ids[:, :, None])
    return table, planar, contributor


def test_covisible_coplanar_fragments_merge_and_lineage_is_preserved(tmp_path):
    table, planar, contributor = _fixture(tmp_path)
    merged, lineage, audit = merge_rendered_covisible_planes(
        table, planar, np.asarray([0, 1, 2]), np.asarray([10, 11]),
        [contributor], minimum_covisible_views=1,
    )
    assert len(merged.plane_ids) == 1
    assert np.array_equal(merged.member_primitive_rows, np.arange(6))
    assert np.array_equal(lineage["plane_observation_offsets"], [0, 2])
    assert np.array_equal(lineage["plane_observation_rows"], [10, 11])
    assert audit["accepted_merge_count"] == 1
    assert merged.metadata["merge_reads_query_or_ground_truth"] is False


def test_non_coplanar_fragment_is_not_merged(tmp_path):
    table, planar, contributor = _fixture(tmp_path)
    normals = planar.normals_world.copy()
    normals[1] = [0.0, 1.0, 0.0]
    altered = GeometryNativePlanarMap(
        metadata=planar.metadata, **{**planar.arrays(), "normals_world": normals,
                                    "offsets_world": np.asarray([2.0, 0.0])},
    ).validated(6)
    merged, _, audit = merge_rendered_covisible_planes(
        table, altered, np.asarray([0, 1, 2]), np.asarray([10, 11]),
        [contributor], minimum_covisible_views=1,
    )
    assert len(merged.plane_ids) == 2
    assert audit["accepted_merge_count"] == 0


def test_rendered_point_refit_is_the_matching_fusion_authority(tmp_path):
    table, planar, contributor = _fixture(tmp_path)
    stats = {
        "pixel_counts": np.asarray([10, 10]),
        "point_sum_world": np.asarray([[1.0, 0.0, 20.0], [4.0, 0.0, 20.0]]),
        "point_second_moment_world": np.asarray([
            [[.12, 0., 2.], [0., .01, 0.], [2., 0., 40.]],
            [[1.62, 0., 8.], [0., .01, 0.], [8., 0., 40.]],
        ]),
        "residual_p95_m": np.asarray([0.01, 0.01]),
    }
    merged, _, audit = merge_rendered_covisible_planes(
        table, planar, np.asarray([0, 1, 2]), np.asarray([0, 1]),
        [contributor], minimum_covisible_views=1, observation_stats=stats,
    )
    assert len(merged.plane_ids) == 1
    assert audit["accepted_merge_count"] == 1
    assert merged.metadata["merge_refit_authority"] == "mapping_rendered_depth_point_moments"


def test_component_complete_linkage_blocks_coplanarity_chains(tmp_path):
    angles = np.deg2rad([0.0, 5.0, 10.0])
    plane_normals = np.stack((np.sin(angles), np.zeros(3), np.cos(angles)), axis=1)
    centers = np.asarray([
        [0.00, 0.0, 2.0], [0.04, 0.0, 2.0],
        [0.10, 0.0, 2.0], [0.14, 0.0, 2.0],
        [0.20, 0.0, 2.0], [0.24, 0.0, 2.0],
    ])
    normals = np.repeat(plane_normals, 2, axis=0)
    tangent1 = np.stack((normals[:, 2], np.zeros(6), -normals[:, 0]), axis=1)
    table = PrimitiveSurfaceTable(
        primitive_ids=np.arange(6), centers=centers, tangent1=tangent1,
        tangent2=np.tile([0.0, 1.0, 0.0], (6, 1)), normals=normals,
        scale1=np.full(6, 0.02), scale2=np.full(6, 0.02), opacity=np.ones(6),
    ).validated()
    frame = np.asarray([_frame(value) for value in plane_normals])
    planar = GeometryNativePlanarMap(
        metadata={"artifact_type": SCHEMA, "uses_parent_child_partition": False,
                  "boundary": "convex_summary_plus_exact_member_primitive_inventory"},
        plane_ids=np.arange(3), normals_world=plane_normals,
        offsets_world=np.sum(plane_normals * centers[::2], axis=1),
        centers_world=centers[::2], frames_world=frame,
        boundary_offsets=np.asarray([0, 4, 8, 12]),
        boundary_uv=np.tile(np.asarray([[-.02, -.02], [.02, -.02], [.02, .02], [-.02, .02]]), (3, 1)),
        boundary_area_m2=np.full(3, .0016), member_offsets=np.asarray([0, 2, 4, 6]),
        member_primitive_rows=np.arange(6), member_counts=np.full(3, 2),
        support_area_m2=np.full(3, .0025), residual_rms_m=np.zeros(3),
        residual_p95_m=np.zeros(3), normal_cosine_p10=np.ones(3),
    ).validated(6)
    contributor = tmp_path / "chain.npz"
    np.savez_compressed(contributor, topk_ids=np.arange(6)[None, :, None])
    merged, _, audit = merge_rendered_covisible_planes(
        table, planar, np.arange(4), np.arange(3), [contributor],
        minimum_covisible_views=1, normal_degrees=6.0,
        reciprocal_plane_distance_m=0.05,
    )
    # A-B and B-C are individually eligible, while A-C differ by 10 degrees.
    # Complete linkage may merge either adjacent pair but must never merge all 3.
    assert len(merged.plane_ids) == 2
    assert audit["accepted_merge_count"] == 1
    assert audit["rejected_component_coplanarity_count"] == 1
