from __future__ import annotations

import json
import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.chart_radio_uv_field import (
    CANONICAL_SCHEMA,
    FAMILY_SCHEMA,
    COORDINATE_CONTRACT,
    CanonicalChartRadioField,
    CanonicalSurfaceFamilyLayout,
    IdealChartCamera,
    RawSimpleRadialCamera,
    SourceViewChartRadioField,
    attach_radio_to_chart_atlas,
    build_carrier_constrained_surface_family_layout,
    build_diagnostic_metric_surface_families,
    chart_name_to_image_id,
    fuse_canonical_chart_radio_field,
)
from feature_extract.vfm.localization_goal_maplet.explicit_chart_atlas import (
    ExplicitChartAtlas,
    SCHEMA as ATLAS_SCHEMA,
)
from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256
from feature_extract.vfm.localization_goal_maplet.chart_submap_selection import (
    ALIGNMENT_SELECTION_CONTRACT,
    ChartSubmapPlan,
    SCHEMA as CHART_SUBMAP_PLAN_SCHEMA,
)
from feature_extract.vfm.localization_goal_maplet.chart_surface_families import (
    CanonicalSurfaceFamilyCarrier,
    SurfaceFamilyConfig,
    build_canonical_surface_families,
)
from feature_extract.tools.vfm.build_goal_maplet_chart_radio_uv_field import (
    _strict_disjoint_surface_audit,
)


def _atlas(*, second_vertex_count: int = 3) -> ExplicitChartAtlas:
    first = np.asarray([[-0.4, -0.3, 1.0], [0.4, -0.3, 1.0], [0.0, 0.3, 1.0]])
    if second_vertex_count == 3:
        second = first.copy()
        second_faces = np.asarray([[3, 4, 5]], dtype=np.int64)
        second_uv = np.asarray([[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]])
    else:
        second = np.asarray(
            [
                [-0.4, -0.3, 1.0], [0.0, -0.3, 1.0], [0.4, -0.3, 1.0],
                [-0.4, 0.3, 1.0], [0.0, 0.3, 1.0], [0.4, 0.3, 1.0],
            ]
        )
        second_faces = np.asarray(
            [[3, 6, 4], [4, 6, 7], [4, 7, 5], [5, 7, 8]], dtype=np.int64,
        )
        second_uv = np.asarray(
            [[0.1, 0.1], [0.5, 0.1], [0.9, 0.1],
             [0.1, 0.9], [0.5, 0.9], [0.9, 0.9]],
        )
    vertices = np.concatenate([first, second])
    uv = np.concatenate(
        [np.asarray([[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]]), second_uv]
    )
    normals = np.tile(np.asarray([[0.0, 0.0, -1.0]]), (len(vertices), 1))
    return ExplicitChartAtlas(
        chart_names=np.asarray(["seq1__frame00001.png", "seq2__frame00002.png"]),
        chart_vertex_offsets=np.asarray([0, 3, len(vertices)], dtype=np.int64),
        vertices_world=vertices,
        normals_world=normals,
        uv=uv,
        confidence=np.ones((len(vertices),), dtype=np.float32),
        chart_face_offsets=np.asarray([0, 1, 1 + len(second_faces)], dtype=np.int64),
        faces=np.concatenate([np.asarray([[0, 1, 2]], dtype=np.int64), second_faces]),
        metadata={
            "artifact_type": ATLAS_SCHEMA,
            "content_sha256": "atlas-test",
            "control_only": False,
            "optimization_saw_excluded_routes": False,
        },
    ).validated()


def _source(atlas: ExplicitChartAtlas) -> SourceViewChartRadioField:
    names = atlas.chart_names.astype(str).tolist()
    camera = RawSimpleRadialCamera(8, 6, 6.0, 4.0, 3.0, 0.08).validated()
    chart_camera = IdealChartCamera(8, 6, 6.0, 4.0, 3.0).validated()
    first = np.zeros((2, 2, 3), dtype=np.float32)
    first[0] = 1.0
    second = np.zeros((2, 2, 3), dtype=np.float32)
    second[1] = 1.0
    return attach_radio_to_chart_atlas(
        atlas,
        radio_by_view={names[0]: first, names[1]: second},
        raw_camera_by_view={names[0]: camera, names[1]: camera},
        chart_camera_by_view={names[0]: chart_camera, names[1]: chart_camera},
        camera_pose_c2w_by_view={names[0]: np.eye(4), names[1]: np.eye(4)},
        atlas_content_sha256="atlas-test",
        lineage_metadata={"token_files_sha256": ["a" * 64, "b" * 64]},
    )


def test_radial_chart_uv_roundtrip_and_strict_name_contract() -> None:
    camera = RawSimpleRadialCamera(1024, 576, 884.0, 512.0, 288.0, 0.043).validated()
    raw = np.asarray([[0.0, 0.0], [512.0, 288.0], [1023.0, 575.0], [321.5, 92.25]])
    uv = camera.raw_xy_to_ideal_uv(raw)
    replay = camera.ideal_uv_to_raw_xy(uv)
    np.testing.assert_allclose(replay, raw, atol=2e-9)
    chart = IdealChartCamera(256, 144, 221.0, 128.0, 72.0).validated()
    chart_uv = chart.raw_xy_to_chart_uv(camera, raw)
    np.testing.assert_allclose(chart.chart_uv_to_raw_xy(camera, chart_uv), raw, atol=2e-9)
    assert chart_name_to_image_id("seq4__frame00001.png") == "seq4/frame00001.png"
    with pytest.raises(ValueError, match="strict flattened"):
        chart_name_to_image_id("frame00001.png")


def test_source_field_preserves_complete_token_layout_and_lineage(tmp_path) -> None:
    source = _source(_atlas())
    assert source.metadata["coordinate_contract"] == COORDINATE_CONTRACT
    assert source.view_token_offsets.tolist() == [0, 6, 12]
    assert source.token_xy.tolist() == [
        [0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1],
        [0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1],
    ]
    np.testing.assert_array_equal(source.token_codes[:6, 0], np.ones(6))
    np.testing.assert_array_equal(source.token_codes[6:, 1], np.ones(6))
    assert np.allclose(source.vertex_token_weights.sum(axis=1), 1.0)
    path = tmp_path / "source.npz"
    sealed = source.save_npz(path)
    replay = SourceViewChartRadioField.load_npz(path)
    assert replay.content_sha256 == sealed["content_sha256"]
    damaged = dict(source.metadata)
    damaged["uses_query_or_ground_truth"] = True
    with pytest.raises(ValueError, match="query/ground truth"):
        SourceViewChartRadioField(**{**source.__dict__, "metadata": damaged}).validated()
    invalid_validity = np.asarray(source.token_chart_uv_valid).copy()
    invalid_validity[0] = ~invalid_validity[0]
    with pytest.raises(ValueError, match="validity does not replay"):
        SourceViewChartRadioField(
            **{**source.__dict__, "token_chart_uv_valid": invalid_validity}
        ).validated()


def test_diagnostic_family_builder_merges_identical_cross_view_vertices() -> None:
    atlas = _atlas()
    layout = build_diagnostic_metric_surface_families(
        atlas,
        atlas_content_sha256="atlas-test",
        maximum_cross_chart_distance_m=0.05,
    )
    assert layout.metadata["diagnostic_only"] is True
    assert layout.metadata["canonical_identity_is_source_view_id"] is False
    assert layout.metadata["multi_view_node_count"] == 3
    assert layout.node_points_world.shape[0] == 3
    assert layout.family_keys.size == 1
    np.testing.assert_array_equal(layout.vertex_node_rows[:3], layout.vertex_node_rows[3:])


def test_canonical_fusion_is_view_balanced_and_anonymizes_sources(tmp_path) -> None:
    atlas = _atlas(second_vertex_count=6)
    source = _source(atlas)
    vertex_count = atlas.vertices_world.shape[0]
    layout = CanonicalSurfaceFamilyLayout(
        family_keys=np.asarray(["surface-family-00000"]),
        family_node_offsets=np.asarray([0, 1], dtype=np.int64),
        node_points_world=np.asarray([[0.0, 0.0, 1.0]]),
        node_normals_world=np.asarray([[0.0, 0.0, -1.0]]),
        vertex_family_rows=np.zeros((vertex_count,), dtype=np.int32),
        vertex_node_rows=np.zeros((vertex_count,), dtype=np.int64),
        vertex_assignment_weight=np.ones((vertex_count,), dtype=np.float32),
        atlas_content_sha256="atlas-test",
        metadata={
            "artifact_type": FAMILY_SCHEMA,
            "atlas_content_sha256": "atlas-test",
            "diagnostic_only": False,
            "production_eligible": True,
            "feature_interface_eligible": True,
        },
    ).validated()
    field = fuse_canonical_chart_radio_field(source, layout, maximum_anonymous_prototypes=2)
    np.testing.assert_allclose(field.codes[0], np.asarray([2.0**-0.5, 2.0**-0.5]), atol=2e-6)
    assert field.view_count.tolist() == [2]
    assert field.prototype_offsets.tolist() == [0, 2]
    assert field.metadata["stores_source_view_names"] is False
    assert "seq1" not in str(field.metadata) and "seq2" not in str(field.metadata)
    assert field.metadata["production_eligible"] is True
    path = tmp_path / "canonical.npz"
    sealed = field.save_npz(path)
    replay = CanonicalChartRadioField.load_npz(path)
    assert replay.content_sha256 == sealed["content_sha256"]
    assert replay.metadata["artifact_type"] == CANONICAL_SCHEMA


def test_authoritative_family_carrier_constrains_radio_nodes(tmp_path) -> None:
    atlas = _atlas()
    carrier, _ = build_canonical_surface_families(
        atlas,
        config=SurfaceFamilyConfig(
            minimum_patch_faces=1,
            minimum_patch_area_m2=0.0,
            maximum_cross_chart_distance_m=0.05,
            minimum_cross_chart_matches=3,
            maximum_overlap_samples_per_patch=16,
            minimum_smaller_patch_overlap=0.5,
            minimum_larger_patch_overlap=0.5,
            maximum_overlap_point_to_plane_median_m=0.05,
            minimum_online_supported_patch_area_fraction=0.5,
        ),
        source_atlas_content_sha256="atlas-test",
    )
    carrier_path = tmp_path / "carrier.npz"
    carrier.save_npz(carrier_path)
    carrier = CanonicalSurfaceFamilyCarrier.load_npz(carrier_path)
    layout = build_carrier_constrained_surface_family_layout(
        atlas,
        carrier,
        atlas_content_sha256="atlas-test",
        carrier_content_sha256=str(carrier.metadata["content_sha256"]),
        maximum_cross_chart_distance_m=0.05,
    )
    assert layout.family_keys.size == 1
    assert layout.node_points_world.shape[0] == 3
    assert layout.metadata["multi_view_node_count"] == 3
    assert layout.metadata["family_merge_authority"].startswith("input_carrier_only")
    assert layout.metadata["feature_interface_eligible"] is True
    assert layout.metadata["production_eligible"] is False
    canonical = fuse_canonical_chart_radio_field(_source(atlas), layout)
    assert canonical.metadata["feature_interface_eligible"] is True
    assert canonical.metadata["production_eligible"] is False


def _sealed_two_chart_plan(path, names, authority_content, source_tree):
    count = len(names)
    zeros = np.zeros((count, count), dtype=np.float64)
    edges = np.ones((count, count), dtype=bool)
    np.fill_diagonal(edges, False)
    metadata = {
        "artifact_type": CHART_SUBMAP_PLAN_SCHEMA,
        "uses_query_or_ground_truth": False,
        "chart_count": count,
        "selected_chart_count": count,
        "source_ordered_names_sha256": canonical_json_sha256(names),
        "selected_chart_names_in_order": names,
        "selected_chart_names_in_order_sha256": canonical_json_sha256(names),
        "alignment_runner_contract": ALIGNMENT_SELECTION_CONTRACT,
        "components": [
            {"operational_coverage_pass": True, "selected_chart_names": names}
        ],
        "operational_submap_count": 1,
        "comparison_inventory_eligible": True,
        "system_control_only": False,
        "lineage": {
            "disjoint_authority_schema": "goal_maplet_disjoint_chart_upstream_authority_v2",
            "disjoint_authority_content_sha256": authority_content,
            "source_tree_sha256": source_tree,
        },
    }
    plan = ChartSubmapPlan(
        chart_names=np.asarray(names),
        camera_centers_world=np.stack([np.asarray([row, 0.0, 0.0]) for row in range(count)]),
        camera_forward_world=np.tile(np.asarray([[0.0, 0.0, 1.0]]), (count, 1)),
        valid_sample_counts=np.full((count,), 100, dtype=np.int64),
        directional_frustum_fraction=zeros.copy(),
        directional_depth_support=zeros.copy(),
        directional_surface_support=zeros.copy(),
        symmetric_surface_overlap=zeros.copy(),
        camera_baseline_m=zeros.copy(),
        median_overlap_depth_m=zeros.copy(),
        baseline_to_depth_ratio=zeros.copy(),
        median_triangulation_angle_deg=zeros.copy(),
        camera_forward_angle_deg=zeros.copy(),
        same_surface_side_fraction=zeros.copy(),
        coverage_edges=edges.copy(),
        alignment_edges=edges.copy(),
        component_ids=np.zeros((count,), dtype=np.int32),
        selected_mask=np.ones((count,), dtype=bool),
        selection_rank=np.arange(count, dtype=np.int32),
        metadata=metadata,
    )
    return plan.save_npz(path)


def test_strict_input_hook_requires_separate_v2_plan_and_family_carrier(tmp_path) -> None:
    atlas = _atlas()
    names = atlas.chart_names.astype(str).tolist()
    source_tree = "d" * 64
    authority = {
        "artifact_type": "goal_maplet_disjoint_chart_upstream_authority_v2",
        "strict_disjoint_upstream": True,
        "source_held_image_disjoint": True,
        "source_held_route_disjoint": True,
        "uses_query_or_ground_truth": False,
        "source": {"ordered_names": names, "tree_sha256": source_tree},
        "held": {"ordered_names": ["seq9__frame00001.png"]},
    }
    authority["content_sha256"] = canonical_json_sha256(authority)
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(json.dumps(authority))
    plan_path = tmp_path / "plan.npz"
    _sealed_two_chart_plan(
        plan_path, names, str(authority["content_sha256"]), source_tree,
    )
    carrier, _ = build_canonical_surface_families(
        atlas,
        config=SurfaceFamilyConfig(
            minimum_patch_faces=1,
            minimum_patch_area_m2=0.0,
            maximum_cross_chart_distance_m=0.05,
            minimum_cross_chart_matches=3,
            maximum_overlap_samples_per_patch=16,
            minimum_smaller_patch_overlap=0.5,
            minimum_larger_patch_overlap=0.5,
            maximum_overlap_point_to_plane_median_m=0.05,
            minimum_online_supported_patch_area_fraction=0.5,
        ),
        source_atlas_content_sha256="c" * 64,
    )
    carrier_path = tmp_path / "carrier.npz"
    carrier_metadata = carrier.save_npz(carrier_path)
    audit = _strict_disjoint_surface_audit(
        authority_path=authority_path,
        submap_plan_path=plan_path,
        family_carrier_path=carrier_path,
        expected_plan_content_sha256=ChartSubmapPlan.load_npz(plan_path).metadata[
            "content_sha256"
        ],
        expected_family_carrier_content_sha256=carrier_metadata["content_sha256"],
        chart_names=names,
        atlas_content_sha256="c" * 64,
    )
    assert audit["strict_promotion_eligible"] is True
    assert audit["chart_submap_plan_verified"] is True
    assert audit["surface_family_carrier_verified"] is True
    assert audit["alignment_runner_contract"] == ALIGNMENT_SELECTION_CONTRACT
    with pytest.raises(ValueError, match="chart order differ"):
        _strict_disjoint_surface_audit(
            authority_path=authority_path,
            submap_plan_path=plan_path,
            family_carrier_path=carrier_path,
            expected_plan_content_sha256=ChartSubmapPlan.load_npz(plan_path).metadata[
                "content_sha256"
            ],
            expected_family_carrier_content_sha256=carrier_metadata["content_sha256"],
            chart_names=list(reversed(names)),
            atlas_content_sha256="c" * 64,
        )
    with pytest.raises(ValueError, match="requires v2 authority"):
        _strict_disjoint_surface_audit(
            authority_path=authority_path,
            submap_plan_path=None,
            family_carrier_path=carrier_path,
            expected_plan_content_sha256=None,
            expected_family_carrier_content_sha256=carrier_metadata["content_sha256"],
            chart_names=names,
            atlas_content_sha256="c" * 64,
        )
