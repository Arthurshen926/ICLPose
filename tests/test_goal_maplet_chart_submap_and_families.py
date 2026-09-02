import json
from dataclasses import replace

import numpy as np

from feature_extract.vfm.localization_goal_maplet.chart_submap_selection import (
    CARDINALITY_SCHEMA,
    ChartSelectionConfig,
    ChartSubmapPlan,
    MappingChartView,
    build_chart_submap_plan,
    load_model_neutral_alignment_selection,
)
from feature_extract.vfm.localization_goal_maplet.chart_surface_families import (
    CanonicalSurfaceFamilyCarrier,
    SurfaceFamilyConfig,
    build_canonical_surface_families,
)
from feature_extract.vfm.localization_goal_maplet.explicit_chart_atlas import (
    ExplicitChartAtlas,
    SCHEMA as ATLAS_SCHEMA,
)


def _plane_view(name, center, *, world_z=5.0, reverse=False):
    height, width = 24, 32
    focal = 24.0
    pose = np.eye(4)
    if reverse:
        pose[:3, :3] = np.diag([-1.0, 1.0, -1.0])
    pose[:3, 3] = np.asarray(center, np.float64)
    yy, xx = np.mgrid[:height, :width]
    direction_camera = np.stack(
        (
            (xx - (width - 1) / 2) / focal,
            (yy - (height - 1) / 2) / focal,
            np.ones_like(xx),
        ),
        axis=-1,
    )
    direction_world = direction_camera @ pose[:3, :3].T
    distance = (world_z - pose[2, 3]) / direction_world[..., 2]
    points_world = pose[:3, 3] + direction_world * distance[..., None]
    points_camera = (points_world - pose[:3, 3]) @ pose[:3, :3]
    normal_world = np.zeros_like(points_world)
    normal_world[..., 2] = 1.0
    normal_camera = normal_world @ pose[:3, :3]
    valid = np.isfinite(points_camera).all(2) & (points_camera[..., 2] > 0)
    return MappingChartView(
        name=name,
        route="seq4",
        camera_to_world=pose,
        focal_px=focal,
        points_camera=points_camera,
        depth_camera=points_camera[..., 2],
        normals_camera=normal_camera,
        valid=valid,
        initializer_file_sha256="a" * 64,
        initializer_content_sha256="b" * 64,
        geometry_source="source_only_mast3r_mapping_pointmap",
    ).validated()


def test_overlap_plan_rejects_opposite_side_and_parallel_offset(tmp_path):
    views = [
        _plane_view("seq4__a.png", [0.0, 0.0, 0.0]),
        _plane_view("seq4__b.png", [0.7, 0.0, 0.0]),
        _plane_view("seq4__c.png", [1.4, 0.0, 0.0]),
        _plane_view("seq4__opposite.png", [0.0, 0.0, 10.0], reverse=True),
        _plane_view("seq4__offset.png", [0.7, 0.0, 0.0], world_z=7.0),
    ]
    config = ChartSelectionConfig(
        sample_stride=2,
        minimum_symmetric_surface_overlap=0.10,
        minimum_alignment_baseline_m=0.2,
        minimum_baseline_to_depth_ratio=0.02,
        minimum_median_triangulation_angle_deg=0.5,
        target_per_view_surface_support=0.20,
        target_supported_view_fraction=0.90,
    )
    plan = build_chart_submap_plan(
        views,
        config=config,
        lineage={
            "comparison_inventory_eligible": True,
            "system_control_only": False,
            "disjoint_authority_schema": "goal_maplet_disjoint_chart_upstream_authority_v2",
            "source_tree_sha256": "c" * 64,
        },
    )
    assert plan.coverage_edges[0, 1]
    assert plan.alignment_edges[1, 2]
    assert not plan.coverage_edges[0, 3]
    assert plan.same_surface_side_fraction[0, 3] == 0.0
    assert not plan.coverage_edges[0, 4]
    assert plan.metadata["comparison_inventory_eligible"] is True
    assert plan.metadata["operational_submap_count"] == 1
    assert plan.metadata["project_support_semantics_version"] == (
        "source_projection_pixel_center_v2"
    )
    assert plan.metadata["projection_principal_point_convention"] == (
        "pixel_centers_cx=(W-1)/2_cy=(H-1)/2"
    )
    assert plan.metadata["project_support_self_reprojection_floor_pass"] is True
    path = tmp_path / "plan.npz"
    metadata = plan.save_npz(path)
    loaded = ChartSubmapPlan.load_npz(path)
    assert np.array_equal(loaded.selected_mask, plan.selected_mask)
    selection = load_model_neutral_alignment_selection(
        path,
        expected_plan_content_sha256=metadata["content_sha256"],
    )
    assert selection.ordered_names == loaded.selected_chart_names_in_order
    assert selection.operational_submaps == (loaded.selected_chart_names_in_order,)
    try:
        load_model_neutral_alignment_selection(
            path,
            expected_plan_content_sha256="0" * 64,
        )
    except ValueError as error:
        assert "runner authority" in str(error)
    else:
        raise AssertionError("alignment runner accepted a stale plan hash")


def test_overlap_plan_rejects_legacy_half_pixel_principal_point():
    views = [
        _plane_view("seq4__a.png", [0.0, 0.0, 0.0]),
        _plane_view("seq4__b.png", [0.7, 0.0, 0.0]),
    ]
    legacy = []
    for view in views:
        points = view.points_camera.copy()
        points[..., 0] += 0.5 * points[..., 2] / view.focal_px
        points[..., 1] += 0.5 * points[..., 2] / view.focal_px
        legacy.append(replace(view, points_camera=points))
    try:
        build_chart_submap_plan(
            legacy,
            config=ChartSelectionConfig(sample_stride=1),
            lineage={"comparison_inventory_eligible": True, "system_control_only": False},
        )
    except ValueError as error:
        assert "self-reprojection failed" in str(error)
    else:
        raise AssertionError("legacy half-pixel projection was accepted")


def test_cardinality_frozen_plan_cannot_stop_before_exact_sixteen():
    views = [
        _plane_view(f"seq4__frame{row:05d}.png", [0.15 * row, 0.0, 0.0])
        for row in range(17)
    ]
    config = ChartSelectionConfig(
        sample_stride=4,
        minimum_symmetric_surface_overlap=0.05,
        minimum_alignment_baseline_m=0.10,
        minimum_baseline_to_depth_ratio=0.005,
        minimum_median_triangulation_angle_deg=0.1,
        target_per_view_surface_support=0.05,
        target_supported_view_fraction=0.5,
        minimum_selected_charts_per_submap=16,
        maximum_selected_charts_per_submap=16,
    )
    plan = build_chart_submap_plan(
        views,
        config=config,
        lineage={"comparison_inventory_eligible": True, "system_control_only": False},
    )
    assert plan.metadata["artifact_type"] == CARDINALITY_SCHEMA
    assert plan.metadata["operational_submap_count"] == 1
    assert plan.metadata["selected_chart_count"] == 16
    assert plan.metadata["selection_cardinality_frozen_before_held_geometry"] is True
    assert plan.metadata["held_geometry_used_for_selection"] is False
    assert plan.metadata["components"][0]["attempted_selected_chart_count"] == 16


def test_cardinality_frozen_plan_fails_when_component_has_only_fifteen():
    views = [
        _plane_view(f"seq4__frame{row:05d}.png", [0.15 * row, 0.0, 0.0])
        for row in range(15)
    ]
    config = ChartSelectionConfig(
        sample_stride=4,
        minimum_selected_charts_per_submap=16,
        maximum_selected_charts_per_submap=16,
    )
    plan = build_chart_submap_plan(
        views,
        config=config,
        lineage={"comparison_inventory_eligible": True, "system_control_only": False},
    )
    assert plan.metadata["operational_submap_count"] == 0
    assert plan.metadata["selected_chart_count"] == 0
    assert plan.metadata["components"][0]["decision_reason"] == (
        "component_below_minimum_selected_cardinality"
    )


def test_cardinality_interval_rejects_minimum_above_maximum():
    try:
        ChartSelectionConfig(
            minimum_selected_charts_per_submap=16,
            maximum_selected_charts_per_submap=15,
        ).validated()
    except ValueError as error:
        assert "cardinality interval" in str(error)
    else:
        raise AssertionError("invalid selected-cardinality interval was accepted")


def _grid_chart(z, chart_offset):
    height, width = 4, 4
    yy, xx = np.mgrid[:height, :width]
    vertices = np.stack((xx.astype(float), yy.astype(float), np.full_like(xx, z, dtype=float)), axis=-1).reshape(-1, 3)
    vertices[:, 0] += 0.01 * chart_offset
    faces = []
    for y in range(height - 1):
        for x in range(width - 1):
            a = y * width + x
            b = a + 1
            c = a + width
            d = c + 1
            faces.extend(((a, c, b), (b, c, d)))
    normals = np.zeros_like(vertices)
    normals[:, 2] = -1.0
    uv = vertices[:, :2] / 3.0
    return vertices, normals, uv, np.asarray(faces, np.int64)


def _three_chart_atlas():
    pieces = [_grid_chart(5.0, 0), _grid_chart(5.0, 1), _grid_chart(7.0, 0)]
    vertices = []
    normals = []
    uv = []
    faces = []
    vertex_offsets = [0]
    face_offsets = [0]
    for verts, norm, tex, tri in pieces:
        vertices.append(verts)
        normals.append(norm)
        uv.append(tex)
        faces.append(tri + vertex_offsets[-1])
        vertex_offsets.append(vertex_offsets[-1] + len(verts))
        face_offsets.append(face_offsets[-1] + len(tri))
    return ExplicitChartAtlas(
        chart_names=np.asarray(["seq4__a.png", "seq4__b.png", "seq4__c.png"]),
        chart_vertex_offsets=np.asarray(vertex_offsets, np.int64),
        vertices_world=np.concatenate(vertices),
        normals_world=np.concatenate(normals),
        uv=np.concatenate(uv),
        confidence=np.ones(sum(len(row[0]) for row in pieces), np.float32),
        chart_face_offsets=np.asarray(face_offsets, np.int64),
        faces=np.concatenate(faces),
        metadata={
            "artifact_type": ATLAS_SCHEMA,
            "control_only": False,
            "optimization_saw_excluded_routes": False,
        },
    ).validated()


def test_surface_family_is_runtime_identity_not_source_chart(tmp_path):
    atlas = _three_chart_atlas()
    config = SurfaceFamilyConfig(
        minimum_cross_chart_matches=4,
        maximum_overlap_samples_per_patch=32,
        minimum_smaller_patch_overlap=0.5,
        minimum_larger_patch_overlap=0.5,
        minimum_online_supported_patch_area_fraction=0.5,
    )
    carrier, lineage = build_canonical_surface_families(
        atlas,
        config=config,
        source_atlas_content_sha256="c" * 64,
    )
    assert carrier.patch_count == 3
    assert carrier.family_count == 2
    assert carrier.family_online_eligible.sum() == 1
    assert carrier.metadata["runtime_candidate_unit"] == "canonical_surface_family"
    assert carrier.metadata["family_canonicalization_gate_pass"] is True
    assert "chart_names" not in carrier.arrays()
    runtime_metadata = json.dumps(carrier.metadata)
    assert "seq4__a.png" not in runtime_metadata
    assert "seq4__b.png" not in runtime_metadata
    assert lineage["parameterization_id_to_source_chart_name"]["0"] == "seq4__a.png"
    path = tmp_path / "families.npz"
    carrier.save_npz(path)
    loaded = CanonicalSurfaceFamilyCarrier.load_npz(path)
    assert np.array_equal(loaded.patch_family_ids, carrier.patch_family_ids)


def test_surface_family_rejects_control_only_atlas():
    atlas = _three_chart_atlas()
    atlas.metadata["control_only"] = True
    try:
        build_canonical_surface_families(
            atlas,
            config=SurfaceFamilyConfig(minimum_cross_chart_matches=4),
            source_atlas_content_sha256="d" * 64,
        )
    except ValueError as error:
        assert "control-only" in str(error)
    else:
        raise AssertionError("control-only atlas was accepted")
