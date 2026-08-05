from types import SimpleNamespace

import numpy as np

from feature_extract.vfm.localization_goal_maplet import (
    DOUBLE_SIDED,
    SINGLE_SIDED,
    GoalMapletPhysicalMap,
    SurfacePrimitiveGeometry,
    build_goal_maplet_physical_map,
)
from feature_extract.vfm.localization_goal_maplet.audit import audit_physical_map


def _inputs():
    geometry = SurfacePrimitiveGeometry(
        primitive_ids=np.arange(10, dtype=np.int64),
        centers=np.stack([np.linspace(-0.4, 0.4, 10), np.zeros(10), np.zeros(10)], axis=1),
        tangent1=np.tile([[1.0, 0.0, 0.0]], (10, 1)),
        tangent2=np.tile([[0.0, 1.0, 0.0]], (10, 1)),
        normals=np.tile([[0.0, 0.0, -1.0]], (10, 1)),
        scale1=np.full(10, 0.04),
        scale2=np.full(10, 0.04),
        opacity=np.ones(10),
    )
    maplets = SimpleNamespace(
        maplet_ids=np.asarray([7]),
        centers=np.asarray([[0.0, 0.0, 0.0]]),
        normals=np.asarray([[0.0, 0.0, -1.0]]),
        tangent_frames=np.asarray([[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]]),
        support_offsets=np.asarray([0, 8]),
        support_element_ids=np.arange(8, dtype=np.int64),
    )
    region = SimpleNamespace(
        anchor_ids=np.asarray([7]),
        support_offsets=np.asarray([0, 8]),
        support_element_ids=np.arange(8, dtype=np.int64),
        support_weights=np.ones(8, dtype=np.float32),
        observed_view_ids=(("seq1/frame00001.png",),),
    )
    pose = np.eye(4)
    pose[2, 3] = -2.0  # camera center is +2 on z, so normals must flip.
    return geometry, maplets, region, {"seq1/frame00001.png": pose}


def test_exact_membership_orientation_children_and_roundtrip(tmp_path):
    geometry, maplets, region, poses = _inputs()
    result = build_goal_maplet_physical_map(
        maplets,
        region,
        geometry,
        poses,
        clean_primitive_ids=np.arange(10),
        minimum_child_count=2,
        maximum_child_count=4,
    )
    assert result.primitive_ids.tolist() == list(range(10))  # includes two scene occluders
    assert result.primitive_ids[result.membership_primitive_rows].tolist() == list(range(8))
    assert result.maplet_sidedness.tolist() == [int(SINGLE_SIDED)]
    assert np.all(result.maplet_normals[:, 2] > 0.99)
    assert np.all(result.primitive_normals[:8, 2] > 0.99)
    assert np.all(result.primitive_sidedness[:8] == SINGLE_SIDED)
    assert np.all(result.primitive_sidedness[8:] == DOUBLE_SIDED)
    assert 2 <= result.child_parent_rows.size <= 4
    output = tmp_path / "physical.npz"
    result.save_npz(output)
    loaded = GoalMapletPhysicalMap.load_npz(output)
    assert loaded.content_sha256 == result.content_sha256
    assert loaded.metadata["stores_mapping_rgb"] is False


def test_physical_audit_reports_independent_gates():
    geometry, maplets, region, poses = _inputs()
    result = build_goal_maplet_physical_map(
        maplets,
        region,
        geometry,
        poses,
        minimum_child_count=2,
        maximum_child_count=4,
    )
    report = audit_physical_map(result)
    assert report["maplet_count"] == 1
    assert report["membership_count"] == 8
    assert "physical_map_ready" in report
    assert set(report["gates"]) == {
        "no_empty_or_tiny_maplets",
        "connected_surface_support",
        "normal_purity",
        "single_depth_layer",
        "normal_orientation_resolved",
        "metric_child_support",
    }
