from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_full_2dgs_held_geometry_control import (
    _camera,
    _depth_normals_world,
    _planar_source_subset,
)
from feature_extract.vfm.official_2dgs_renderer import Official2DGSSource


def test_depth_normals_world_recovers_frontoparallel_plane() -> None:
    depth = np.full((7, 9), 4.0, np.float64)
    valid = np.ones_like(depth, bool)
    normal = _depth_normals_world(
        depth,
        valid,
        np.asarray([100.0, 100.0]),
        np.asarray([4.0, 3.0]),
        np.eye(4),
    )
    expected = np.zeros_like(normal[2:-2, 2:-2])
    expected[..., 2] = 1.0
    np.testing.assert_allclose(normal[2:-2, 2:-2], expected, atol=1e-12)


def test_depth_normals_require_complete_five_point_stencil() -> None:
    depth = np.full((5, 5), 3.0, np.float64)
    valid = np.ones_like(depth, bool)
    valid[2, 3] = False
    normal = _depth_normals_world(
        depth,
        valid,
        np.asarray([80.0, 80.0]),
        np.asarray([2.0, 2.0]),
        np.eye(4),
    )
    np.testing.assert_array_equal(normal[2, 2], np.zeros(3))


def test_camera_uses_exact_frozen_intrinsics() -> None:
    camera = _camera(np.asarray([91.0, 92.0]), np.asarray([127.5, 71.5]), 256, 144)
    assert camera.model_id == 1
    assert camera.width == 256 and camera.height == 144
    assert camera.params == (91.0, 92.0, 127.5, 71.5)


def test_planar_source_subset_uses_physical_primitive_lineage() -> None:
    source = Official2DGSSource(
        xyz=np.arange(18, dtype=np.float32).reshape(6, 3),
        sh_features=np.zeros((6, 1, 3), np.float32),
        opacity_logits=np.zeros(6, np.float32),
        log_scales_2d=np.zeros((6, 2), np.float32),
        rotations=np.tile([1.0, 0.0, 0.0, 0.0], (6, 1)),
        loc_features=None,
        sh_degree=0,
        path="synthetic.ply",
    )
    physical = type("Physical", (), {"primitive_ids": np.asarray([5, 2, 4])})()
    planar = type(
        "Planar",
        (),
        {
            "member_primitive_rows": np.asarray([2, 0, 2]),
            "validated": lambda self, count: self,
        },
    )()
    selected, rows = _planar_source_subset(source, physical, planar)
    np.testing.assert_array_equal(rows, [4, 5])
    np.testing.assert_array_equal(selected.xyz, source.xyz[[4, 5]])
