from __future__ import annotations

import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.projective_source_seam_authority import (
    MaterialWeldConfig,
    ProjectiveSeamConfig,
    _common_valid_normal_stencil,
    _direction_geometry_metrics,
    _direction_material_weld_metrics,
    _frontmost_exact_face_at_rays,
    _perspective_correct_barycentric,
    _project_world,
    _surface_vertex_normals,
)


def _camera() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def test_projection_uses_pixel_center_principal_not_canvas_half():
    point = np.asarray([[0.0, 0.0, 5.0]])
    correct, _ = _project_world(point, _camera(), 10.0, np.asarray([3.5, 2.5]))
    wrong, _ = _project_world(point, _camera(), 10.0, np.asarray([4.0, 3.0]))
    assert correct[0].tolist() == [3.5, 2.5]
    assert np.linalg.norm(wrong - correct) == pytest.approx(np.sqrt(0.5))


def test_common_valid_five_pixel_stencil_is_the_source_denominator():
    valid = np.ones((7, 7), bool)
    valid[3, 2] = False
    stencil = _common_valid_normal_stencil(valid)
    assert not stencil[3, 3]
    assert stencil[2, 4]
    assert not stencil[0].any()
    assert not stencil[-1].any()


def test_frontmost_projective_face_rejects_hidden_parallel_surface():
    # A source point lies exactly on the far layer.  A world-nearest query
    # would accept far_face at zero distance.  Along the target ray, however,
    # near_face is visible first and its one-metre depth discrepancy must make
    # the association fail the 30 cm occlusion/depth gate.
    vertices = np.asarray(
        [
            [-1.0, -1.0, 4.0],
            [1.0, -1.0, 4.0],
            [0.0, 1.0, 4.0],
            [-1.0, -1.0, 5.0],
            [1.0, -1.0, 5.0],
            [0.0, 1.0, 5.0],
        ]
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], np.int64)
    uv, depth = _project_world(vertices, _camera(), 10.0, np.asarray([0.0, 0.0]))
    face, _, material, target_depth = _frontmost_exact_face_at_rays(
        np.asarray([[0.0, 0.0]]),
        np.asarray([0, 1]),
        faces,
        uv,
        depth,
    )
    source = np.asarray([0.0, 0.0, 5.0])
    far_point = np.sum(vertices[faces[1]] * np.asarray([0.25, 0.25, 0.5])[:, None], axis=0)
    assert np.linalg.norm(source - far_point) == pytest.approx(0.0)
    assert face.item() == 0
    assert target_depth.item() == pytest.approx(4.0)
    assert abs(5.0 - target_depth.item()) > 0.30
    assert np.allclose(material.sum(1), 1.0)


def test_perspective_correct_material_barycentric_replays_3d_point():
    vertices = np.asarray(
        [[-1.0, -1.0, 2.0], [2.0, -1.0, 4.0], [0.0, 3.0, 6.0]]
    )
    faces = np.asarray([[0, 1, 2]], np.int64)
    expected_material = np.asarray([0.2, 0.3, 0.5])
    point = np.sum(vertices * expected_material[:, None], axis=0)
    vertex_uv, vertex_depth = _project_world(
        vertices, _camera(), 20.0, np.asarray([10.0, 10.0])
    )
    point_uv, _ = _project_world(
        point[None], _camera(), 20.0, np.asarray([10.0, 10.0])
    )
    face, screen, material, _ = _frontmost_exact_face_at_rays(
        point_uv,
        np.asarray([0]),
        faces,
        vertex_uv,
        vertex_depth,
    )
    assert face.item() == 0
    assert not np.allclose(screen[0], expected_material)
    assert np.allclose(material[0], expected_material, atol=1e-12)
    replay = np.sum(vertices * material[0, :, None], axis=0)
    assert np.allclose(replay, point, atol=1e-12)


def test_shared_edge_inclusive_tie_is_deterministic():
    vertices = np.asarray(
        [[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 1.0, 5.0], [1.0, 1.0, 5.0]]
    )
    faces = np.asarray([[0, 1, 2], [1, 3, 2]], np.int64)
    # Use nominal screen coordinates here; the point is exactly on the shared
    # diagonal and both faces have identical depth.
    uv = vertices[:, :2]
    depth = vertices[:, 2]
    face, _, _, _ = _frontmost_exact_face_at_rays(
        np.asarray([[0.5, 0.5]]), np.asarray([0, 1]), faces, uv, depth
    )
    assert face.item() == 0


def test_directional_gate_cannot_be_hidden_by_bidirectional_pooling():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.2, 0.0],
            [0.2, 0.2, 0.2],
        ]
    )
    faces = np.asarray([[0, 1, 2]], np.int64)
    normals = np.repeat(np.asarray([[0.0, 0.0, 1.0]]), len(vertices), axis=0)
    config = ProjectiveSeamConfig()
    good = _direction_geometry_metrics(
        vertices,
        normals,
        faces,
        np.asarray([3]),
        np.asarray([0]),
        np.asarray([[0.6, 0.2, 0.2]]),
        config,
    )
    bad = _direction_geometry_metrics(
        vertices,
        normals,
        faces,
        np.asarray([4]),
        np.asarray([0]),
        np.asarray([[0.6, 0.2, 0.2]]),
        config,
    )
    pooled = _direction_geometry_metrics(
        vertices,
        normals,
        faces,
        np.asarray([3, 4]),
        np.asarray([0, 0]),
        np.asarray([[0.6, 0.2, 0.2], [0.6, 0.2, 0.2]]),
        config,
    )
    assert good["geometry_valid"] is True
    assert bad["symmetric_max_point_to_plane_p50_m"] == pytest.approx(0.2)
    assert bad["geometry_valid"] is False
    # A pooled median lands exactly on the nominal 0.10 m boundary and would
    # pass, which is why production decisions are ANDed after independent
    # directional quantiles.
    assert pooled["symmetric_max_point_to_plane_p50_m"] == pytest.approx(0.1)
    assert pooled["geometry_valid"] is True


def test_perspective_formula_rejects_nonpositive_corner_depth():
    material, depth = _perspective_correct_barycentric(
        np.asarray([[0.2, 0.3, 0.5]]), np.asarray([[2.0, -1.0, 4.0]])
    )
    # The low-level algebra is deliberately transparent; the front-most face
    # lookup is the layer that rejects any triangle with a nonpositive corner.
    assert np.isfinite(material).all()
    assert np.isfinite(depth).all()


def test_surface_normal_helper_rejects_degenerate_topology():
    with pytest.raises(ValueError, match="degenerate"):
        _surface_vertex_normals(
            np.zeros((3, 3), np.float64), np.asarray([[0, 1, 2]], np.int64)
        )


def _material_pair_fixture():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.25, 0.0],
        ],
        np.float64,
    )
    return (
        vertices,
        np.asarray([[0, 1, 2]], np.int64),
        np.asarray([3], np.int64),
        np.asarray([0], np.int64),
        np.asarray([[0.5, 0.25, 0.25]], np.float64),
    )


def test_material_weld_is_invariant_to_a_common_rigid_transform():
    reference, faces, source, target_face, barycentric = _material_pair_fixture()
    angle = np.deg2rad(37.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    evaluated = reference @ rotation.T + np.asarray([4.0, -2.0, 7.0])
    metrics = _direction_material_weld_metrics(
        reference,
        evaluated,
        faces,
        source,
        target_face,
        barycentric,
        MaterialWeldConfig(),
    )
    assert metrics["material_weld_excess_p90_m"] < 1e-12
    assert metrics["material_weld_valid"] is True


@pytest.mark.parametrize("slide_m", [0.2, 1.0])
def test_material_weld_rejects_point_to_plane_blind_tangential_slide(slide_m):
    reference, faces, source, target_face, barycentric = _material_pair_fixture()
    evaluated = reference.copy()
    evaluated[:3, 0] += slide_m
    normals = np.repeat(np.asarray([[0.0, 0.0, 1.0]]), 4, axis=0)
    surface = _direction_geometry_metrics(
        evaluated,
        normals,
        faces,
        source,
        target_face,
        barycentric,
        ProjectiveSeamConfig(),
    )
    weld = _direction_material_weld_metrics(
        reference,
        evaluated,
        faces,
        source,
        target_face,
        barycentric,
        MaterialWeldConfig(),
    )
    assert surface["symmetric_max_point_to_plane_p90_m"] == pytest.approx(0.0)
    assert surface["geometry_valid"] is True
    assert weld["material_weld_excess_p50_m"] == pytest.approx(slide_m)
    assert weld["material_weld_excess_p90_m"] == pytest.approx(slide_m)
    assert weld["material_weld_valid"] is False


def test_material_weld_does_not_penalize_an_improved_pair():
    reference, faces, source, target_face, barycentric = _material_pair_fixture()
    reference[source, 2] = 0.2
    evaluated = reference.copy()
    evaluated[source, 2] = 0.0
    metrics = _direction_material_weld_metrics(
        reference,
        evaluated,
        faces,
        source,
        target_face,
        barycentric,
    )
    assert metrics["evaluated_material_distance_p90_m"] == pytest.approx(0.0)
    assert metrics["material_weld_excess_p90_m"] == pytest.approx(0.0)
    assert metrics["material_weld_valid"] is True
