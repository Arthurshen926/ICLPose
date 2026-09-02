from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.source_seam_correspondence import (
    SourceSeamConfig,
    SourceSeamCorrespondenceAuthority,
    evaluate_source_seam_geometry,
    freeze_source_seam_correspondences,
)


def _plane_chart(xs: list[float], ys: list[float], z: float = 0.0):
    vertices = np.asarray(
        [(x, y, z) for y in ys for x in xs], dtype=np.float64
    )
    faces = []
    width = len(xs)
    for row in range(len(ys) - 1):
        for column in range(len(xs) - 1):
            top_left = row * width + column
            top_right = top_left + 1
            bottom_left = top_left + width
            bottom_right = bottom_left + 1
            faces.extend(
                (
                    (top_left, bottom_left, top_right),
                    (top_right, bottom_left, bottom_right),
                )
            )
    return vertices, np.asarray(faces, np.int64)


def _two_chart_authority(*, separated: bool = False):
    first_vertices, first_faces = _plane_chart(
        [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]
    )
    second_xs = (
        [4.0, 5.0, 6.0, 7.0]
        if separated
        else [-0.25, 0.75, 1.75, 2.75]
    )
    second_vertices, second_faces = _plane_chart(
        second_xs, [0.0, 1.0, 2.0]
    )
    vertex_offsets = np.asarray(
        [0, len(first_vertices), len(first_vertices) + len(second_vertices)],
        np.int64,
    )
    face_offsets = np.asarray(
        [0, len(first_faces), len(first_faces) + len(second_faces)], np.int64
    )
    faces = np.concatenate((first_faces, second_faces + len(first_vertices)))
    vertices = np.concatenate((first_vertices, second_vertices))
    names = np.asarray(["seq4__a.png", "seq4__b.png"])
    return freeze_source_seam_correspondences(
        chart_names=names,
        chart_vertex_offsets=vertex_offsets,
        sampled_vertex_pixel_indices=np.arange(len(vertices), dtype=np.int64),
        chart_face_offsets=face_offsets,
        faces=faces,
        reference_vertices_world=vertices,
        plan_chart_names=names,
        coverage_edges=np.asarray([[False, True], [True, False]]),
        symmetric_surface_overlap=np.asarray([[1.0, 0.8], [0.8, 1.0]]),
        config=SourceSeamConfig(),
    )


def test_continuous_correspondence_rejects_vertex_sampling_phase_as_thickness():
    authority = _two_chart_authority()
    report = evaluate_source_seam_geometry(
        authority, {"phase_shifted_plane": authority.reference_vertices_world}
    )
    m0 = report["m0_source_reference"]
    arm = report["arms"]["phase_shifted_plane"]

    # A vertex-only matcher sees a 25 cm lattice phase offset.  Continuous
    # target-triangle closest points recover the same plane, so this must not
    # become a surface-thickness failure.
    first = authority.reference_vertices_world[
        authority.chart_vertex_offsets[0] : authority.chart_vertex_offsets[1]
    ]
    second = authority.reference_vertices_world[
        authority.chart_vertex_offsets[1] : authority.chart_vertex_offsets[2]
    ]
    vertex_nn = np.linalg.norm(
        first[:, None, :] - second[None, :, :], axis=2
    ).min(axis=1)
    assert np.median(vertex_nn) == pytest.approx(0.25)
    assert m0["euclidean_distance_p90_m_diagnostic_only"] > 0.10
    assert m0["point_to_plane_p90_m"] == pytest.approx(0.0, abs=1e-12)
    assert m0["m0_all_frozen_edges_reachable"] is True
    assert report["formal_arm_gate_eligible"] is True
    assert arm["formal_source_seam_gate_pass"] is True


def test_fixed_material_correspondence_detects_normal_offset_surface():
    authority = _two_chart_authority()
    shifted = authority.reference_vertices_world.copy()
    second = slice(
        int(authority.chart_vertex_offsets[1]),
        int(authority.chart_vertex_offsets[2]),
    )
    shifted[second, 2] += 0.20
    report = evaluate_source_seam_geometry(authority, {"offset": shifted})
    arm = report["arms"]["offset"]
    assert arm["point_to_plane_p50_m"] == pytest.approx(0.20, abs=1e-12)
    assert arm["all_reachable_edges_geometry_pass"] is False
    assert arm["formal_decision"] == "KILL"


def test_m0_unreachable_edge_disables_formal_arm_gate():
    authority = _two_chart_authority(separated=True)
    report = evaluate_source_seam_geometry(
        authority, {"unchanged": authority.reference_vertices_world}
    )
    assert authority.metadata["m0_reachable_edge_count"] == 0
    assert authority.metadata["m0_unreachable_edge_count"] == 1
    assert report["m0_source_reference"]["m0_all_frozen_edges_reachable"] is False
    assert report["formal_arm_gate_eligible"] is False
    assert report["arms"]["unchanged"]["formal_decision"] == "KILL"


def test_barycentric_material_location_tamper_is_rejected(tmp_path):
    authority = _two_chart_authority()
    path = tmp_path / "authority.npz"
    authority.save_npz(path)
    replayed = SourceSeamCorrespondenceAuthority.load_npz(path)
    changed = replayed.target_barycentric.copy()
    changed[0] = np.asarray([0.8, 0.8, -0.6])
    with pytest.raises(ValueError, match="barycentric"):
        replace(replayed, target_barycentric=changed).validated()
