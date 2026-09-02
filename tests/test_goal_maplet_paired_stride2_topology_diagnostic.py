from __future__ import annotations

import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.chart_comparison_domain import (
    _one_stride_topology,
)
from feature_extract.vfm.localization_goal_maplet.paired_stride2_topology_diagnostic import (
    GEOMETRY_NAMES,
    PairedStride2TopologyDiagnostic,
    require_stride4_bit_parity,
    unit_edge_safe_face_masks,
)
from feature_extract.vfm.localization_goal_maplet.projective_source_seam_authority import (
    ProjectiveSeamConfig,
    _chunked_frontmost_exact_face_at_rays,
    _frontmost_exact_face_at_rays,
)


def _flat_points(height: int, width: int) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    return np.stack((0.1 * x, 0.1 * y, np.full_like(x, 10.0)), axis=2)[
        None
    ].astype(np.float64)


def _metadata() -> dict[str, object]:
    return {
        "artifact_type": "goal_maplet_paired_stride2_topology_diagnostic_v1",
        "paired_comparison_topology": True,
        "model_neutral_map_topology_claimed": False,
        "uses_query_or_ground_truth": False,
        "held_geometry_consumed": False,
        "aligned_arm_geometry_consumed": False,
        "stride4_control_bit_equal_corrected_v3": True,
        "stride2_repacked_without_orphans": True,
        "source_eligible_definition_stride_invariant": True,
        "sampled_source_inventory_stride_invariant": False,
        "support_fractions_only_comparable_within_same_stride": True,
        "geometry_names": list(GEOMETRY_NAMES),
        "edge_safety": {
            "absolute_edge_threshold_m": 0.5,
            "relative_edge_threshold": 0.05,
            "threshold_formula": "max(0.5m,0.05*median_camera_centered_range)",
            "edge_inventory": "all_unit_pixel_edges_in_inclusive_stride_patch",
        },
    }


def test_three_geometry_edge_safety_and_exact_repack() -> None:
    valid = np.ones((1, 9, 9), bool)
    base = _flat_points(9, 9)
    unsafe = base.copy()
    unsafe[0, 2, 2, 0] += 5.0
    point_sets = {
        "DAV2_clean_v2": base,
        "MoGe3": unsafe,
        "source_MASt3R_reference": base.copy(),
    }
    strict4, individual4 = unit_edge_safe_face_masks(
        valid, point_sets, np.zeros((1, 3)), stride=4
    )
    assert individual4["DAV2_clean_v2"].sum() == 4
    assert individual4["MoGe3"].sum() < 4
    assert np.array_equal(
        strict4,
        np.logical_and.reduce([individual4[name] for name in GEOMETRY_NAMES]),
    )

    strict2, _ = unit_edge_safe_face_masks(
        valid, point_sets, np.zeros((1, 3)), stride=2
    )
    topology4 = _one_stride_topology(valid, strict4, stride=4)
    topology2 = _one_stride_topology(valid, strict2, stride=2)
    diagnostic = PairedStride2TopologyDiagnostic(
        chart_names=np.asarray(["chart.png"]),
        valid=valid,
        face_valid_stride4_control=strict4,
        sampled_vertex_offsets_stride4_control=topology4[
            "sampled_vertex_offsets_stride4"
        ],
        sampled_vertex_pixel_indices_stride4_control=topology4[
            "sampled_vertex_pixel_indices_stride4"
        ],
        face_offsets_stride4_control=topology4["face_offsets_stride4"],
        faces_stride4_control=topology4["faces_stride4"],
        face_valid_stride2=strict2,
        metadata=_metadata(),
        **topology2,
    )
    # A formal artifact requires at least two charts, but exact repacking itself
    # must still use every packed vertex and no others.
    used = np.unique(diagnostic.faces_stride2)
    assert np.array_equal(
        used, np.arange(len(diagnostic.sampled_vertex_pixel_indices_stride2))
    )


def test_generic_stride4_parity_gate_fails_closed() -> None:
    valid = np.ones((1, 9, 9), bool)
    face = np.ones((1, 2, 2), bool)
    target = {"face_valid_stride4": face.copy(), **_one_stride_topology(valid, face, stride=4)}
    control = require_stride4_bit_parity(valid, face, target)
    assert control["faces_stride4_control"].shape == (8, 3)
    target["faces_stride4"] = target["faces_stride4"].copy()
    target["faces_stride4"][0] = target["faces_stride4"][0, ::-1]
    with pytest.raises(ValueError, match="differs from corrected v3"):
        require_stride4_bit_parity(valid, face, target)


def test_frontmost_chunk_order_is_bit_identical() -> None:
    # Two screen-identical triangles at z=2 and z=4 exercise front-most depth.
    faces = np.asarray(((0, 1, 2), (3, 4, 5)), np.int64)
    uv = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)) * 2, np.float64
    )
    depth = np.asarray((2.0, 2.0, 2.0, 4.0, 4.0, 4.0), np.float64)
    rays = np.asarray(((0.2, 0.2), (0.1, 0.4), (2.0, 2.0)), np.float64)
    direct = _frontmost_exact_face_at_rays(
        rays, np.asarray((0, 1)), faces, uv, depth
    )
    forward = _chunked_frontmost_exact_face_at_rays(
        rays,
        np.asarray((0, 1)),
        faces,
        uv,
        depth,
        maximum_ray_face_pairs=2,
    )
    reverse = _chunked_frontmost_exact_face_at_rays(
        rays,
        np.asarray((0, 1)),
        faces,
        uv,
        depth,
        maximum_ray_face_pairs=2,
        reverse_chunk_order=True,
    )
    for expected, observed, reversed_observed in zip(direct, forward, reverse):
        assert np.array_equal(expected, observed, equal_nan=True)
        assert np.array_equal(expected, reversed_observed, equal_nan=True)
    assert forward[0].tolist() == [0, 0, -1]


def test_stride2_config_is_diagnostic_only() -> None:
    assert ProjectiveSeamConfig(topology_stride=2).validated().topology_stride == 2
    with pytest.raises(ValueError, match="supports only"):
        ProjectiveSeamConfig(topology_stride=1).validated()
