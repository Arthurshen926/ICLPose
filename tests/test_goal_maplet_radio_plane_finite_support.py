from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_hybrid_oracle import (
    projected_finite_member_mask,
)


def test_exact_member_projection_preserves_gap_between_disconnected_support() -> None:
    physical = type(
        "Physical",
        (),
        {
            "primitive_centers": np.asarray([[-2.0, 0.0, 5.0], [2.0, 0.0, 5.0]]),
            "primitive_tangent1": np.asarray([[1.0, 0.0, 0.0]] * 2),
            "primitive_tangent2": np.asarray([[0.0, 1.0, 0.0]] * 2),
            "primitive_scale1": np.asarray([0.35, 0.35]),
            "primitive_scale2": np.asarray([0.35, 0.35]),
        },
    )()
    plane = type(
        "Plane",
        (),
        {
            "member_offsets": np.asarray([0, 2]),
            "member_primitive_rows": np.asarray([0, 1]),
            # A convex hull would cover the center; exact members must not.
            "boundary_uv": np.asarray([[-2.35, -0.35], [2.35, -0.35], [2.35, 0.35], [-2.35, 0.35]]),
        },
    )()
    mask = projected_finite_member_mask(
        physical,
        plane,
        0,
        (101, 101),
        np.eye(3),
        np.zeros(3),
        np.asarray([[50.0, 0.0, 50.0], [0.0, 50.0, 50.0], [0.0, 0.0, 1.0]]),
    )
    assert mask[50, 30]
    assert mask[50, 70]
    assert not mask[50, 50]
