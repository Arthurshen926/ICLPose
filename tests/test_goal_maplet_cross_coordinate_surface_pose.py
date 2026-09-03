import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from feature_extract.tools.vfm.select_goal_maplet_cross_coordinate_surface_pose import (
    POINT_SCHEMA,
    SURFACE_SCHEMA,
    _interpolate_pose,
    _paired_contract,
    _select_pareto_line,
)


def _arrays():
    return {
        "names": np.asarray(["q"]),
        "correspondence_offsets": np.asarray([0, 1]),
        "query_tokens": np.asarray([3]),
        "provenance_region_plane_atlas_row": np.asarray([[0, 1, 2]]),
        "prototype_atlas_row": np.asarray([4]),
        "camera_matrices": np.eye(3)[None],
        "radial_k1": np.asarray([0.0]),
        "radio_match_score": np.asarray([0.8]),
    }


def _meta(schema):
    return {
        "artifact_type": schema,
        "plane_uv_atlas_content_sha256": "atlas",
        "plane_ranking_file_sha256": "rank",
        "query_camera_only_inventory_content_sha256": "camera",
        "homography_threshold_m": 0.25,
        "hypotheses_per_query_token": 3,
        "topk_planes": 10,
        "query_support_policy": "uniform",
    }


def test_pairing_accepts_identical_anonymous_match_inventory():
    _paired_contract(_arrays(), _meta(POINT_SCHEMA), _arrays(), _meta(SURFACE_SCHEMA))


def test_pairing_rejects_changed_match_identity():
    point = _arrays(); surface = _arrays(); surface["prototype_atlas_row"] = np.asarray([5])
    with pytest.raises(ValueError, match="prototype_atlas_row"):
        _paired_contract(point, _meta(POINT_SCHEMA), surface, _meta(SURFACE_SCHEMA))


def test_pose_interpolation_preserves_endpoints_and_camera_center_midpoint():
    first = np.eye(4)
    second = np.eye(4)
    second[:3, :3] = Rotation.from_euler("z", 20.0, degrees=True).as_matrix()
    second[:3, 3] = -second[:3, :3] @ np.asarray([2.0, 0.0, 0.0])
    np.testing.assert_allclose(_interpolate_pose(first, second, 0.0), first, atol=1e-12)
    np.testing.assert_allclose(_interpolate_pose(first, second, 1.0), second, atol=1e-12)
    middle = _interpolate_pose(first, second, 0.5)
    center = -middle[:3, :3].T @ middle[:3, 3]
    np.testing.assert_allclose(center, [1.0, 0.0, 0.0], atol=1e-12)


def test_pareto_line_rejects_candidate_that_self_grades():
    scores = np.asarray([[[-2.0, -2.0], [-2.2, -1.0], [-1.9, -1.8]]])
    assert _select_pareto_line(scores).tolist() == [2]


def test_pareto_line_falls_back_when_every_update_hurts_one_arm():
    scores = np.asarray([[[-2.0, -2.0], [-2.1, -1.0], [-1.0, -2.1]]])
    assert _select_pareto_line(scores).tolist() == [0]


def test_fixed_quarter_surface_update_constant_is_not_runtime_tunable():
    from feature_extract.tools.vfm.select_goal_maplet_cross_coordinate_surface_pose import (
        FIXED_QUARTER_SURFACE_FRACTION,
    )
    assert FIXED_QUARTER_SURFACE_FRACTION == 0.25
