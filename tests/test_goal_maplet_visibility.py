import numpy as np

from feature_extract.vfm.localization_goal_maplet.physical_map import DOUBLE_SIDED, SINGLE_SIDED
from feature_extract.vfm.localization_goal_maplet import build_goal_maplet_physical_map
from feature_extract.vfm.localization_goal_maplet.surface_renderer import dominant_child_owner
from feature_extract.vfm.localization_goal_maplet.visibility import signed_surface_visibility
from test_goal_maplet_physical_map import _inputs


def test_signed_visibility_rejects_backside_but_double_sided_is_explicit():
    centers = np.zeros((3, 3), dtype=np.float64)
    normals = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    sidedness = np.asarray([SINGLE_SIDED, SINGLE_SIDED, DOUBLE_SIDED])
    pose = np.eye(4)
    pose[2, 3] = -2.0  # camera center = +2 z
    visible, incidence = signed_surface_visibility(centers, normals, sidedness, pose)
    assert visible.tolist() == [True, False, True]
    np.testing.assert_allclose(incidence, [1.0, 0.0, 1.0])


def test_dominant_child_owner_is_defined_only_for_owned_primitives():
    geometry, maplets, region, poses = _inputs()
    physical = build_goal_maplet_physical_map(
        maplets, region, geometry, poses, minimum_child_count=2, maximum_child_count=4
    )
    owner = dominant_child_owner(physical)
    cached = dominant_child_owner(physical)
    assert np.all(owner[:8] >= 0)
    assert np.all(owner[8:] == -1)
    assert cached is owner
    assert not owner.flags.writeable
