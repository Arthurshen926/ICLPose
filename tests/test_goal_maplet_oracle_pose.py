import numpy as np

from feature_extract.vfm.localization_goal_maplet import build_goal_maplet_physical_map
from feature_extract.vfm.localization_goal_maplet.oracle_pose import (
    grouped_oracle_correspondences,
    token_oracle_evidence,
)
from feature_extract.vfm.localization_goal_maplet.pfir import ContributorLabels
from test_goal_maplet_physical_map import _inputs


def test_oracle_evidence_resolves_parent_child_and_local_surface():
    geometry, maplets, region, poses = _inputs()
    physical = build_goal_maplet_physical_map(maplets, region, geometry, poses, minimum_child_count=2, maximum_child_count=4)
    labels = ContributorLabels(
        np.asarray([[[0], [1]], [[2], [8]]]),
        np.ones((2, 2, 1), dtype=np.float32),
        np.eye(4),
    )
    evidence = token_oracle_evidence(
        labels,
        physical,
        np.asarray([[0, 0], [1, 1]]),
        token_height=2,
        token_width=2,
        image_height=200,
        image_width=200,
    )
    assert evidence.parent_rows[0] == 0
    assert evidence.child_rows[0] >= 0
    assert evidence.parent_rows[1] == -1
    correspondence = grouped_oracle_correspondences(evidence, physical, np.asarray([0, 1, 2]), np.asarray([0, 1]))
    assert correspondence["child_local"][0].shape == (1, 2)
    np.testing.assert_allclose(correspondence["child_local"][1][0], physical.primitive_centers[0])
