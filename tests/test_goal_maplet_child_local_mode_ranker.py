import numpy as np

from feature_extract.vfm.localization_goal_maplet.child_local_mode_ranker import FEATURE_NAMES


def test_child_local_mode_ranker_feature_contract_has_no_absolute_pose():
    assert len(FEATURE_NAMES) == 11
    assert not any("world" in name or "absolute" in name for name in FEATURE_NAMES)
    assert "normalized_reprojection_residual" in FEATURE_NAMES
    assert "vfm_geometry_joint_score" in FEATURE_NAMES
