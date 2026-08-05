import numpy as np

from feature_extract.vfm.localization_goal_maplet.pose_proposal import CoarsePoseModes


def test_pose_mode_contract_accepts_empty_modes():
    result = CoarsePoseModes(
        poses_w2c=np.zeros((0, 4, 4), dtype=np.float64),
        scores=np.zeros((0,), dtype=np.float64),
        supporting_region_count=np.zeros((0,), dtype=np.int64),
    )
    assert result.poses_w2c.shape == (0, 4, 4)


def test_pose_mode_contract_rejects_misaligned_values():
    try:
        CoarsePoseModes(np.zeros((1, 4, 4)), np.zeros((2,)), np.zeros((1,)))
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned pose modes must fail")
