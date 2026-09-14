import numpy as np
from feature_extract.tools.vfm.select_stable_joint_pose import stable_update


def pose(x):
    p=np.eye(4);p[0,3]=x;return p


def test_agreement_uses_fixed_first_seed():
    result,accepted=stable_update(pose(0),pose(2),pose(2.1),True,True)
    assert accepted and np.array_equal(result,pose(2))


def test_disagreement_or_unconfirmed_restart_preserves_baseline():
    for second,flag in [(pose(4),True),(pose(2.1),False)]:
        result,accepted=stable_update(pose(0),pose(2),second,True,flag)
        assert not accepted and np.array_equal(result,pose(0))
