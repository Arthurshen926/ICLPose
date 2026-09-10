import numpy as np
from feature_extract.tools.vfm.audit_goal_maplet_mapping_pose_bias import linear_bias


def test_shared_pose_projection_recovers_known_small_bias():
    rng=np.random.default_rng(8);j=rng.normal(size=(20,2,6));delta=np.array([.001,.002,0,.01,.02,.03])
    r=np.einsum('nij,j->ni',j,delta)
    result=linear_bias(j,r,7)
    np.testing.assert_allclose(result['translation_bias_m'],np.linalg.norm(delta[3:]))
    np.testing.assert_allclose(result['rotation_bias_deg'],np.rad2deg(np.linalg.norm(delta[:3])))


def test_rank_deficient_shared_pose_is_not_claimed_observable():
    assert linear_bias(np.zeros((8,2,6)),np.ones((8,2)),5) is None
