import cv2
import numpy as np
from feature_extract.tools.vfm.cambridge_core_readout_control import global_refit
from feature_extract.tools.vfm.token_hypothesis_ransac import score_pose


def test_common_readout_recovers_known_pose_and_never_reduces_consensus():
    rng=np.random.default_rng(415)
    world=rng.uniform([-2.,-1.,5.],[2.,1.,12.],size=(50,3))
    K=np.array([[220.,0.,128.],[0.,220.,72.],[0.,0.,1.]])
    pixels=cv2.projectPoints(world,np.zeros(3),np.zeros(3),K,np.zeros(5))[0].reshape(-1,2)
    tokens=np.arange(50);groups=[np.array([i]) for i in tokens]
    initial=np.eye(4);initial[:3,3]=[.04,-.03,.04]
    result,info=global_refit(initial,world,tokens,pixels,K,0.)
    assert score_pose(result,world,pixels,groups,K,0.)[0]>=score_pose(initial,world,pixels,groups,K,0.)[0]
    assert np.linalg.norm(result[:3,3])<1e-6
    assert info['accepted_steps']>=1


def test_common_readout_preserves_invalid_endpoint_and_insufficient_support():
    K=np.eye(3);world=np.array([[1.,1.,5.]])
    invalid=np.full((4,4),np.nan)
    result,info=global_refit(invalid,world,np.array([0]),np.array([[.2,.2]]),K,0.)
    assert np.isnan(result).all() and info['accepted_steps']==0
    initial=np.eye(4)
    result,info=global_refit(initial,world,np.array([0]),np.array([[.2,.2]]),K,0.)
    np.testing.assert_array_equal(result,initial)
    assert info['accepted_steps']==0
