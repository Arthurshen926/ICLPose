import numpy as np
import pytest
from feature_extract.tools.vfm.refine_goal_maplet_global_token_lm import refine


@pytest.mark.parametrize("loss", ["linear", "soft_l1"])
def test_global_lm_corrects_pose_and_ignores_duplicate_order(loss):
    rng=np.random.default_rng(43);world=rng.uniform(-1,1,(30,3));world[:,2]+=5
    K=np.array([[150.,0,128],[0,150,72],[0,0,1.]])
    pixels=world[:,:2]/world[:,2,None]*150+np.array([128,72]);tok=np.arange(30)
    pose=np.eye(4);pose[:3,3]=[.04,-.03,.02]
    result,n=refine(pose,world,pixels,tok,K,0.,loss=loss)
    assert n>0 and np.linalg.norm(result-np.eye(4))<1e-5
    order=rng.permutation(np.tile(np.arange(30),3))
    duplicate,m=refine(pose,world[order],pixels[order],tok[order],K,0.,loss=loss)
    assert np.array_equal(result,duplicate) and n==m


def test_global_lm_keeps_unusable_input():
    pose=np.eye(4);result,n=refine(pose,np.zeros((0,3)),np.zeros((0,2)),np.zeros(0,int),np.eye(3),0.)
    assert np.array_equal(result,pose) and n==0
