import numpy as np
import pytest
from feature_extract.tools.vfm.directional_feature_memory import directional_target

@pytest.mark.parametrize('rule',['nearest','kernel'])
def test_directional_memory_padding_permutation_and_world_frame(rule):
    rng=np.random.default_rng(349);m=rng.normal(size=(4,3,8));m/=np.linalg.norm(m,axis=-1,keepdims=True)
    d=rng.normal(size=(4,3,3));d/=np.linalg.norm(d,axis=-1,keepdims=True);world=rng.normal(size=(4,3));pose=np.eye(4);pose[:3,3]=[1,2,3]
    target=directional_target(m,d,np.full(4,3),world,pose,rule)
    order=[2,0,1];assert np.allclose(target,directional_target(m[:,order],d[:,order],np.full(4,3),world,pose,rule))
    padded=np.concatenate([m,np.full((4,1,8),999)],axis=1);pd=np.concatenate([d,np.full((4,1,3),999)],axis=1)
    assert np.allclose(target,directional_target(padded,pd,np.full(4,3),world,pose,rule))
    import cv2
    R=cv2.Rodrigues(np.array([.3,-.2,.1]))[0];t=np.array([5.,-3.,8.]);transform=np.eye(4);transform[:3,:3]=R;transform[:3,3]=t
    assert np.allclose(target,directional_target(m,d@R.T,np.full(4,3),world@R.T+t,pose@np.linalg.inv(transform),rule))

def test_directional_memory_rejects_empty_modes():
    with pytest.raises(ValueError):directional_target(np.zeros((1,2,8)),np.zeros((1,2,3)),np.zeros(1),np.zeros((1,3)),np.eye(4),'kernel')
