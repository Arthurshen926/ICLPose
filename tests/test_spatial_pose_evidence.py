import cv2
import numpy as np
from feature_extract.tools.vfm.spatial_pose_evidence import evidence


def test_duplicate_alternatives_do_not_increase_evidence():
    rng=np.random.default_rng(4);w=rng.normal(size=(24,3));w[:,2]+=8
    K=np.array([[120.,0,128],[0,120,72],[0,0,1.]])
    xy=cv2.projectPoints(w,np.zeros(3),np.zeros(3),K,None)[0].reshape(-1,2)
    t=np.arange(len(w));p=np.eye(4)
    a=evidence(p,w,t,xy,K,0)
    b=evidence(p,np.repeat(w,3,axis=0),np.repeat(t,3),np.repeat(xy,3,axis=0),K,0)
    assert np.allclose(a,b)


def test_local_information_is_independent_of_world_origin():
    rng=np.random.default_rng(8);w=rng.normal(size=(24,3));w[:,2]+=8
    K=np.array([[120.,0,128],[0,120,72],[0,0,1.]])
    xy=cv2.projectPoints(w,np.zeros(3),np.zeros(3),K,None)[0].reshape(-1,2);p=np.eye(4)
    a=evidence(p,w,np.arange(24),xy,K,0);shift=np.array([1000.,-900.,500.]);p[:3,3]=-shift
    b=evidence(p,w+shift,np.arange(24),xy,K,0)
    assert np.allclose(a,b,atol=1e-8)


def test_equal_count_concentrated_support_has_lower_coverage():
    rng=np.random.default_rng(12);w=rng.normal(size=(32,3));w[:,2]+=10
    K=np.array([[120.,0,128],[0,120,72],[0,0,1.]])
    def measure(points):
        xy=cv2.projectPoints(points,np.zeros(3),np.zeros(3),K,None)[0].reshape(-1,2)
        return evidence(np.eye(4),points,np.arange(32),xy,K,0)
    wide=measure(w);w[:,:2]*=.05;narrow=measure(w)
    assert wide[1]==narrow[1]==1
    assert wide[4]>narrow[4] and wide[7]<narrow[7]


def test_invalid_pose_never_has_positive_support():
    a=evidence(None,np.empty((0,3)),np.array([],int),np.empty((0,2)),np.eye(3),0)
    assert np.isfinite(a).all() and a[1]==0
