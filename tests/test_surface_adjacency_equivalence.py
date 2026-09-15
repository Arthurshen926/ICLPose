import numpy as np
import pytest
from feature_extract.vfm.vfm_2dgs_mapping import _build_adjacency


def scalar_reference(centers,normals,radius,threshold,extent,cap):
    effective=np.maximum(extent,0.)
    if cap>0:effective=np.minimum(effective,cap)
    return tuple(np.array([j for j in range(len(centers)) if i!=j and
        float(np.dot(normals[i],normals[j]))>=threshold and
        float(np.linalg.norm(centers[i]-centers[j]))<=float(radius)+float(effective[i])+float(effective[j])],dtype=np.int64)
        for i in range(len(centers)))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('cap',[0.,.1])
def test_matches_scalar_graph(dtype,cap):
    rng=np.random.default_rng(418);c=rng.normal(0,.12,(150,3)).astype(dtype);n=rng.normal(size=(150,3)).astype(dtype);n/=np.linalg.norm(n,axis=1,keepdims=True);e=rng.uniform(-.01,.2,150)
    actual=_build_adjacency(c,n,.05,.8,e,cap);expected=scalar_reference(c,n,.05,.8,e,cap)
    assert all(np.array_equal(x,y) for x,y in zip(actual,expected))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_scalar_decisions_at_threshold(dtype):
    c=np.array([[0,0,0],[.05,0,0],[.05+1e-8,0,0],[.05-1e-8,0,0]],dtype=dtype)
    n=np.array([[1,0,0],[.8,.6,0],[.8+1e-8,.6,0],[.8-1e-8,.6,0]],dtype=dtype);e=np.zeros(4)
    actual=_build_adjacency(c,n,.05,.8,e);expected=scalar_reference(c,n,.05,.8,e,0.)
    assert all(np.array_equal(x,y) for x,y in zip(actual,expected))
