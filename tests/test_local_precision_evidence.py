import numpy as np
from feature_extract.tools.vfm.local_precision_evidence import paired_features


def test_pairwise_evidence_swap_and_identity():
    rng=np.random.default_rng(11);grid=rng.normal(size=(36,64,64)).astype(np.float32);grid/=np.linalg.norm(grid,axis=-1,keepdims=True)
    xy=np.c_[np.linspace(40,200,16),np.linspace(35,110,16)];K=np.array([[100.,0,128],[0,100,72],[0,0,1.]])
    w=np.c_[(xy[:,0]-128)/100*4,(xy[:,1]-72)/100*4,np.full(16,4.)];tokens=np.arange(16);a=np.eye(4);b=a.copy();b[0,3]=.01;target=rng.normal(size=(16,64))
    args=(w,xy,tokens,K,0,[grid,grid],[target,target]);x,n=paired_features(a,b,*args);y,m=paired_features(b,a,*args)
    assert n==m==16 and np.allclose(x,-y,atol=1e-8)
    z,n=paired_features(a,a,*args);assert not z.any()


def test_no_shared_support_abstains():
    args=(np.array([[0.,0.,-1.]]),np.zeros((1,2)),np.array([0]),np.eye(3),0,[],[])
    x,n=paired_features(np.eye(4),np.eye(4),*args);assert n==0 and not x.any()
