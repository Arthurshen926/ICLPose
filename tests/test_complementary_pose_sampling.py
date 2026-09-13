import numpy as np
from feature_extract.tools.vfm.complementary_pose_sampling import query_information,ComplementarySampler


def test_information_is_psd_and_invariant_to_moge_global_scale():
    rng=np.random.default_rng(5);p=rng.normal(size=(64,3));p[:,2]+=5
    K=np.array([[100.,0,128],[0,100,72],[0,0,1.]])
    a=query_information(p,np.ones(64,bool),K,0);b=query_information(p*7,np.ones(64,bool),K,0)
    assert np.allclose(a,b,atol=1e-10)
    assert np.linalg.eigvalsh(a).min()>-1e-10


def test_invalid_depth_has_no_information():
    p=np.ones((8,3));p[2]=np.nan;p[3,2]=-1
    H=query_information(p,np.ones(8,bool),np.eye(3),0)
    assert np.isfinite(H).all() and not H[2].any() and not H[3].any()


def test_sampler_never_duplicates_a_query_token_and_keeps_exploration():
    groups=[np.array([2*i,2*i+1]) for i in range(12)];tokens=np.repeat(np.arange(12),2)
    information=np.zeros((12,6,6));information[:4]=np.eye(6)*100
    sampler=ComplementarySampler(information);rng=np.random.default_rng(17);seen=np.zeros(12,int)
    for _ in range(500):
        rows=sampler(rng,groups,np.zeros((24,3)),tokens)
        assert len(set(tokens[rows]))==4;seen[tokens[rows]]+=1
    assert seen.min()>30
