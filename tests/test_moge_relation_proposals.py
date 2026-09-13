import numpy as np
from scipy.spatial.transform import Rotation
from feature_extract.tools.vfm.moge_relation_proposals import distance_dispersion,RelationSampler


def test_distance_relation_invariance_and_mismatch():
    q=np.array([[0.,0,0],[1,0,0],[0,2,0],[0,0,3]])
    r=Rotation.from_rotvec([.3,-.2,.5]).as_matrix();w=2*q@r.T+[10,-2,3];bad=w.copy();bad[3]+=20
    e=distance_dispersion(np.stack([w,bad]),q)
    assert e[0]<1e-20 and e[1]>.01
    np.testing.assert_allclose(e,distance_dispersion(np.stack([w,bad])*3+2,q*7-8),atol=1e-15)
    assert np.isinf(distance_dispersion(np.zeros((1,4,3)),q)[0])


def test_sampler_unique_tokens_shared_budget_invalid_fallback():
    rng=np.random.default_rng(4);q=rng.normal(size=(8,3));world=np.repeat(q,3,axis=0);tokens=np.repeat(np.arange(8),3);groups=[np.flatnonzero(tokens==i) for i in range(8)]
    samplers=[RelationSampler(q,np.zeros(8,bool),p) for p in ['uniform_control','relation','shuffled']]
    draws=[s(np.random.default_rng(5),groups,world,tokens) for s in samplers]
    for s,d in zip(samplers,draws):
        np.testing.assert_array_equal(d,draws[0]);assert len(np.unique(tokens[d]))==4;assert s.quartet_proposals==8 and s.guided==0
