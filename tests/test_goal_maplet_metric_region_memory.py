import numpy as np
from feature_extract.vfm.localization_goal_maplet.metric_region_memory import build_regions,activate_regions,region_features
from feature_extract.tools.vfm.report_goal_maplet_pose_metrics import metrics


def test_physical_regions_translate_with_geometry():
    w=np.random.default_rng(1).normal(size=(100,3));c,m=build_regions(w,2.)
    d,n=build_regions(w+[12.,-5.,2.],2.)
    assert np.allclose(c+[12.,-5.,2.],d)
    assert all(np.array_equal(a,b) for a,b in zip(m,n))


def test_partial_region_activation_and_duplicate_token_evidence():
    w=np.c_[np.arange(12)*.1,np.zeros((12,2))];t=np.arange(12);s=np.ones(12)
    centers=np.array([[.2,0,0],[.8,0,0],[5,0,0]])
    a,ids=activate_regions(w,t,s,centers,.65,2,True)
    b,other=activate_regions(np.repeat(w,3,axis=0),np.repeat(t,3),np.repeat(s,3),centers,.65,2,True)
    assert ids==other and 2 not in ids
    assert all(np.array_equal(np.unique(t[x]),np.unique(np.repeat(t,3)[y])) for x,y in zip(a,b))


def test_coverage_selects_complementary_visible_evidence():
    w=np.c_[np.r_[np.arange(6)*.05,5+np.arange(6)*.05],np.zeros((12,2))]
    groups,ids=activate_regions(w,np.arange(12),np.ones(12),np.array([[0,0,0],[.1,0,0],[5,0,0]]),1.,2,True)
    assert ids==[0,2]


def test_region_features_do_not_multiply_duplicate_votes():
    w=np.random.default_rng(2).normal(size=(12,3));tokens=np.arange(12);scores=np.ones(12)
    a=region_features(w,tokens,scores,np.arange(12),12)
    b=region_features(np.repeat(w,2,axis=0),np.repeat(tokens,2),np.repeat(scores,2),np.arange(24),12)
    assert np.allclose(a,b)
    assert np.allclose(a,region_features(w+15,tokens,scores,np.arange(12),12))


def test_metrics_keep_failed_queries_in_denominator_and_mean():
    report=metrics(np.array([[.05,.2],[np.inf,np.inf]]))
    assert report['translation_mean_m']==np.inf
    assert report['invalid_pose_rate']==.5
    assert report['recall_percent']['0.1m_1deg']==50.
