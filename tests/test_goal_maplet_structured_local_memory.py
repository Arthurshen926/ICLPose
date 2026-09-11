import numpy as np
from feature_extract.vfm.localization_goal_maplet.structured_local_memory import sector_descriptors,arrangement_scores,relative_shape_features


def test_arrangement_distinguishes_reversal_with_same_unordered_content():
    q=np.eye(4)[None];m=q[:,[1,0,2,3]]
    score=arrangement_scores(q,m)[0];same=arrangement_scores(q,q)[0]
    assert score[0]==same[0] and score[1]<same[1]
    assert score[2]<0 and same[2]>0


def test_sectors_do_not_wrap_or_add_centre():
    grid=np.zeros((5,5,2),np.float32);grid[0,0]=[1,0];grid[-1,-1]=[0,1]
    s=sector_descriptors(grid,1)
    assert not s[0,0].any()
    assert np.array_equal(s[0,1,0],[1,0])
    assert not s[0,1,:,1].any()


def test_relative_shape_is_rigid_scale_and_normal_sign_invariant():
    tokens=np.array([0,2,4,6]);p=np.zeros((2304,3));p[tokens]=np.c_[np.arange(4),np.zeros(4),np.ones(4)]
    n=np.tile([0.,0,1],(2304,1));v=np.zeros(2304,bool);v[tokens]=True
    w=p[tokens].copy();mn=n[tokens].copy();cos=np.ones(4)
    a=relative_shape_features(tokens,p,n,v,w,mn,cos)
    rot=np.array([[0.,0,1],[0,1,0],[-1,0,0]])
    b=relative_shape_features(tokens,p*7@rot,n@rot,v,w+100,-mn,cos)
    np.testing.assert_allclose(a,b,atol=1e-6)
    assert (a[:,:2]>.99).all()


def test_duplicate_explanations_do_not_multiply_relational_votes():
    tokens=np.array([0,2,4,6]);p=np.zeros((2304,3));p[tokens]=np.c_[np.arange(4),np.zeros(4),np.ones(4)]
    n=np.tile([0.,0,1],(2304,1));v=np.zeros(2304,bool);v[tokens]=True
    w=p[tokens];mn=n[tokens];cos=np.ones(4)
    a=relative_shape_features(tokens,p,n,v,w,mn,cos)
    ids=np.array([0,1,1,2,3]);b=relative_shape_features(tokens[ids],p,n,v,w[ids],mn[ids],cos[ids])
    np.testing.assert_allclose(a,b[[0,1,3,4]],atol=1e-6)


def test_moge_geometry_rejects_pose_bearing_or_undeclared_cache(tmp_path):
    import json
    import pytest
    from feature_extract.tools.vfm.build_goal_maplet_structured_memory_features import moge_tokens
    p=tmp_path/'geometry.npz'
    for meta in [{},{'uses_pose':True,'uses_ground_truth':False}]:
        np.savez(p,points_camera=np.ones((144,256,3)),normal_camera=np.ones((144,256,3)),valid=np.ones((144,256),bool),metadata_json=np.asarray(json.dumps(meta)))
        with pytest.raises(ValueError,match='pose-free'):moge_tokens(p)
