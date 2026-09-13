import numpy as np
from feature_extract.tools.vfm.partial_visibility_memory import diverse_tokens,local_modes,retrieve_local_regions


def test_rare_appearance_survives_repetitive_foreground():
    features=np.tile([1.,0.],(100,1));features[51]=[0.,1.]
    selected=diverse_tokens(features,np.ones(100,bool),10)
    assert 51 in selected and len(np.unique(selected))==10
    valid=np.ones(100,bool);valid[51]=False
    assert 51 not in diverse_tokens(features,valid,10)


def test_local_identity_can_retrieve_despite_small_visible_area():
    grid=np.zeros((12,12,2));grid[...,0]=1;grid[1,1]=[0,1]
    ids,evidence=retrieve_local_regions(grid,np.eye(2),np.array([[0.,0,0],[10.,0,0]]),2)
    assert set(ids)=={0,1}
    assert evidence[1]['similarity']>.99


def test_local_modes_preserve_identity_and_exclude_unavailable():
    world=np.zeros((6,3));context=np.array([[1,0],[1,0],[0,1],[0,1],[-1,0],[0,-1.]])
    available=np.array([1,1,1,1,0,0],bool)
    rows=local_modes(world,context,available,modes_per_voxel=2)
    assert len(rows)==2 and available[rows].all()
    assert np.array_equal(np.sort(context[rows],axis=0),np.sort(np.eye(2),axis=0))


def test_region_selection_obeys_world_separation_and_empty_budget():
    grid=np.tile([1.,0.],(12,12,1));ids,_=retrieve_local_regions(grid,np.tile([1.,0.],(3,1)),np.array([[0.,0,0],[1.,0,0],[10.,0,0]]),3)
    assert ids==[0,2]
    assert len(diverse_tokens(np.eye(2),np.zeros(2,bool)))==0


def test_cross_parity_depth_consistency_and_missing_support():
    from feature_extract.tools.vfm.verify_goal_maplet_partial_candidates import depth_support
    tokens=np.arange(24);world=np.c_[np.linspace(-1,1,24),np.zeros(24),np.linspace(4,8,24)];query=world/3;K=np.eye(3);pixels=world[:,:2]/world[:,2,None];groups=[np.array([i]) for i in tokens];valid=np.ones(24,bool)
    score,scale=depth_support(np.eye(4),world,tokens,query,valid,pixels,groups,K,0.)
    assert score==12 and np.isclose(scale,np.log(3))
    wrong=query.copy();wrong[1::2,2]*=2
    assert depth_support(np.eye(4),world,tokens,wrong,valid,pixels,groups,K,0.)[0]==0
    valid[:20]=False
    assert depth_support(np.eye(4),world,tokens,query,valid,pixels,groups,K,0.)==(0,None)


def test_query_relative_unknown_is_per_token_and_not_cross_query():
    from feature_extract.tools.vfm.surface_configuration_matching import configuration_match
    u=np.array([[.4,.2],[.4,.2]]);w=np.tile(np.array([[[0.,0,1],[1.,0,1]]]),(2,1,1));n=np.zeros_like(w);qp=np.array([[0.,0,1],[1.,0,1]]);qn=np.zeros_like(qp)
    chosen,valid,belief,_=configuration_match(u,w,n,qp,qn,np.array([[0.,0],[1.,0]]),policy='independent',absolute_null=np.array([.3,.5]))
    assert valid.tolist()==[True,False]
    assert chosen.tolist()==[0,2]


def test_pose_modes_use_camera_centers_and_backend_trust_limits():
    from feature_extract.tools.vfm.select_goal_maplet_separated_modes import separated_mode
    base=np.eye(4);near=base.copy();near[0,3]=.49;far=base.copy();far[0,3]=.51
    assert not separated_mode(base,near)
    assert separated_mode(base,far)
