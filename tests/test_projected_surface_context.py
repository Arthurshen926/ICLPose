import numpy as np
from scipy.spatial.transform import Rotation
from feature_extract.tools.vfm.projected_surface_context import project_field,score_pair,exclude_generation,neighbour_valid


def test_projection_duplicate_and_world_frame_invariance():
    world=np.array([[0.,0.,5.],[0.,0.,7.],[.2,.1,5.]])
    features=np.eye(3,dtype=np.float32);directions=np.tile([0,0,-1.],(3,1));k=np.array([[30.,0,5.5],[0,30,5.5],[0,0,1.]])
    p=np.eye(4);f,m=project_field(world,features,directions,p,k,0,grid=(4,4))
    ff,mm=project_field(np.repeat(world,2,axis=0),np.repeat(features,2,axis=0),np.repeat(directions,2,axis=0),p,k,0,grid=(4,4))
    np.testing.assert_array_equal(f,ff);np.testing.assert_array_equal(m,mm)
    g=np.eye(4);g[:3,:3]=Rotation.from_rotvec([.1,.2,-.4]).as_matrix();g[:3,3]=[4.,2.,-3.]
    ff,mm=project_field(world@g[:3,:3].T+g[:3,3],features,directions@g[:3,:3].T,p@np.linalg.inv(g),k,0,grid=(4,4))
    np.testing.assert_allclose(f,ff);np.testing.assert_array_equal(m,mm)


def test_symmetric_domains_budgets_holdout_and_score_exchange():
    rng=np.random.default_rng(310);q=rng.normal(size=(12,16,8));f=[q.copy(),rng.normal(size=q.shape)];m=[np.ones((12,16),bool)]*2;t=np.array([0,33,92])
    a=score_pair(q,f,m,t,budget=16);b=score_pair(q,f[::-1],m[::-1],t,budget=16)
    for k in a:
        assert a[k]['selected_tokens']==b[k]['selected_tokens']
        np.testing.assert_allclose(a[k]['scores'],b[k]['scores'][::-1])
    assert len(a['context_uniform']['selected_tokens'])==len(a['context_disagreement']['selected_tokens'])==16
    assert a['context_disagreement']['selected_tokens']==a['context_reflected']['selected_tokens']
    allowed=exclude_generation(np.ones((12,16),bool),t)
    assert allowed.ravel()[a['context_spatial_holdout']['selected_tokens']].all()
    assert a['context_disagreement']['scores'][0]>a['context_reflected']['scores'][0]


def test_missing_neighbours_and_empty_evidence():
    m=np.ones((4,5),bool);m[2,2]=False
    assert not neighbour_valid(m)[1:4,1:4].any()
    out=score_pair(np.ones((4,5,2)),[np.zeros((4,5,2))]*2,[np.zeros((4,5),bool)]*2,[],4)
    assert all(r['insufficient_evidence'] and r['selected']==0 and r['scores']==[None,None] for r in out.values())


def test_centered_layout_equals_permutation_excess_and_removes_common_component():
    import itertools
    from feature_extract.tools.vfm.projected_surface_context import centered_similarity,spatial_agreement
    rng=np.random.default_rng(313);q=rng.normal(size=(4,8));m=rng.normal(size=(4,8))
    score=centered_similarity(q,m);aligned=np.mean(np.sum(q*m,axis=-1));shuffled=np.mean([np.mean(np.sum(q*m[list(p)],axis=-1)) for p in itertools.permutations(range(4))])
    np.testing.assert_allclose(score,aligned-shuffled)
    np.testing.assert_allclose(score,centered_similarity(q+3,m-2))
    assert not spatial_agreement([1,2,3],[1,1,1])['allow']
    ids=np.array([0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27])
    assert spatial_agreement(ids,np.ones(16))['allow']
    assert not spatial_agreement(ids,np.r_[np.ones(8),-np.ones(8)])['allow']


def test_rescue_requires_actual_reference_identity():
    from feature_extract.tools.vfm.retain_goal_maplet_base_bound_rescue import eligible_rescue
    cur=np.repeat(np.eye(4)[None],4,axis=0);ref=cur.copy();ref[1,0,3]=1e-9
    np.testing.assert_array_equal(eligible_rescue(cur,ref,[1,1,0,1],[1,1,1,0]),[1,0,0,0])


def test_foreground_visibility_scale_symmetry_and_missing():
    from feature_extract.tools.vfm.projected_surface_context import foreground_visibility
    q=np.ones((8,8))*2;z=np.ones((8,8))*10;m=np.ones((8,8),bool)
    q[0,0]=.2
    keep,info=foreground_visibility([z,z*3],[m,m],q,m)
    assert info['sufficient'] and not keep[0,0] and keep.sum()==63
    other,rev=foreground_visibility([z*3,z],[m,m],q*7,m)
    assert np.array_equal(keep,other)
    assert np.allclose(np.asarray(info['scales'])[::-1]/7,rev['scales'])
    empty,info=foreground_visibility([z,z],[m,m],q,np.zeros_like(m))
    assert not empty.any() and not info['sufficient']


def test_projected_depth_is_selected_geometry():
    from feature_extract.tools.vfm.projected_surface_context import project_field
    world=np.array([[0.,0.,2.],[0.,0.,4.]])
    features=np.eye(2,dtype=np.float32);directions=np.array([[0.,0.,-1.]]*2)
    K=np.eye(3);K[:2,2]=1.5
    f,m,d=project_field(world,features,directions,np.eye(4),K,0,grid=(1,1),return_depth=True)
    assert m[0,0] and d[0,0]==2 and np.allclose(f[0,0],[1,0])
