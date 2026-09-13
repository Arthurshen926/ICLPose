"""Properties of existing geometric evidence, not claims of global uniqueness."""
import numpy as np
from scipy.spatial.transform import Rotation
from feature_extract.tools.vfm.token_hypothesis_ransac import canonical_hypotheses,score_pose


def fixture():
    rng=np.random.default_rng(305)
    w=rng.normal(size=(16,3));w[:,2]+=6
    t=np.repeat(np.arange(8),2)
    k=np.array([[150.,0,128],[0,150.,72],[0,0,1]])
    xy=w[:,:2]/w[:,2,None]*150+[128,72]
    xy[1::2]+=15
    return w,t,xy,k


def scored(w,t,xy,k,pose):
    w,t,xy=canonical_hypotheses(w,t,xy)
    groups=[np.flatnonzero(t==u) for u in np.unique(t)]
    return score_pose(pose,w,xy,groups,k,0)[0]


def test_world_rigid_transform_preserves_geometric_evidence():
    w,t,xy,k=fixture();pose=np.eye(4)
    g=np.eye(4);g[:3,:3]=Rotation.from_rotvec([.4,-.2,.3]).as_matrix();g[:3,3]=[30,-12,4]
    transformed=w@g[:3,:3].T+g[:3,3]
    original=scored(w,t,xy,k,pose);changed=scored(transformed,t,xy,k,pose@np.linalg.inv(g))
    np.testing.assert_allclose(original,changed,atol=1e-18)


def test_duplicate_alternatives_do_not_multiply_token_votes():
    w,t,xy,k=fixture();pose=np.eye(4)
    base=scored(w,t,xy,k,pose)
    duplicated=scored(np.repeat(w,4,axis=0),np.repeat(t,4),np.repeat(xy,4,axis=0),k,pose)
    assert base==duplicated
    assert base[0]==8


def test_removing_explanations_cannot_increase_fixed_token_support():
    w,t,xy,k=fixture();pose=np.eye(4)
    base=scored(w,t,xy,k,pose)
    keep=np.ones(len(t),bool);keep[::4]=False
    reduced=scored(w[keep],t[keep],xy[keep],k,pose)
    assert reduced[0]<=base[0]
    # Removing a correct explanation retains its token's wrong explanation.
    assert reduced[1]<=base[1]


def test_diverse_candidates_use_camera_centers_and_are_frame_invariant():
    from feature_extract.tools.vfm.export_goal_maplet_pnp_endpoint import diverse_indices
    ref=np.eye(4);poses=np.repeat(ref[None],4,axis=0)
    poses[1,0,3]=2.;poses[2,:3,:3]=Rotation.from_euler('z',20,degrees=True).as_matrix();poses[3]=np.nan
    np.testing.assert_array_equal(diverse_indices(poses,ref),[1,2])
    g=np.eye(4);g[:3,:3]=Rotation.from_euler('y',33,degrees=True).as_matrix();g[:3,3]=[15,-4,10]
    np.testing.assert_array_equal(diverse_indices(poses@np.linalg.inv(g),ref@np.linalg.inv(g)),[1,2])


def test_dense_control_symmetry_missing_and_ties():
    from feature_extract.tools.vfm.select_goal_maplet_full_dense_control import select_dense
    scores=np.array([[.2,.4],[.5,.1],[0,0],[-np.inf,-np.inf]])
    valid=np.ones((4,2),bool)
    choice,missing=select_dense(scores,valid)
    np.testing.assert_array_equal(choice,[1,0,0,0]);np.testing.assert_array_equal(missing,[0,0,1,1])
    reverse,_=select_dense(scores[:,::-1],valid)
    np.testing.assert_array_equal(reverse[:2],1-choice[:2])
    valid[1,0]=False
    assert select_dense(scores,valid)[0][1]==1


def test_dense_control_rejects_unbounded_confidence():
    import pytest
    from feature_extract.tools.vfm.select_goal_maplet_full_dense_control import select_dense
    for value in [np.nan,np.inf,1.01,-.1]:
        with pytest.raises(ValueError):select_dense([[.3,value]],[[True,True]])
