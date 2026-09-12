import numpy as np
from feature_extract.tools.vfm.native_pose_conditioned_region_value import token_residuals,pose_features


def test_tokens_vote_once_and_behind_camera_cannot_support():
    world=np.array([[1.5,1.5,1],[1.5,1.5,1],[0,0,-1.]])
    t,e,p=token_residuals(world,np.array([0,0,1]),np.eye(4),np.eye(3),0.)
    np.testing.assert_array_equal(t,[0,1])
    assert e[0]==0 and np.isinf(e[1])
    np.testing.assert_array_equal(p,[True,False])


def test_novel_support_and_invalid_pose():
    rt=[np.array([],int) for _ in range(16)];rp=[x.copy() for x in rt];rs=[np.array([]) for _ in rt]
    rt[8]=np.array([1]);rp[8]=np.array([1]);rs[8]=np.array([.9])
    world=np.array([[1.5,1.5,1],[5.5,1.5,1]])
    args=(np.array([0]),np.array([0]),rt,rp,rs,world)
    f=pose_features(*args,np.eye(4),np.eye(3),0.)
    assert f.shape==(8,14) and np.isfinite(f).all()
    assert f[0,9]==1/2304 and f[0,10]==1/2304 and f[0,11]==0
    assert np.all(f[1:,9:12]==0)
    assert not pose_features(*args,np.full((4,4),np.nan),np.eye(3),0.).any()


def test_budget_can_remove_existing_geometric_support():
    rt=[np.array([],int) for _ in range(16)];rp=[x.copy() for x in rt];rs=[np.array([]) for _ in rt]
    rt[0]=np.arange(1024);rp[0]=np.arange(1024);rs[0]=np.full(1024,.5)
    rt[8]=np.arange(1024,2048);rp[8]=np.arange(1024,2048);rs[8]=np.ones(1024)
    tokens=np.arange(2048)
    world=np.c_[(tokens%64)*4+1.5,(tokens//64)*4+1.5,np.ones(2048)]
    world[1024:,2]=-1
    f=pose_features(np.array([],int),np.array([],int),rt,rp,rs,world,np.eye(4),np.eye(3),0.)
    assert f[0,10]==0 and f[0,11]==1024/2304
    assert f[0,8]==1 and np.all(f[1:,10:12]==0)


def test_projection_matches_backend_with_radial_distortion():
    from feature_extract.tools.vfm.refine_goal_maplet_plane_pose_with_uncertainty import _project
    world=np.array([[.2,.1,2.],[-.5,.3,3.]])
    K=np.array([[180.,0,128.],[0,175.,72.],[0,0,1.]])
    pose=np.eye(4);pose[:3,3]=[.1,-.2,.3]
    tokens=np.array([12,1200]);xy,_=_project(pose,world,K,.03)
    expected=np.linalg.norm(xy-np.c_[(tokens%64)*4+1.5,(tokens//64)*4+1.5],axis=1)
    _,actual,_=token_residuals(world,tokens,pose,K,.03)
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
