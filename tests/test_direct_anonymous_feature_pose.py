import cv2
import numpy as np
from feature_extract.tools.vfm.direct_anonymous_feature_pose import sample_with_gradient,objective,refine


def test_feature_gradient_and_pose_gradient():
    rng=np.random.default_rng(9);g=rng.normal(size=(20,30,8));p=np.array([[90.13,50.12],[130.44,80.34]])
    f,j=sample_with_gradient(g,p)
    for k in range(2):
        d=np.zeros_like(p);d[:,k]=1e-5
        numeric=(sample_with_gradient(g,p+d)[0]-sample_with_gradient(g,p-d)[0])/2e-5
        assert np.allclose(numeric,j[:,:,k],atol=1e-7)
    K=np.array([[150.,0,128],[0,150,72],[0,0,1]])
    world=np.c_[(p-K[:2,2])/150*5,np.ones(2)*5];target=f.copy();target[:,0]+=.05;x=np.array([.001,-.002,.001,.001,.002,-.001])
    value,grad=objective(x,world,K,.01,g,target,p,.02)
    for k in range(6):
        d=np.zeros(6);d[k]=1e-7
        numeric=(objective(x+d,world,K,.01,g,target,p,.02)[0]-objective(x-d,world,K,.01,g,target,p,.02)[0])/2e-7
        assert np.isclose(numeric,grad[k],rtol=1e-4,atol=1e-6)


def test_no_gradient_no_pose_change_and_world_frame_invariance():
    rng=np.random.default_rng(12);world=rng.uniform([-1,-1,4],[1,1,6],size=(40,3));K=np.array([[150.,0,128],[0,150,72],[0,0,1]])
    grid=np.ones((36,64,4));target=np.ones((40,4));pose=np.eye(4)
    out,info=refine(pose,world,K,0,grid,target)
    assert np.array_equal(out,pose) and not info['accepted']
    grid=rng.normal(size=(36,64,4));pixels=cv2.projectPoints(world,np.zeros(3),np.zeros(3),K,None)[0].reshape(-1,2)
    target=sample_with_gradient(grid,pixels+np.array([.15,-.1]))[0]
    out,a=refine(pose,world,K,0,grid,target)
    G=np.eye(4);G[:3,:3]=cv2.Rodrigues(np.array([.1,.2,-.1]))[0];G[:3,3]=[2,3,1]
    transformed=world@G[:3,:3].T+G[:3,3]
    other,b=refine(np.linalg.inv(G),transformed,K,0,grid,target)
    assert a['accepted']==b['accepted']
    assert np.allclose(other@G,out,atol=1e-7)


def test_local_consensus_equivariance_duplicates_and_far_rejection():
    from feature_extract.tools.vfm.local_pose_consensus import consensus
    poses=np.repeat(np.eye(4)[None],3,axis=0);poses[:,0,3]=[-.1,.05,.2]
    G=np.eye(4);G[:3,:3]=cv2.Rodrigues(np.array([.1,-.2,.05]))[0];G[:3,3]=[1,2,3]
    for method in ['mean','median']:
        out,ok=consensus(np.eye(4),poses,method);other,good=consensus(np.linalg.inv(G),poses@np.linalg.inv(G),method)
        duplicate,dup=consensus(np.eye(4),np.concatenate([poses,poses[:1]]),method)
        assert ok and good and dup and np.allclose(other@G,out,atol=1e-8)
        assert np.allclose(duplicate,out)
    poses[0,0,3]=2
    out,ok=consensus(np.eye(4),poses)
    assert not ok and np.array_equal(out,np.eye(4))


def test_robust_weighted_objective_gradient():
    rng=np.random.default_rng(92);g=rng.normal(size=(36,64,8));K=np.array([[150.,0,128],[0,150,72],[0,0,1.]])
    points=np.array([[-.4,.1,5],[.7,-.3,6],[.3,.4,4]])
    pix=cv2.projectPoints(points,np.zeros(3),np.zeros(3),K,None)[0].reshape(-1,2)
    target=rng.normal(size=(3,8));target/=np.linalg.norm(target,axis=1,keepdims=True);weights=np.array([1.,.5,.3]);x=np.array([.0001,.0003,-.0002,.002,-.001,.002])
    fn=lambda v:objective(v,points,K,0,g,target,pix,.02,weights,True)
    _,gradient=fn(x)
    for k in range(6):
        d=np.zeros(6);d[k]=1e-7
        numerical=(fn(x+d)[0]-fn(x-d)[0])/2e-7
        assert np.isclose(numerical,gradient[k],atol=1e-6,rtol=1e-4)


def test_pose_interpolation_endpoints_and_world_frame():
    from feature_extract.tools.vfm.local_pose_consensus import interpolate_pose
    a=np.eye(4);b=np.eye(4);b[:3,:3]=cv2.Rodrigues(np.array([.03,-.02,.01]))[0];b[:3,3]=[.1,.2,-.1]
    G=np.eye(4);G[:3,:3]=cv2.Rodrigues(np.array([.1,.2,.3]))[0];G[:3,3]=[4,2,1]
    assert np.array_equal(interpolate_pose(a,b,0),a)
    assert np.array_equal(interpolate_pose(a,b,1),b)
    for alpha in [.25,.5,.75]:
        x=interpolate_pose(a,b,alpha);y=interpolate_pose(a@np.linalg.inv(G),b@np.linalg.inv(G),alpha)
        assert np.allclose(y@G,x,atol=1e-9)
        assert np.allclose(x[:3,:3].T@x[:3,:3],np.eye(3))


def test_trust_constraint_jacobian():
    from feature_extract.tools.vfm.direct_anonymous_feature_pose import trust_constraints
    world=np.array([[-.4,.2,5],[.3,-.1,6],[.1,.5,4]])
    K=np.array([[150.,0,128],[0,150,72],[0,0,1.]])
    pix=cv2.projectPoints(world,np.zeros(3),np.zeros(3),K,None)[0].reshape(-1,2)
    x=np.array([.002,-.001,.001,.01,-.02,.005]);v,j=trust_constraints(x,world,K,0,pix)
    for k in range(6):
        d=np.zeros(6);d[k]=1e-7
        numeric=(trust_constraints(x+d,world,K,0,pix)[0]-trust_constraints(x-d,world,K,0,pix)[0])/2e-7
        assert np.allclose(numeric,j[:,k],rtol=1e-5,atol=1e-5)


def test_profiled_bias_gradient_with_robust_nonuniform_weights():
    rng=np.random.default_rng(193);grid=rng.normal(size=(36,64,8));K=np.array([[150.,0,128],[0,150,72],[0,0,1.]])
    points=np.array([[-.4,.1,5],[.7,-.3,6],[.3,.4,4]])
    pix=cv2.projectPoints(points,np.zeros(3),np.zeros(3),K,None)[0].reshape(-1,2)
    target=rng.normal(size=(3,8));weights=np.array([1.,.5,.3]);x=np.array([.0001,.0003,-.0002,.002,-.001,.002])
    for robust in [False,True]:
        fn=lambda v:objective(v,points,K,0,grid,target,pix,.02,weights,robust,profile_bias=True)
        _,gradient=fn(x)
        for k in range(6):
            d=np.zeros(6);d[k]=1e-7
            numerical=(fn(x+d)[0]-fn(x-d)[0])/2e-7
            assert np.isclose(numerical,gradient[k],atol=1e-6,rtol=1e-4)
        shifted=objective(x,points,K,0,grid,target+np.arange(8)*.02,pix,.02,weights,robust,profile_bias=True)
        assert np.allclose(shifted[0],fn(x)[0]) and np.allclose(shifted[1],gradient)


def test_cross_resolution_selection_and_ties():
    from feature_extract.tools.vfm.build_goal_maplet_cross_resolution_selection import choose
    assert choose([[.2,.3],[.1,.2]],'strict_both')==1
    assert choose([[.2,.3],[.21,.1]],'strict_both')==0
    assert choose([[.2,.3],[.21,.1]],'mean_loss')==1
    for rule in ['strict_both','mean_loss']:
        assert choose([[.2,.3],[.2,.3]],rule)==0
        assert choose([[np.nan,.3],[.1,.2]],rule)==0


def test_mapping_depth_visibility_rejects_occlusion_and_invalid_depth():
    from feature_extract.tools.vfm.build_goal_maplet_multiview_feature_map import depth_visible
    depth=np.full((144,256),10.)
    xy=np.array([[10.,10.],[20.,20.],[30.,30.],[-1.,5.],[40.,40.]])
    depth[40,40]=np.nan
    assert depth_visible(xy,np.array([10.05,12.,-1.,10.,10.]),depth).tolist()==[True,False,False,False,False]


def test_cross_resolution_contrast_gradient():
    from feature_extract.tools.vfm.direct_anonymous_feature_pose import contrast_with_gradient
    rng=np.random.default_rng(352);fine=rng.normal(size=(54,96,8));coarse=rng.normal(size=(36,64,8));pixels=np.array([[120.2,80.3],[50.6,30.8]])
    f,g=contrast_with_gradient(fine,coarse,pixels)
    for axis in range(2):
        delta=np.zeros_like(pixels);delta[:,axis]=1e-5
        numerical=(contrast_with_gradient(fine,coarse,pixels+delta)[0]-contrast_with_gradient(fine,coarse,pixels-delta)[0])/2e-5
        assert np.allclose(numerical,g[:,:,axis],atol=1e-7,rtol=1e-5)
