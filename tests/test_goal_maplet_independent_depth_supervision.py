import numpy as np
from feature_extract.tools.vfm.build_goal_maplet_independent_depth_supervision import depth_targets,reprojection_depth_agreement


def test_depth_target_reprojects_at_token_center_with_scale():
    depth=np.full((144,256),10.);pose=np.eye(4);pose[:3,3]=[1,2,3]
    K=np.array([[100.,0,128],[0,100,72],[0,0,1]])
    world,valid,spread=depth_targets(depth,pose,K,0.)
    cam=world@pose[:3,:3].T+pose[:3,3]
    pix=cam[:,:2]/cam[:,2,None]*100+[128,72]
    tok=np.arange(2304)
    np.testing.assert_allclose(pix,np.c_[(tok%64)*4+1.5,(tok//64)*4+1.5])
    assert valid.all() and not spread.any()
    assert reprojection_depth_agreement(world,depth,pose,K,0.).all()


def test_missing_and_discontinuous_depth_does_not_become_label():
    depth=np.full((144,256),10.);depth[:4,:4]=np.nan;depth[4:6,4:8]=20.
    world,valid,_=depth_targets(depth,np.eye(4),np.eye(3),0.)
    assert not valid[0] and not valid[65]
    ref=np.full((144,256),3.)
    assert not reprojection_depth_agreement(world,ref,np.eye(4),np.eye(3),0.).any()
