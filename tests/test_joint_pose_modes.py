import numpy as np
from feature_extract.tools.vfm.retain_joint_pose_modes import retain_modes


def pose(x):
    p=np.eye(4);p[0,3]=x;return p


def test_diverse_retention_keeps_a_lower_ranked_separated_explanation():
    candidates=[((20,-1.),pose(0)),((19,-1.),pose(.1)),((18,-1.),pose(2))]
    assert [p[0,3] for p in retain_modes(candidates,limit=2)]==[0,2]
    assert [p[0,3] for p in retain_modes(candidates,limit=2,diverse=False)]==[0,.1]


def test_duplicate_models_do_not_consume_retention_budget():
    candidates=[((20,-1.),pose(0)),((20,-1.),pose(0)),((18,-1.),pose(2))]
    assert len(retain_modes(candidates,limit=2,diverse=False))==2


def test_polishing_rejects_a_worse_reprojection_update(monkeypatch):
    import cv2
    from feature_extract.tools.vfm.retain_joint_pose_modes import polish_pose
    rng=np.random.default_rng(3)
    world=rng.uniform([-1.,-1.,4.],[1.,1.,8.],(20,3))
    K=np.array([[140.,0.,128.],[0.,145.,72.],[0.,0.,1.]])
    pixels=cv2.projectPoints(world,np.zeros(3),np.zeros(3),K,None)[0].reshape(-1,2)
    monkeypatch.setattr(cv2,'solvePnPRefineLM',lambda *a,**kw:(np.zeros(3),np.array([100.,0.,0.])))
    np.testing.assert_array_equal(polish_pose(np.eye(4),world,np.arange(20),pixels,K,0.),np.eye(4))
