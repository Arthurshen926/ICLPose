import numpy as np
import pytest

from feature_extract.tools.vfm.calibrate_goal_maplet_joint_reprojection import covariance, metrics, calibrate


def test_zero_correlation_recovers_independent():
    p = np.array([[[2., .3], [.3, 1.]]])
    np.testing.assert_allclose(covariance([.5], p), p+.5*np.eye(2))


@pytest.mark.parametrize("rho", [-.99, -.5, 0., .5, .99])
def test_correlated_covariance_spd(rho):
    rng = np.random.default_rng(18)
    a = rng.normal(size=(100,2,2))
    p = a @ a.transpose(0,2,1)
    assert np.all(np.linalg.eigvalsh(covariance(np.exp(rng.normal(size=100)),p,rho)) > 0)


def test_covariance_rotation_equivariance():
    r = np.array([[.6,-.8],[.8,.6]])
    p = np.array([[[3.,.2],[.2,1.]]])
    np.testing.assert_allclose(covariance([.3],r@p@r.T,.8),r@covariance([.3],p,.8)@r.T)


@pytest.mark.parametrize("rho,scale", [(1.,1.), (0.,0.), (0.,float('nan'))])
def test_invalid_parameters_rejected(rho,scale):
    with pytest.raises(ValueError): covariance([1.],np.eye(2)[None],rho,scale)


def test_synthetic_calibration_generalizes_and_does_not_read_evaluation_for_fit():
    rng = np.random.default_rng(49); n=6000
    q = np.exp(rng.normal(size=n))
    p = np.zeros((n,2,2)); p[:,0,0]=np.exp(rng.normal(size=n)); p[:,1,1]=np.exp(rng.normal(size=n))
    true = covariance(q,p,.7,1.2)
    residual = np.einsum('nij,nj->ni',np.linalg.cholesky(true),rng.normal(size=(n,2)))
    bank = dict(residual=residual,query_variance=q.astype(np.float32),projected_covariance=p,
                calibration_rows=np.arange(3000),evaluation_rows=np.arange(3000,n),
                source_views=np.arange(n),image_error=residual,projected_world_error=residual*.5,
                head_content_sha256=np.asarray('synthetic'))
    report = calibrate(bank)
    assert report['held_mapping_gate_pass']
    fitted=report['models']['correlated']
    assert abs(fitted['rho']-.7)<.06
    bank['residual']=residual.copy();bank['residual'][3000:]*=10
    again=calibrate(bank)['models']['correlated']
    assert fitted['rho']==again['rho'] and fitted['scale']==again['scale']
    bank['source_views'][3000]=0
    with pytest.raises(ValueError,match='leakage'):calibrate(bank)


def test_standard_gaussian_nll():
    assert metrics(np.zeros((1,2)),np.eye(2)[None])['nll']==pytest.approx(np.log(2*np.pi))


def test_runtime_covariance_identity_calibration_parity():
    from feature_extract.tools.vfm.refine_goal_maplet_plane_pose_with_uncertainty import _image_covariance
    pose=np.eye(4); world=np.array([[1.,2.,10.],[2.,1.,12.]])
    centroid=np.repeat(np.eye(3)[None]*.01,2,axis=0)
    k=np.array([[100.,0.,32.],[0.,100.,18.],[0.,0.,1.]])
    args=(pose,world,centroid,np.array([.3,.5]),np.array([.6,1.]),k,.01,False)
    base=_image_covariance(*args)
    np.testing.assert_allclose(_image_covariance(*args,calibration={}),base)
    np.testing.assert_allclose(_image_covariance(*args,calibration={'scale':.7}),.7*base)
