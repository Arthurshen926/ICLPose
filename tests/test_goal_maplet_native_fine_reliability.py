import numpy as np
import pytest
from feature_extract.tools.vfm.native_fine_reliability import reliability_features,fit,predict


def test_reliability_features_ignore_invalid_candidates_and_preserve_measurement_units():
 sim=np.full((2,9),-np.inf);sim[:,4]=.5;sim[:,0]=.6;sim[:,1]=.55
 f=reliability_features(sim,np.array([[1.,0.],[0.,0.]]),np.array([1.,2.]))
 np.testing.assert_allclose(f[:,0],.5);np.testing.assert_allclose(f[:,1],.1);np.testing.assert_allclose(f[:,2],.05)
 np.testing.assert_allclose(f[:,3],[1,0]);np.testing.assert_allclose(f[:,4],[0,np.log(2)]);np.testing.assert_allclose(f[:,5],1/3)
 sim[0,4]=-np.inf
 with pytest.raises(ValueError):reliability_features(sim,np.zeros((2,2)),np.ones(2))


def test_conditional_fit_learns_reliability_without_leaving_token_segment():
 n=80;x=np.zeros((n,6));x[:,0]=np.linspace(-1,1,n);old=np.zeros((n,2));update=np.tile([1.,0.],(n,1));target=np.c_[.5+.3*x[:,0],np.full(n,.3)];var=np.ones(n)
 model=fit(x,old,update,target,var,np.ones(n),.5,1.)
 alpha,v=predict(model,x,var)
 assert np.all((alpha>0)&(alpha<1));assert np.isfinite(v).all() and np.all(v>0)
 assert alpha[-1]>alpha[0]+.3
 assert np.mean((old+alpha[:,None]*(update-old)-target)**2)<np.mean((old+.5*(update-old)-target)**2)
 with pytest.raises(ValueError):predict(model,x,-var)
