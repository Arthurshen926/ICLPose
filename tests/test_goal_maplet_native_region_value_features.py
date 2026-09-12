import numpy as np
from feature_extract.tools.vfm.native_region_value_features import addition_features
from feature_extract.tools.vfm.train_goal_maplet_native_region_marginal_value import fit_ridge,predict


def test_addition_features_condition_on_existing_set_and_do_not_multiply_token_coverage():
 args=[np.array([1,2]),np.array([.5,.6]),np.array([2,2,3]),np.array([.7,.7,.8]),np.ones(4)*.8,np.ones((8,4))*.6,np.array([6.,0,0]),np.zeros((8,3))]
 x=addition_features(*args)
 assert x.shape==(13,) and np.isfinite(x).all()
 np.testing.assert_allclose(x[:4],[3/1024,2/2304,1/2304,2/2304])
 np.testing.assert_allclose(x[9:11],[.8,1.])
 args[0]=np.array([1,2,3]);y=addition_features(*args)
 assert y[2]==0 and y[3]>x[3]
 args[2]=np.array([],int);args[3]=np.array([]);z=addition_features(*args)
 assert np.isfinite(z).all() and z[0]==z[4]==z[5]==0


def test_marginal_model_preserves_negative_value_and_uses_frozen_scaling():
 x=np.zeros((100,13));x[:,0]=np.linspace(-1,1,100);y=x[:,0]*.2
 m=fit_ridge(x,y);p=predict(m,x)
 assert p[0]<0<p[-1]
 np.testing.assert_allclose(predict(m,x[:1]),p[:1])
 assert np.mean((p-y)**2)<np.mean(y**2)*.05
