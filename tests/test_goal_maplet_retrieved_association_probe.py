import numpy as np
from feature_extract.tools.vfm.train_goal_maplet_retrieved_association_probe import token_image_weights,fit,auc


def test_duplicate_hypotheses_do_not_multiply_token_training_weight():
    w=token_image_weights(np.array([0,0,0,1]),np.array([2,2,3,4]))
    np.testing.assert_allclose(w,[.125,.125,.25,.5])


def test_fixed_probe_recovers_separation_without_coordinate_model():
    x=np.c_[np.ones(20),np.linspace(-1,1,20)];y=(x[:,1]>0).astype(float)
    beta=fit(x,y,np.ones(20)/20)
    assert auc(y,x@beta)==1
