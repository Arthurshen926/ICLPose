import numpy as np
import pytest
from feature_extract.tools.vfm.train_goal_maplet_mapping_canonical_subtoken_head import _tail_constrained_coordinate_shrinkage, _closed_form_coordinate_shrinkage

def test_perfect_prediction_keeps_full_correction():
    x=np.random.default_rng(2).normal(size=(100,2))
    assert _tail_constrained_coordinate_shrinkage(x,x)==1.

def test_tail_constraint_controls_outliers():
    target=np.tile([1.,0.],(100,1));target[-9:]*=10;prediction=target.copy();prediction[:20]*=3
    alpha=_tail_constrained_coordinate_shrinkage(prediction,target)
    old=_closed_form_coordinate_shrinkage(prediction,target)
    assert np.quantile(np.linalg.norm(old*prediction-target,axis=1),.9)>1.
    assert np.quantile(np.linalg.norm(alpha*prediction-target,axis=1),.9)<=1.
    assert 0<alpha<old
    assert np.mean((alpha*prediction-target)**2)<np.mean(target**2)

def test_invalid_calibration_is_rejected():
    with pytest.raises(ValueError):_tail_constrained_coordinate_shrinkage([[np.nan,0]],[[0,0]])
