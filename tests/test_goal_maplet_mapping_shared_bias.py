import numpy as np
import pytest
from feature_extract.tools.vfm.audit_goal_maplet_mapping_shared_bias import grouped_bias


def test_signed_bias_does_not_disappear_in_global_average():
    result=grouped_bias([[1,0],[3,0],[-1,0],[-3,0]],[0,0,1,1])
    assert result[0]['signed_mean_px']==[2,0]
    assert result[1]['signed_mean_px']==[-2,0]
    assert result[0]['bias_norm_px']==2
    assert result[0]['centered_mse_px2']==1
    assert result[0]['mse_px2']==5


def test_nonfinite_bias_is_rejected():
    with pytest.raises(ValueError):grouped_bias([[np.nan,0]],[0])
