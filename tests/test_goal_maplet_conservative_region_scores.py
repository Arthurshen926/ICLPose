import numpy as np
from feature_extract.tools.vfm.prepare_goal_maplet_conservative_region_scores import conservative_scores

def test_old_scores_preserved_and_new_floor_is_query_and_plane_specific():
 old=np.array([3.,-2.,1.,7.]);out=conservative_scores(np.array([0,0,0,1]),np.array([2,2,3,2]),old,np.array([0,0,1,2]),np.array([2,9,2,3]))
 np.testing.assert_array_equal(out[:4],old);np.testing.assert_array_equal(out[4:],[-2,-2,7,-2])
