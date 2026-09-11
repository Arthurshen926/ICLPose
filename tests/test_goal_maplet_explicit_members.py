import numpy as np
from feature_extract.tools.vfm.fit_goal_maplet_explicit_members import member_evidence
from feature_extract.vfm.localization_goal_maplet.metric_region_memory import activate_regions

def test_unknown_preserved_and_repeated_rows_do_not_multiply_evidence():
 source=np.r_[np.zeros(20,int),np.arange(4)];proto=np.r_[np.zeros(20,int),np.ones(4,int)];labels=np.zeros(24,int)
 drop,seen,pos,neg=member_evidence(source,proto,labels,3)
 assert not drop[0] and drop[1] and not drop[2] and not seen[2];assert neg[0]==1

def test_point_mask_reaches_actual_seed_group_and_can_keep_noncontiguous_members():
 w=np.c_[np.arange(9),np.zeros((9,2))];t=np.arange(9);allow=np.array([0,1,3,5,7,8]);groups,ids=activate_regions(w,t,np.zeros(9),np.array([[4.,0,0]]),6,1,False,prototype_ids=np.arange(9),allowed_members=[allow]);np.testing.assert_array_equal(groups[0],allow)
