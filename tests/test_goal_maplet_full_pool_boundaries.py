import numpy as np
from feature_extract.tools.vfm.fit_goal_maplet_full_pool_boundaries import optimize
from feature_extract.tools.vfm.fit_goal_maplet_selection_aware_boundaries import selected_quality

def test_optimizer_uses_final_selected_utility_and_budget():
 q=np.zeros((2,5,3));q[:,0,1]=.1;q[:,0,2]=.8
 s=np.ones_like(q);s[:,0,:]=10
 counts=np.ones_like(q);counts[:,0,:]=20
 cost=np.tile([1,2,3],(5,1))
 choice,report=optimize(q,s,counts,np.zeros_like(q),[0,0],[0,0],[0,0],cost)
 assert choice[0]==2
 assert cost[np.arange(5),choice].sum()<=cost[:,1].sum()
 assert report['final_value']==.8

def test_geometry_tie_uses_stable_global_first_and_ignores_gt_quality():
 q=np.ones((2,5,3));support=np.ones_like(q);counts=np.ones_like(q)*10
 assert selected_quality(np.ones(5,int),q,support,counts,np.zeros_like(q),[10,10],[0,0],[.2,.4])==np.mean([.2,.4])
