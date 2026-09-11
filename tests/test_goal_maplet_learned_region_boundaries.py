import numpy as np
from feature_extract.vfm.localization_goal_maplet.learned_region_boundaries import optimize_boundaries,boundary_value
from feature_extract.vfm.localization_goal_maplet.metric_region_memory import activate_regions


def test_set_value_rewards_complementary_queries():
 q=np.array([[[0.,.9,0.],[0.,.9,0.]],[[0.,0.,0.],[0.,0.,.8]]]);observed=np.ones((2,2),bool)
 assert boundary_value(q,observed,np.array([1,2]),'set')>boundary_value(q,observed,np.array([1,1]),'set')


def test_boundary_exchange_respects_reference_budget_and_keeps_unobserved():
 q=np.zeros((2,3,3));q[:,0,1]=.2;q[:,0,2]=1.;q[:,1,0]=.1;q[:,1,1]=.2
 observed=np.array([[True,True,False],[True,True,False]])
 cost=np.array([[1,2,3],[1,2,3],[1,2,3]])
 choice,trace=optimize_boundaries(q,observed,cost,6,'set')
 assert choice.tolist()==[2,0,1]
 assert all(t['references']<=6 for t in trace)
 assert np.all(np.diff([.2]+[t['value'] for t in trace])>=0)


def test_learned_radii_change_actual_pnp_members():
 world=np.c_[np.arange(8),np.zeros((8,2))].astype(float);tokens=np.arange(8);scores=np.zeros(8)
 groups,_=activate_regions(world,tokens,scores,np.array([[0.,0,0],[0.,0,0]]),np.array([5.,7.]),2,False)
 assert sorted(len(g) for g in groups)==[6,8]


def test_selection_aware_value_uses_deployed_selector_not_quality_oracle():
 from feature_extract.tools.vfm.fit_goal_maplet_selection_aware_boundaries import selected_quality
 q=np.zeros((1,5,3));q[0,:,1]=[.1,.9,.8,.7,1.]
 support=np.zeros_like(q);support[0,:,1]=[5,4,3,2,1]
 counts=np.zeros_like(q);counts[0,:,1]=[9,8,7,6,99]
 value=selected_quality(np.ones(5,int),q,support,counts,np.zeros_like(q),[0],[0],[0])
 assert value==.1
