import numpy as np
from feature_extract.vfm.localization_goal_maplet.subpixel_peak import quadratic_peak

def test_recovers_interior_concave_peak():
 xy=np.array([(x,y) for y in [-4/3,0.,4/3] for x in [-4/3,0.,4/3]])
 s=-(xy[:,0]-.3)**2-2*(xy[:,1]+.2)**2
 p,ok,_=quadratic_peak(s[None]);assert ok[0];np.testing.assert_allclose(p[0],[.3,-.2],atol=1e-12)

def test_flat_surface_does_not_drift():
 p,ok,_=quadratic_peak(np.ones((2,9)));assert not ok.any();assert np.all(p==0)

def test_convex_and_out_of_bounds_fall_back():
 xy=np.array([(x,y) for y in [-4/3,0.,4/3] for x in [-4/3,0.,4/3]])
 s=np.array([np.sum(xy**2,1),-(xy[:,0]-5)**2-xy[:,1]**2]);p,ok,d=quadratic_peak(s);assert not ok.any();np.testing.assert_array_equal(p,d)
