import numpy as np
from feature_extract.tools.vfm.native_hybrid_region_value import pooled_additions,hybrid_features


def test_pool_preserves_original_identity_and_deduplicates_before_cap():
 rt=[np.array([1,2,3]),np.array([2,4])];rp=[np.array([10,20,30]),np.array([20,40])];rs=[np.array([.9,.8,.7]),np.array([.8,.6])]
 t,p,s=pooled_additions([1],[10],rt,rp,rs,[0,1],limit=2)
 np.testing.assert_array_equal(t,[2,3]);np.testing.assert_array_equal(p,[20,30])
 # A different physical identity for the same image token remains a hypothesis.
 t,p,s=pooled_additions([1],[11],rt,rp,rs,[0,1],limit=2)
 np.testing.assert_array_equal(t,[1,2])


def test_hybrid_novelty_is_relative_to_plane_frontend_and_budget():
 rt=[np.array([1]) for _ in range(16)];rp=[np.array([i]) for i in range(16)];rs=[np.array([.7]) for _ in range(16)];rt[8]=np.array([2]);context=np.ones((16,4));centers=np.zeros((16,3))
 x=hybrid_features(np.array([2]),np.array([8]),np.array([.8]),rt,rp,rs,context,centers)
 assert x.shape==(8,19) and np.isfinite(x).all()
 assert x[0,2]==0 and x[0,16]==0
 y=hybrid_features(np.array([],int),np.array([],int),np.array([]),rt,rp,rs,context,centers)
 assert y[0,2]>0 and y[0,16]>0
