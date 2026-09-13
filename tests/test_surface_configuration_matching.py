import numpy as np
import cv2
from feature_extract.tools.vfm.surface_configuration_matching import configuration_match

def problem():
 rng=np.random.default_rng(360);q=rng.normal(size=(20,3));qn=rng.normal(size=(20,3));qn/=np.linalg.norm(qn,axis=1,keepdims=True);w=np.stack([q*2+3,rng.normal(size=q.shape)*4],axis=1);n=np.stack([qn,qn],axis=1);u=np.tile([.8,.79],(20,1));xy=rng.normal(size=(20,2));return u,w,n,q,qn,xy

def test_relations_preserve_coordinate_frames_scale_and_mass():
 u,w,n,q,qn,xy=problem();a=configuration_match(u,w,n,q,qn,xy);R=cv2.Rodrigues(np.array([.3,-.1,.2]))[0];S=cv2.Rodrigues(np.array([-.2,.4,.1]))[0]
 b=configuration_match(u,w@R.T+7,n@R.T,(q@S.T-4)*5,qn@S.T,xy)
 assert np.allclose(a[2],b[2],atol=1e-10);assert np.allclose(a[2].sum(1),1)
 assert np.mean(a[0]==0)>.9

def test_alternative_permutation_and_independent_control():
 u,w,n,q,qn,xy=problem();a=configuration_match(u,w,n,q,qn,xy);b=configuration_match(u[:,::-1],w[:,::-1],n[:,::-1],q,qn,xy)
 assert np.allclose(a[2][:,:2],b[2][:,:2][:,::-1]);assert np.allclose(a[2][:,2],b[2][:,2])
 c=configuration_match(u,w,n,q,qn,xy,'independent');d=configuration_match(u,w,n,q*2,qn,xy,'independent');assert np.array_equal(c[0],d[0])

def test_unknown_normals_are_not_negative_observations():
 u,w,n,q,qn,xy=problem();n[:]=0
 a=configuration_match(u,w,n,q,qn,xy)
 b=configuration_match(u,w,n,q,-qn,xy)
 assert np.isfinite(a[2]).all();assert np.array_equal(a[2],b[2])

def test_absolute_unknown_and_column_capacity():
 u,w,n,q,qn,xy=problem();low=configuration_match(u-1,w,n,q,qn,xy,absolute_null=.6,exclusive=True)
 assert not low[1].any()
 # Identical proposed anchors cannot acquire more than one total soft unit.
 w[:]=w[0];n[:]=n[0]
 result=configuration_match(u,w,n,q,qn,xy,absolute_null=.6,exclusive=True)
 assert np.all(result[2][:,:2].sum(0)<=1+1e-10);assert np.allclose(result[2].sum(1),1)

def test_visibility_supervision_preserves_unknown_and_counts_distinct_cells():
 from feature_extract.tools.vfm.prepare_goal_maplet_memory_confusion_training import visibility_label
 inside=np.ones(8,bool)
 assert visibility_label(inside,np.zeros(8,bool),np.arange(8))=='unknown'
 assert visibility_label(inside,inside,np.zeros(8,int))=='unknown'
 assert visibility_label(inside,inside,np.arange(8))=='positive_visible'
 assert visibility_label(np.zeros(8,bool),np.zeros(8,bool),np.arange(8))=='negative_outside_patch'
 assert visibility_label(np.zeros(0,bool),np.zeros(0,bool),np.arange(0))=='unknown'
