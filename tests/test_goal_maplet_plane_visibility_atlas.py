import numpy as np
from feature_extract.vfm.localization_goal_maplet.plane_visibility_atlas import PlaneVisibilityAtlas,mask_dice,normalized_token_counts
def test_visibility_atlas_roundtrip_and_dice(tmp_path):
 count=np.zeros((1,36,64),np.uint8);count[0,:2,:3]=16
 value=PlaneVisibilityAtlas(np.array([0,1]),np.array([3]),np.array(['a.npz']),np.eye(4)[None],np.zeros((1,3)),count)
 path=tmp_path/'atlas.npz';value.save_npz(path,{'uses_query_or_ground_truth':False});loaded,_=PlaneVisibilityAtlas.load_npz(path)
 assert np.array_equal(loaded.token_pixel_counts,count);assert mask_dice(count[0],count[0])==1
def test_visibility_atlas_rejects_duplicate_observation():
 import pytest
 with pytest.raises(ValueError):PlaneVisibilityAtlas(np.array([0,2]),np.array([1,1]),np.array(['a','b']),np.repeat(np.eye(4)[None],2,0),np.zeros((2,3)),np.zeros((2,36,64),np.uint8)).validated()

def test_normalized_token_counts_supports_shopfacade_grid():
 mask=np.zeros((144,256),np.uint8);mask[:,:128]=1
 count=normalized_token_counts(mask,(68,120))
 assert count.shape==(68,120) and count.dtype==np.uint8
 assert np.all(count[:,:60]==16) and np.all(count[:,60:]==0)

def test_coarse_visibility_and_nearest_template():
 from feature_extract.tools.vfm.evaluate_goal_maplet_plane_visibility_pose import coarse_mask,dice,nearest_templates
 fine=np.zeros((36,64),np.uint8);fine[:4,:4]=16
 assert dice(coarse_mask(fine),coarse_mask(fine))==1
 poses=np.repeat(np.eye(4)[None],2,0);centers=np.array([[20.,0,0],[1.,0,0]])
 atlas=PlaneVisibilityAtlas(np.array([0,2]),np.array([0,1]),np.array(['far','near']),poses,centers,np.stack([fine,fine]))
 indices,reliability=nearest_templates(atlas,0,np.eye(3),np.zeros(3),limit=1);assert indices.tolist()==[1] and 0<reliability<1
