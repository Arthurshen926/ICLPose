import numpy as np
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import extract_query_plane_regions
def test_query_plane_ransac_recovers_finite_mask():
 y,x=np.mgrid[:20,:30];p=np.stack((x*.02,y*.02,np.full_like(x,4.0)),axis=-1);n=np.zeros_like(p);n[...,2]=1;v=np.ones((20,30),bool);o=extract_query_plane_regions(p,n,v,minimum_pixels=50)
 assert len(o.normals_camera)==1 and np.mean(o.labels>=0)==1 and abs(o.offsets_camera[0]-4)<1e-9

def test_query_plane_cache_roundtrip_and_tamper(tmp_path):
 from dataclasses import replace
 from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions
 labels=np.full((4,4),-1,np.int32);labels[:2]=0
 value=QueryPlaneRegions(labels,np.array([[0.,0.,1.]]),np.array([2.]),np.array([8]),np.array([0.]),np.array([0.]))
 path=tmp_path/'q.npz';value.save_npz(path,{'uses_pose_or_ground_truth':False});loaded,_=QueryPlaneRegions.load_npz(path);assert np.array_equal(loaded.labels,labels)
 import pytest
 with pytest.raises(ValueError):replace(value,pixel_counts=np.array([7])).validated()
