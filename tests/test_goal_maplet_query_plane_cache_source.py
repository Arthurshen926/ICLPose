import numpy as np
import pytest
from feature_extract.tools.vfm.build_goal_maplet_query_plane_regions import _work, _sha
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions, SparseOcclusionCarrierConfig


@pytest.mark.parametrize('mismatch',['source','pose','base'])
def test_reused_cache_rejects_wrong_source(tmp_path,mismatch):
    source=tmp_path/'source.npz';np.savez(source,x=np.array([1]))
    output=tmp_path/'regions.npz'
    value=QueryPlaneRegions(np.zeros((2,2),np.int32),np.array([[0.,0.,1.]]),np.array([2.]),np.array([4]),np.array([0.]),np.array([0.]))
    meta={'source_file_sha256':_sha(source),'uses_pose_or_ground_truth':False,'base_query_plane_file_sha256':None,'sparse_occlusion_carrier':False}
    if mismatch=='source':meta['source_file_sha256']='wrong'
    if mismatch=='pose':meta['uses_pose_or_ground_truth']=True
    if mismatch=='base':meta['base_query_plane_file_sha256']='wrong'
    value.save_npz(output,meta)
    with pytest.raises(ValueError,match='existing query-plane cache differs'):
        _work((str(source),str(output),None,False,SparseOcclusionCarrierConfig().payload()))


def test_reused_cache_accepts_same_source(tmp_path):
    source=tmp_path/'source.npz';np.savez(source,x=np.array([1]))
    output=tmp_path/'regions.npz'
    value=QueryPlaneRegions(np.zeros((2,2),np.int32),np.array([[0.,0.,1.]]),np.array([2.]),np.array([4]),np.array([0.]),np.array([0.]))
    value.save_npz(output,{'source_file_sha256':_sha(source),'uses_pose_or_ground_truth':False,'base_query_plane_file_sha256':None,'sparse_occlusion_carrier':False})
    result=_work((str(source),str(output),None,False,SparseOcclusionCarrierConfig().payload()))
    assert result[1:3]==(1,4)
