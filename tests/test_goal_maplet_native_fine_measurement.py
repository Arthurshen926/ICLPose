import numpy as np
import pytest
from feature_extract.tools.vfm.native_fine_measurement import same_surface_sample,fit_update_scale
from feature_extract.tools.vfm.build_goal_maplet_fine_readout import sample_grid


def test_surface_readout_matches_bilinear_interior_and_excludes_other_surface():
    grid=np.zeros((2,2,2));grid[:,0,0]=1;grid[:,1,1]=1
    pixel=np.array([[127.5,71.5]])
    allsame=np.zeros((144,256),int)
    actual,valid=same_surface_sample(grid,pixel,allsame,np.array([0]))
    np.testing.assert_allclose(actual,sample_grid(grid,pixel));assert valid.all()
    labels=allsame.copy();labels[:,128:]=1
    # Sample center slightly left of the boundary, still interpolating both sides.
    actual,valid=same_surface_sample(grid,np.array([[127.,71.5]]),labels,[0])
    np.testing.assert_allclose(actual,[[1,0]]);assert valid.all()
    actual,valid=same_surface_sample(grid,np.array([[128.,71.5]]),labels,[0]);assert not valid.any()
    actual,valid=same_surface_sample(grid,pixel,np.full((144,256),-1),[0]);assert not valid.any();assert not actual.any()


def test_mapping_calibration_shrinkage_no_extrapolation_and_variance():
    old=np.zeros((4,2));update=np.tile([2.,0.],(4,1));target=np.tile([.5,1.],(4,1));v=np.ones(4)
    fitted=fit_update_scale(old,update,target,v)
    assert fitted['alpha']==pytest.approx(.25)
    assert fitted['variance_scale']==pytest.approx(.5)
    assert fit_update_scale(old,update,-target,v)['alpha']==0
    assert fit_update_scale(old,update,target*10,v)['alpha']==1
    assert fit_update_scale(old,old,target,v)['alpha']==0
    with pytest.raises(ValueError):fit_update_scale(old,update,target,-v)


def test_compound_whitening_matches_dense_covariance_and_singletons():
    from feature_extract.tools.vfm.native_fine_measurement import compound_whiten
    r=np.array([[1.,2.],[3.,-1.],[-2.,4.],[5.,1.],[8.,9.]])
    ids=np.array([0,0,0,1,-1]);rho=.3
    w=compound_whiten(r,ids,rho);cov=(1-rho)*np.eye(3)+rho*np.ones((3,3))
    assert np.sum(w[:3]**2)==pytest.approx(np.sum(r[:3]*(np.linalg.inv(cov)@r[:3])))
    np.testing.assert_array_equal(w[3:],r[3:]);np.testing.assert_array_equal(compound_whiten(r,ids,0),r)
    common=np.ones((3,2));white=compound_whiten(common,np.zeros(3,int),rho)
    np.testing.assert_allclose(white,common/np.sqrt(1+2*rho))
    with pytest.raises(ValueError):compound_whiten(r,ids,1)


def test_compound_sidecar_rejects_duplicate_token_votes(tmp_path):
    import json
    from feature_extract.tools.vfm.native_fine_measurement import load_measurement_groups
    from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256
    path=tmp_path/'corr';path.write_text('frozen')
    corr=dict(names=np.array(['q']),correspondence_offsets=np.array([0,2]),query_tokens=np.array([1,2]),provenance_region_plane_atlas_row=np.array([[0,0,0],[0,0,1]]))
    arr=dict(names=corr['names'],correspondence_offsets=corr['correspondence_offsets'],group_ids=np.array([0,0]))
    meta=dict(artifact_type='goal_maplet_correlated_measurement_groups_v1',query_ground_truth_read=False,rho=.1,arrays_sha256=arrays_sha256(arr),correspondence_file_sha256=file_sha256(path));meta['content_sha256']=canonical_json_sha256(meta)
    side=tmp_path/'groups.npz';np.savez(side,**arr,metadata_json=np.array(json.dumps(meta)))
    load_measurement_groups(side,path,corr)
    corr['query_tokens'][1]=1
    with pytest.raises(ValueError,match='physical evidence'):load_measurement_groups(side,path,corr)
    corr['query_tokens'][1]=2;path.write_text('tamper')
    with pytest.raises(ValueError,match='lineage'):load_measurement_groups(side,path,corr)
