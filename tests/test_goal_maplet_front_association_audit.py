import numpy as np
import pytest
import cv2
import json
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _solve,_unique_token_rows


def test_unique_token_association_residual_then_stable_row_tie():
    rows=np.array([3,2,1,0]);tokens=np.array([0,0,1,1])
    np.testing.assert_array_equal(_unique_token_rows(rows,tokens,[1.,1.,2.,1.]),[0,2])
    with pytest.raises(ValueError):_unique_token_rows(rows,tokens,[1.,2.,3.,np.nan])


def test_lm_receives_one_row_per_token(monkeypatch):
    world=np.c_[np.repeat(np.arange(6),2),np.zeros(12),np.ones(12)*10].astype(float)
    tokens=np.repeat(np.arange(6),2);K=np.eye(3)
    pixels=world[:,:2]/world[:,2:];pixels[1::2]+=.1
    monkeypatch.setattr(cv2,'solvePnPRansac',lambda *a,**kw:(True,np.zeros((3,1)),np.zeros((3,1)),np.arange(12)[:,None]))
    counts=[]
    def refine(points,*a,**kw):
        counts.append(len(points));return np.zeros((3,1)),np.zeros((3,1))
    monkeypatch.setattr(cv2,'solvePnPRefineLM',refine)
    _solve(world,tokens,K,0.,np.arange(12),pixels)
    _solve(world,tokens,K,0.,np.arange(12),pixels,unique_token_lm=False)
    assert counts==[6,12]


def test_duplicate_hypotheses_cannot_fake_six_measurements():
    assert _solve(np.ones((12,3)),np.zeros(12,int),np.eye(3),0.,np.arange(12)) is None


def test_lm_rejects_inliers_behind_camera(monkeypatch):
    world=np.ones((6,3));world[:,2]=-1
    monkeypatch.setattr(cv2,'solvePnPRansac',lambda *a,**kw:(True,np.zeros(3),np.zeros(3),np.arange(6)[:,None]))
    assert _solve(world,np.arange(6),np.eye(3),0.,np.arange(6)) is None


def test_within_plane_ranking_preserves_cross_plane_ambiguity():
    from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _select_match_rows
    tokens=np.array([0,0,0,1]);provenance=np.array([[0,1,0],[0,1,1],[0,2,2],[0,1,3]])
    scores=np.array([.8,.9,.7,.6])
    np.testing.assert_array_equal(_select_match_rows(tokens,provenance,scores),[1,3])
    np.testing.assert_array_equal(_select_match_rows(tokens,provenance,scores,True),[1,2,3])


def test_zero_residual_is_not_treated_as_missing():
    from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _choice_key
    perfect=dict(inlier_count=10,reprojection_median_px=0.)
    worse=dict(inlier_count=10,reprojection_median_px=.01)
    missing=dict(inlier_count=10,reprojection_median_px=None)
    assert _choice_key(perfect,'raw_inliers')>_choice_key(worse,'raw_inliers')>_choice_key(missing,'raw_inliers')


def test_seed_groups_use_measurements_not_duplicate_hypotheses():
    from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _top_groups
    values=np.r_[np.zeros(20,int),np.ones(6,int)]
    tokens=np.r_[np.zeros(20,int),np.arange(6)]
    chosen=_top_groups(values,1,tokens)
    assert len(chosen)==1 and np.all(values[chosen[0]]==1)


@pytest.mark.parametrize('corruption',['offset_end','offset_start','token','nan_world','nan_radial'])
def test_loader_rejects_invalid_payload_even_with_valid_hash(tmp_path,corruption):
    from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load,arrays_sha256
    arrays=dict(names=np.array(['q']),correspondence_offsets=np.array([0,6]),world_points=np.ones((6,3)),
        query_tokens=np.arange(6),provenance_region_plane_atlas_row=np.zeros((6,3),int),
        camera_matrices=np.eye(3)[None],radial_k1=np.zeros(1))
    if corruption=='offset_end':arrays['correspondence_offsets'][-1]=5
    if corruption=='offset_start':arrays['correspondence_offsets'][0]=1
    if corruption=='token':arrays['query_tokens'][0]=2304
    if corruption=='nan_world':arrays['world_points'][0,0]=np.nan
    if corruption=='nan_radial':arrays['radial_k1'][0]=np.nan
    metadata=dict(artifact_type='goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v1',
                  pose_or_ground_truth_opened=False,arrays_sha256=arrays_sha256(arrays))
    path=tmp_path/'bad.npz';np.savez_compressed(path,**arrays,metadata_json=np.asarray(json.dumps(metadata)))
    with pytest.raises(ValueError):_load(path)
