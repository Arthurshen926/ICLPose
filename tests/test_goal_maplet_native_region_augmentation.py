import numpy as np
import pytest
from feature_extract.tools.vfm.build_goal_maplet_native_region_augmentation import append_query_rows
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _top_groups


def test_ragged_augmentation_preserves_prefix_and_empty_query():
    original=dict(names=np.array(['a','b','c']),camera_matrices=np.tile(np.eye(3),(3,1,1)),
                  radial_k1=np.zeros(3),correspondence_offsets=np.array([0,2,2,3]),
                  query_tokens=np.array([3,8,2]),world_points=np.arange(9).reshape(3,3).astype(float))
    added=[dict(query_tokens=np.array([9]),world_points=np.ones((1,3))),
           dict(query_tokens=np.array([],int),world_points=np.empty((0,3))),
           dict(query_tokens=np.array([11,12]),world_points=np.ones((2,3))*2)]
    result=append_query_rows(original,added)
    np.testing.assert_array_equal(result['correspondence_offsets'],[0,3,3,6])
    np.testing.assert_array_equal(result['query_tokens'],[3,8,9,2,11,12])
    for i in range(3):
        lo,hi=original['correspondence_offsets'][i:i+2];start=result['correspondence_offsets'][i]
        for k in ['query_tokens','world_points']:
            np.testing.assert_array_equal(original[k][lo:hi],result[k][start:start+hi-lo])
    with pytest.raises(ValueError):append_query_rows(original,added[:2])


def test_reference_caps_count_unique_token_groups_including_zero():
    values=np.r_[np.zeros(8,int),np.ones(10,int)]
    tokens=np.r_[np.arange(8),np.zeros(10,int)]
    cap=len(_top_groups(values,16,tokens))
    assert cap==1
    # Added real support creates eligible groups but cannot expand reference cap.
    newvalues=np.r_[values,np.full(8,2),np.full(8,3)]
    newtokens=np.r_[tokens,np.arange(8),np.arange(8)]
    assert len(_top_groups(newvalues,cap,newtokens))==cap
    assert _top_groups(newvalues,0,newtokens)==[]


def test_metric_boundaries_preserve_identity_and_forbid_leakage(tmp_path):
    from feature_extract.tools.vfm.build_goal_maplet_native_region_augmentation import readout_radii
    centers=np.array([[0.,0,0],[10.,0,0]])
    path=tmp_path/'boundary.npz'
    np.testing.assert_array_equal(readout_radii(None,centers),[6.,6.])
    np.savez(path,centers=centers,radii=[3.,9.],training_images=['mapping'])
    np.testing.assert_array_equal(readout_radii(path,centers,['query']),[3.,9.])
    with pytest.raises(ValueError,match='overlap'):readout_radii(path,centers,['mapping'])
    with pytest.raises(ValueError,match='identity'):readout_radii(path,centers[::-1])
    np.savez(path,centers=centers,radii=[3.,9.],member_policy='explicit_subset')
    with pytest.raises(ValueError,match='explicit'):readout_radii(path,centers)
    np.savez(path,centers=centers,radii=[3.,float('nan')])
    with pytest.raises(ValueError,match='invalid'):readout_radii(path,centers)


def test_context_coverage_keeps_first_anchor_and_avoids_redundant_region():
    from feature_extract.tools.vfm.build_goal_maplet_native_region_augmentation import select_context_modes
    scores=np.array([[.99,.98,.2,.97],[.1,.1,.9,.1]])
    ids=np.array([0,1,2,0])
    assert select_context_modes(scores,ids,2,'max')==[0,1]
    assert select_context_modes(scores,ids,2,'coverage')==[0,2]
    # Distinct modes of an already selected physical region cannot add a vote.
    assert select_context_modes(scores,ids,3,'coverage')==[0,2,1]
    with pytest.raises(ValueError):select_context_modes(scores,ids,4,'coverage')
