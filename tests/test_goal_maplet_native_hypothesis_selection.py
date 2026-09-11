import numpy as np
import pytest
from feature_extract.tools.vfm.filter_goal_maplet_native_region_augmentation import select_added_rows,filter_inventory


def test_equal_token_coverage_and_stable_ties():
    tokens=np.array([4,2,4,5,2]);scores=np.array([.8,.5,.9,.1,.5])
    np.testing.assert_array_equal(select_added_rows(tokens,scores,'cosine',1),[1,2,3])
    for policy in ['cosine','probability','random']:
        rows=select_added_rows(tokens,scores,policy,20)
        np.testing.assert_array_equal(np.sort(tokens[rows]),[2,4,5])
        np.testing.assert_array_equal(rows,select_added_rows(tokens,scores,policy,20))
    assert len(select_added_rows(np.array([],int),np.array([]),'random',1))==0
    with pytest.raises(ValueError):select_added_rows(tokens,np.full(5,np.nan),'cosine',0)


def test_reference_prefix_survives_and_only_added_hypotheses_compete():
    ref=dict(names=np.array(['q','empty']),camera_matrices=np.tile(np.eye(3),(2,1,1)),radial_k1=np.zeros(2),
             correspondence_offsets=np.array([0,2,2]),query_tokens=np.array([3,3]),radio_match_score=np.array([.9,.8]),
             correspondence_match_probability=np.array([.8,.7]),world_points=np.array([[1.,0,0],[2.,0,0]]))
    cand={k:v.copy() for k,v in ref.items()};cand.update(correspondence_offsets=np.array([0,5,5]),query_tokens=np.array([3,3,3,3,4]),radio_match_score=np.array([.9,.8,.7,.6,.5]),correspondence_match_probability=np.array([.8,.7,.3,.9,.2]),world_points=np.arange(15).reshape(5,3).astype(float));cand['world_points'][:2]=ref['world_points']
    result,audit=filter_inventory(cand,ref,'probability')
    np.testing.assert_array_equal(result['query_tokens'],[3,3,3,4])
    np.testing.assert_array_equal(result['world_points'],cand['world_points'][[0,1,3,4]])
    np.testing.assert_array_equal(result['correspondence_offsets'],[0,4,4])
    assert audit[0]['added_after']==2
    cand['world_points'][0,0]+=1
    with pytest.raises(ValueError,match='prefix'):filter_inventory(cand,ref,'cosine')


def test_row_sampler_control_can_repeat_token_but_token_sampler_cannot():
    from feature_extract.tools.vfm.token_hypothesis_ransac import guided_sample
    groups=[np.arange(20),np.array([20]),np.array([21]),np.array([22])]
    centers=np.zeros((4,2));planes=np.zeros(23,int);scores=np.zeros(23)
    tokens=np.r_[np.zeros(20,int),[1,2,3]]
    row_repeated=False
    for seed in range(10):
        row=guided_sample(np.random.default_rng(seed),groups,centers,planes,scores,'row_uniform')
        token=guided_sample(np.random.default_rng(seed),groups,centers,planes,scores,'uniform')
        assert len(np.unique(token))==4 and len(np.unique(tokens[token]))==4
        assert len(np.unique(row))==4
        row_repeated |= len(np.unique(tokens[row]))<4
    assert row_repeated
