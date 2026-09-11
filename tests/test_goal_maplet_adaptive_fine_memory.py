import numpy as np
import pytest
from feature_extract.tools.vfm.build_goal_maplet_fine_readout import sample_grid
from feature_extract.tools.vfm.learn_goal_maplet_memory_units import capacity_mask,policy_design
from feature_extract.tools.vfm.build_goal_maplet_roi_readout import local_pixels
from feature_extract.tools.vfm.fit_goal_maplet_fine_readout_scores import selection
from feature_extract.tools.vfm.rerank_goal_maplet_memory_depth_relations import depth_relations


def test_grid_readout_uses_pixel_centers_at_both_resolutions():
    for h,w in [(36,64),(54,96)]:
        grid=np.zeros((h,w,3),np.float32);grid[...,0]=1
        grid[h//2,w//2]=[0,1,0]
        pixel=np.array([[(w//2+.5)*256/w-.5,(h//2+.5)*144/h-.5]])
        assert np.allclose(sample_grid(grid,pixel),[[0,1,0]],atol=1e-6)


def test_grid_readout_never_wraps_image_edges():
    grid=np.zeros((2,2,2));grid[:,0,0]=1;grid[:,1,1]=1
    assert np.allclose(sample_grid(grid,np.array([[-100,50],[500,50]])),np.eye(2))


def test_capacity_is_hard_and_ties_are_deterministic():
    scores=np.array([1.,3.,3.,2.,0.]);before=scores.copy()
    assert np.array_equal(np.flatnonzero(capacity_mask(scores,.4)),[1,2])
    assert capacity_mask(scores,1).all()
    assert np.array_equal(before,scores)
    with pytest.raises(ValueError):capacity_mask([np.nan],.5)


def test_policy_features_do_not_change_geometry_or_depend_on_query():
    attributes=np.arange(48,dtype=float).reshape(2,3,8)/48
    copy=attributes.copy();out=policy_design(attributes)
    assert out.shape==(2,3,21)
    assert np.array_equal(attributes,copy)
    assert np.array_equal(out[1],policy_design(attributes[1]))


def test_roi_coordinates_represent_same_rgb_locations():
    for q in range(4):
        center=np.array([q%2*128+63.5,q//2*72+35.5])
        assert np.allclose(local_pixels(center,q),[127.5,71.5])


def test_refinement_activation_budget_counts_tokens_not_overlapping_modes():
    c={'source_image':np.zeros(20,int),'homography_keep':np.ones(20,bool),
       'prototype_rows':np.tile([0,1],10),'query_token':np.repeat(np.arange(10),2)}
    mask=selection(c,np.repeat(np.arange(10),2),np.ones(2,bool),3)
    assert np.array_equal(np.unique(c['query_token'][mask]),[7,8,9])
    c['homography_keep'][-2:]=False
    assert not selection(c,np.ones(20),np.ones(2,bool),3)[-2:].any()


def test_depth_relations_ignore_common_scale_and_duplicate_support():
    t=np.array([0,2,4,6]);q=np.array([1.,2.,2.5,4.]);z=q*7
    a=depth_relations(t,q,z,np.ones(4,bool))
    b=depth_relations(np.repeat(t,2),np.repeat(q*3,2),np.repeat(z*5,2),np.ones(8,bool))
    assert np.allclose(a,b)
    assert a[0]==pytest.approx(1.) and a[1]==pytest.approx(1.)
    assert depth_relations(t,q,-z,np.ones(4,bool))==(0.,0.,0)
