import numpy as np
import pytest
from feature_extract.tools.vfm.train_goal_maplet_mapping_canonical_subtoken_head import (
    _source_view_mode_pool, _diverse_mode_indices, _fit_source_view_mode_dataset)
from feature_extract.tools.vfm.build_goal_maplet_plane_uv_radio_atlas import _fuse_plane_texel_prototypes


def test_source_mode_pool_exact_atlas_features_and_geometry():
    rng=np.random.default_rng(51)
    world=rng.uniform(.05,.4,(12,3));feat=rng.normal(size=(12,5)).astype(np.float32)
    feat/=np.linalg.norm(feat,axis=1,keepdims=True)
    views=np.repeat(np.arange(4),3)
    keys,f,p=_source_view_mode_pool(np.zeros(12,int),views,world,feat)
    selected=_diverse_mode_indices(f,3)
    atlas=_fuse_plane_texel_prototypes(world[:,:2],feat,views,cell_size_m=.5,
                                     minimum_views=2,maximum_prototypes=3,surface_height_m=world[:,2])
    np.testing.assert_allclose(atlas[0],p[selected,:2],atol=1e-8)
    np.testing.assert_allclose(atlas[1],f[selected],atol=1e-7)
    np.testing.assert_allclose(atlas[-2],p[selected,2],atol=1e-8)


def test_center_modes_exclude_whole_source_and_validation_route(monkeypatch):
    import feature_extract.tools.vfm.train_goal_maplet_mapping_canonical_subtoken_head as module
    monkeypatch.setattr(module,'_project_world_to_pixel',lambda points,*a:(np.tile([1.5,1.5],(len(points),1)),np.ones(len(points))))
    world=np.c_[np.arange(6)*.01,np.zeros(6),np.ones(6)]
    source=np.array([0,0,1,2,3,3]);routes=np.array(['fit']*4+['val']*2)
    features=np.eye(6,dtype=np.float32)
    data=_fit_source_view_mode_dataset(np.arange(6),np.zeros(6,int),np.zeros(6,int),np.arange(6),
        source,routes,{'fit'},'val',world,features,np.zeros(6,int),np.tile(np.eye(4),(6,1,1)),
        np.tile(np.eye(3),(6,1,1)),np.zeros(6),4)
    for q,feature in zip(data['fit_query_rows'],data['fit_map_features']):
        assert np.all(feature[source==source[q]]==0)
    assert np.all(data['validation_map_features'][:,4:]==0)
    valworld=data['validation_map_world'][:,0]
    assert set(np.round(valworld,4))=={.005,.02,.03}


def test_two_independent_views_required_after_exclusion(monkeypatch):
    import feature_extract.tools.vfm.train_goal_maplet_mapping_canonical_subtoken_head as module
    monkeypatch.setattr(module,'_project_world_to_pixel',lambda points,*a:(np.tile([1.5,1.5],(len(points),1)),np.ones(len(points))))
    source=np.array([0,0,1,2]);routes=np.array(['fit']*3+['val'])
    data=_fit_source_view_mode_dataset(np.arange(4),np.zeros(4,int),np.zeros(4,int),np.arange(4),source,
        routes,{'fit'},'val',np.zeros((4,3)),np.eye(4,dtype=np.float32),np.zeros(4,int),
        np.tile(np.eye(4),(4,1,1)),np.tile(np.eye(3),(4,1,1)),np.zeros(4),4)
    assert len(data['fit_query_rows'])==0
    assert len(data['validation_query_rows'])==2


def test_nulls_exclude_source_cell_and_near_geometry():
    from feature_extract.tools.vfm.train_goal_maplet_mapping_canonical_subtoken_head import _isolated_null_indices
    world=np.array([[0.,0.,0.],[0.,0.,0.],[2.,0.,0.],[2.,0.,0.]])
    ids=np.array([0,0,1,1]);source=np.array([0,1,0,1]);mapsource=np.array([1,0,1,0])
    rows,valid=_isolated_null_indices(np.arange(4),world,mapsource,np.arange(4),source,ids,np.zeros(4,int),world)
    assert valid.all()
    assert np.all(ids[rows]!=ids)
    assert np.all(mapsource[rows]!=source)
    rows,valid=_isolated_null_indices(np.arange(4),world,mapsource,np.arange(4),source,np.zeros(4,int),np.zeros(4,int),world)
    assert not valid.any()


def test_runtime_rejects_misaligned_or_unfixed_mode_contract():
    from feature_extract.tools.vfm.build_goal_maplet_plane_uv_radio_correspondences import _validate_source_mode_center_contract
    head=dict(prototype_policy='source_view_modes',source_view_training_contract=True,source_isolated_nulls=True,
              center_mode_token_pooling='all_tokens_per_source_view_cell',center_mode_geometry_binding='selected_mode_own_mean_world',
              center_mode_exclusion='entire_source_image_before_mode_selection',maximum_prototypes_per_cell=4)
    atlas=dict(minimum_independent_mapping_views=2,maximum_anonymous_view_prototypes_per_texel=4,cell_size_m=.5)
    _validate_source_mode_center_contract(head,atlas)
    for key,value in [('source_isolated_nulls',False),('maximum_prototypes_per_cell',3)]:
        bad=head.copy();bad[key]=value
        with pytest.raises(ValueError):_validate_source_mode_center_contract(bad,atlas)
