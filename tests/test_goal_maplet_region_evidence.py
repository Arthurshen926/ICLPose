import json
import numpy as np
import pytest
from feature_extract.tools.vfm.build_goal_maplet_native_region_augmentation import allocate_added_matches,quadrant_support
from feature_extract.tools.vfm.token_hypothesis_ransac import guided_sample
from feature_extract.tools.vfm.region_sampling_prior import load_region_sampling_prior
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256


def test_balanced_budget_keeps_minor_origin_and_preserves_uncapped_order():
    triples={(0,0):.99,(1,1):.98,(2,2):.97,(32,3):.7,(1200,4):.6}
    owners={k:0 if k[1]<3 else 1 for k in triples}
    assert allocate_added_matches(triples,owners,2,'cosine')==[(0,0),(1,1)]
    assert allocate_added_matches(triples,owners,3,'balanced')==[(0,0),(32,3),(1200,4)]
    assert allocate_added_matches(triples,owners,10,'balanced')==allocate_added_matches(triples,owners,10)
    with pytest.raises(ValueError):allocate_added_matches(triples,owners,0)


def test_quadrant_support_uses_retrieval_origin_and_equal_ties():
    a=np.array([[.8],[.2],[.5],[.5]])
    assert quadrant_support(a,0,0)==1.
    assert quadrant_support(a,0,32)==.25
    assert quadrant_support(a,0,18*64)==.5
    assert quadrant_support(a,0,18*64+32)==.5


def test_context_prior_retains_uniform_exploration_and_unique_tokens():
    groups=[np.array([2*i,2*i+1]) for i in range(6)];weights=np.tile([.01,1.],6);rng=np.random.default_rng(12);draws=[]
    for _ in range(500):
        rows=guided_sample(rng,groups,None,None,weights,'context_prior')
        assert len(set(rows//2))==4
        draws.extend(rows%2)
    rate=np.mean(draws)
    assert .65<rate<.85


def test_prior_rejects_wrong_binding_and_tampering(tmp_path):
    corr=tmp_path/'corr';corr.write_text('frozen')
    arrays=dict(names=np.array(['q']),correspondence_offsets=np.array([0,2]),sampling_weights=np.array([1.,.25]))
    meta=dict(artifact_type='goal_maplet_region_sampling_prior_v1',query_pose_or_ground_truth_read=False,correspondence_file_sha256=file_sha256(corr),arrays_sha256=arrays_sha256(arrays))
    meta['content_sha256']=canonical_json_sha256(meta)
    path=tmp_path/'prior.npz'
    np.savez(path,**arrays,metadata_json=np.array(json.dumps(meta)))
    source=dict(names=arrays['names'],correspondence_offsets=arrays['correspondence_offsets'],query_tokens=np.array([1,2]))
    np.testing.assert_array_equal(load_region_sampling_prior(path,corr,source),[1.,.25])
    corr.write_text('changed')
    with pytest.raises(ValueError,match='lineage'):load_region_sampling_prior(path,corr,source)


def test_shared_evidence_ties_missing_and_invalid_candidates():
    from feature_extract.tools.vfm.select_goal_maplet_shared_evidence import select_supported
    counts=np.array([[0,0],[4,5],[5,4],[3,4],[4,3]])
    usable=np.array([[1,1],[1,1],[1,1],[1,0],[0,1]],bool)
    baseline=np.array([1,0,1,0,1],np.int8)
    np.testing.assert_array_equal(select_supported(counts,usable,baseline,np.ones(5,bool)),[1,1,0,0,1])
    np.testing.assert_array_equal(select_supported(counts,usable,baseline,np.zeros(5,bool)),baseline)


def test_patch_offsets_share_translation_and_respect_invalid_probes():
    from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import select_offsets
    sim=np.zeros((4,9));sim[:2,0]=2;sim[2,2]=3;sim[3,3]=4
    np.testing.assert_array_equal(select_offsets(sim,[np.arange(3),np.array([3])],False),[0,0,2,3])
    np.testing.assert_array_equal(select_offsets(sim,[np.arange(3),np.array([3])],True),[0,0,0,4])
    sim[1,0]=-np.inf
    np.testing.assert_array_equal(select_offsets(sim,[np.arange(3)],True)[:3],[2,2,2])
    np.testing.assert_array_equal(select_offsets(np.zeros((3,9)),[np.arange(3)],True),[4,4,4])


def test_native_fine_inventory_checks_anchor_and_within_token_contract(tmp_path):
    from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load
    arrays=dict(names=np.array(['q']),correspondence_offsets=np.array([0,1]),world_points=np.array([[0.,0.,1.]]),query_tokens=np.array([0]),provenance_region_plane_atlas_row=np.zeros((1,3),int),camera_matrices=np.eye(3)[None],radial_k1=np.zeros(1),prototype_atlas_row=np.array([0]),query_plane_visible_fraction=np.ones(1),radio_match_score=np.ones(1),prototype_world_covariance_m2=np.eye(3)[None],prototype_plane_pixel_purity=np.ones(1),prototype_plane_depth_dispersion_m=np.zeros(1),query_measurements_xy=np.array([[1.5,1.5]]),query_measurement_variance_px2=np.ones(1),correspondence_match_probability=np.ones(1))
    meta=dict(artifact_type='goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v5',pose_or_ground_truth_opened=False,query_measurement_semantics='native_matched_fine_query_pixel_update_inside_original_4x4_token',query_measurement_uncertainty_semantics='predicted_isotropic_centroid_measurement_variance_px2_not_surface_footprint',fine_readout_protocol=dict(anchors_fixed=True,query_ground_truth_read=False),fine_readout_arm='joint',coarse_correspondence_file_sha256='a'*64,frozen_initial_pose_sha256='b'*64,fine_measurement_variance_recalibrated=False)
    path=tmp_path/'fine.npz'
    def write():
        meta['arrays_sha256']=arrays_sha256(arrays)
        np.savez(path,**arrays,metadata_json=np.array(json.dumps(meta)))
    write();_load(path)
    arrays['query_measurements_xy'][0,0]=4.;write()
    with pytest.raises(ValueError,match='native fine'):_load(path)
    arrays['query_measurements_xy'][0,0]=1.5;meta['fine_readout_protocol']['anchors_fixed']=False;write()
    with pytest.raises(ValueError,match='native fine'):_load(path)
    meta['fine_readout_protocol']['anchors_fixed']=True
    meta['fine_readout_arm']='fine_calibrated';meta['fine_measurement_variance_recalibrated']=True;write()
    with pytest.raises(ValueError,match='calibration'):_load(path)
    meta['fine_readout_protocol']['mapping_calibration']=dict(query_ground_truth_read=False,heldout_used_to_fit=False,file_sha256='c'*64,alpha=.35,variance_scale=1.25);write();_load(path)
    meta['fine_readout_protocol']['mapping_calibration']['heldout_used_to_fit']=True;write()
    with pytest.raises(ValueError,match='calibration'):_load(path)
