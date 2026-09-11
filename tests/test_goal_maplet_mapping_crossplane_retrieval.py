import numpy as np
from feature_extract.tools.vfm.audit_goal_maplet_mapping_crossplane_retrieval import physical_labels


def test_close_cross_plane_and_cell_boundary_are_not_forced_negatives():
    np.testing.assert_array_equal(physical_labels([.1,.1,.3,.6],np.array([True,False,True,False])),[1,-1,-1,0])
def test_conflicting_mapping_targets_are_not_selected_by_distance():
    import numpy as np
    from feature_extract.tools.vfm.audit_goal_maplet_mapping_crossplane_retrieval import ambiguous_source_tokens
    bad=ambiguous_source_tokens(np.array([0,0,0,1]),np.array([2,2,3,2]),np.array([[0.,0,0],[1.,0,0],[0.,0,0],[2.,0,0]]))
    assert bad=={(0,2)}


def test_full_input_keeps_unknown_and_fixed_first_supervision():
    from feature_extract.tools.vfm.audit_goal_maplet_mapping_crossplane_retrieval import attach_supervision
    tokens=np.array([9,2,5,7])
    result=attach_supervision(tokens,np.array([2,2,7]),np.array([10,11,12]))
    np.testing.assert_array_equal(result,[-1,10,-1,12])
    np.testing.assert_array_equal(tokens,[9,2,5,7])
    np.testing.assert_array_equal(attach_supervision(tokens,[],[]),[-1]*4)


def test_context_ring_crosses_surface_boundary_without_wrapping_image():
    from feature_extract.tools.vfm.audit_goal_maplet_mapping_crossplane_retrieval import context_ring
    np.testing.assert_array_equal(context_ring([0],1,height=3,width=4),[1,4,5])
    np.testing.assert_array_equal(context_ring([0,1],1,height=3,width=4),[2,4,5,6])
    assert len(context_ring(np.arange(12),2,height=3,width=4))==0
    np.testing.assert_array_equal(context_ring([0,0],1,height=3,width=4),[1,4,5])


def test_full_mode_includes_image_with_no_mapping_observations(tmp_path,monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    from feature_extract.tools.vfm import audit_goal_maplet_mapping_crossplane_retrieval as m
    bank={'observation_offsets':np.array([0,4,8]),'world_points':np.zeros((8,3)),
          'radio_features':np.ones((8,1280),np.float32),'token_ids':np.tile(np.arange(4),2)}
    vis=SimpleNamespace(view_names=np.array(['seq1__a.npz','seq2__b.npz']),plane_offsets=np.array([0,2]))
    planes=SimpleNamespace(plane_ids=np.array([0]),centers_world=np.zeros((1,3)),frames_world=np.eye(3)[None])
    monkeypatch.setattr(m,'_load_observation_bank',lambda p:(bank,{'visibility_atlas_content_sha256':'h','content_sha256':'h'}))
    monkeypatch.setattr(m.PlaneVisibilityAtlas,'load_npz',lambda p:(vis,{'content_sha256':'h','planar_map_file_sha256':'h'}))
    monkeypatch.setattr(m.GeometryNativePlanarMap,'load_npz',lambda p:planes)
    monkeypatch.setattr(m,'_load_projection',lambda p:(np.eye(64,1280),{'observation_bank_content_sha256':'h'}))
    monkeypatch.setattr(m,'file_sha256',lambda p:'h')
    monkeypatch.setattr(m,'_records',lambda p:{})
    monkeypatch.setattr(m,'_radio',lambda n,r:np.ones((2304,1280),np.float32))
    monkeypatch.setattr(m.QueryPlaneRegions,'load_npz',lambda p:(SimpleNamespace(pixel_counts=np.array([5]),labels=None),{'uses_pose_or_ground_truth':False}))
    monkeypatch.setattr(m,'_region_token_support',lambda l,r:(np.arange(5),np.ones(5)))
    monkeypatch.setattr(m,'_mutual_matches',lambda q,f:(np.arange(len(q)),np.zeros(len(q),int),np.ones(len(q))))
    monkeypatch.setattr(m,'_metric_homography_filter_with_projection',lambda t,u,threshold_m:(np.ones(len(t),bool),u,np.ones(len(t),bool)))
    regions=tmp_path/'regions';regions.mkdir();(regions/'seq9__unlabelled.npz').touch()
    out=tmp_path/'result.json'
    argv=['audit']
    for k in ['observation_bank','visibility_atlas','planar_map','radio_projection']:argv += ['--'+k,'unused']
    argv+=['--output',str(out),'--query_plane_dir',str(regions),'--radio_manifest','unused']
    monkeypatch.setattr(sys,'argv',argv);m.main()
    with np.load(out.with_suffix('.npz')) as c:
        np.testing.assert_array_equal(c['query_token'],np.arange(5))
    with np.load(out.with_suffix('.labels.npz')) as l:
        np.testing.assert_array_equal(l['labels'],[-2]*5)
        assert not l['supervision_valid'].any()
        assert not l['supervision_available'].any()
    report=json.loads(out.read_text())
    assert report['query_region_count']==1
    assert report['after_homography']['unknown']==5


def test_conflict_checks_pairwise_diameter_not_only_first_target():
    from feature_extract.tools.vfm.audit_goal_maplet_mapping_crossplane_retrieval import ambiguous_source_tokens
    world=np.array([[0.,0,0],[.2,0,0],[-.2,0,0]])
    assert ambiguous_source_tokens(np.zeros(3,int),np.zeros(3,int),world)=={(0,0)}
