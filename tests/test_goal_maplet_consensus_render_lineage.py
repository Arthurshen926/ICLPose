import json
import numpy as np
import pytest
from feature_extract.tools.vfm.build_goal_maplet_direct_plane_pnp_render_consistency import (
    _load_frozen_poses, _validate_consensus_camera_lineage,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256, canonical_json_sha256, file_sha256,
)


def _write(path, arrays, meta):
    meta=dict(meta, arrays_sha256=arrays_sha256(arrays))
    meta['content_sha256']=canonical_json_sha256({k:v for k,v in meta.items() if k!='content_sha256'})
    np.savez_compressed(path,**arrays,metadata_json=np.asarray(json.dumps(meta)))
    return meta


def _fixture(tmp_path):
    camera=tmp_path/'camera.npz';camera.write_bytes(b'camera-only fixture')
    base=dict(names=np.array(['q']),pose_w2c=np.eye(4)[None],usable=np.array([True]))
    parent=dict(base,candidate_correspondence_count=np.array([8]),pnp_inlier_count=np.array([6]))
    pm=_write(tmp_path/'parent.npz',parent,dict(artifact_type='goal_maplet_uncertainty_weighted_plane_pose_refinement_v1',query_pose_or_ground_truth_read=False,query_depth_or_scale_used_by_pose_solver=False,query_camera_only_inventory_file_sha256=file_sha256(camera)))
    selected=dict(base,selected_branch=np.array([0]),candidate_plane_geometry_objective=np.ones((1,2)),candidate_dense_normal_good_ray_recall=np.ones((1,2)))
    meta=dict(artifact_type='goal_maplet_coordinate_pose_geometry_consensus_v1',query_pose_or_ground_truth_read=False,selection_has_continuous_fusion_weight=False)
    manifest={}
    for prefix in ['primary','alternate']:
        meta[prefix+'_pose_file_sha256']=file_sha256(tmp_path/'parent.npz')
        meta[prefix+'_pose_content_sha256']=pm['content_sha256']
        manifest[prefix+'_pose']=str(tmp_path/'parent.npz')
    _write(tmp_path/'selected.npz',selected,meta)
    (tmp_path/'lineage.json').write_text(json.dumps(manifest))
    return camera,selected,meta


def test_consensus_render_authenticates_endpoints_without_inventing_counts(tmp_path):
    camera,_,_=_fixture(tmp_path)
    _validate_consensus_camera_lineage(tmp_path/'selected.npz',camera,tmp_path/'lineage.json')
    arrays,_=_load_frozen_poses(tmp_path/'selected.npz')
    assert 'candidate_correspondence_count' not in arrays
    np.testing.assert_array_equal(arrays['pose_w2c'],np.eye(4)[None])


@pytest.mark.parametrize('tamper',['pose','camera','endpoint'])
def test_consensus_lineage_rejects_substitution_even_with_resealed_selection(tmp_path,tamper):
    camera,selected,meta=_fixture(tmp_path)
    if tamper=='pose':
        selected['pose_w2c'][0,0,3]=1
        _write(tmp_path/'selected.npz',selected,meta)
    elif tamper=='camera':
        camera.write_bytes(b'another camera')
    else:
        (tmp_path/'parent.npz').write_bytes((tmp_path/'parent.npz').read_bytes()+b'changed')
    with pytest.raises(ValueError):
        _validate_consensus_camera_lineage(tmp_path/'selected.npz',camera,tmp_path/'lineage.json')
