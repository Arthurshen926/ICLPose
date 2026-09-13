import json
import sys
import numpy as np
import pytest
from feature_extract.tools.vfm.retain_goal_maplet_base_bound_rescue import main
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256


def test_consensus_rescue_checks_reference_binding(tmp_path,monkeypatch):
    def save(path,arr,extra):
        meta=dict(artifact_type='goal_maplet_coordinate_pose_geometry_consensus_v1',query_pose_or_ground_truth_read=False,source_rgb_stored_or_consumed_at_runtime=False,arrays_sha256=arrays_sha256(arr),**extra)
        meta['content_sha256']=canonical_json_sha256(meta)
        np.savez(path,**arr,metadata_json=np.array(json.dumps(meta)))
        return meta
    pose=np.repeat(np.eye(4)[None],2,axis=0)
    arr=dict(names=np.array(['a','b']),pose_w2c=pose,usable=np.ones(2,bool))
    ref=tmp_path/'ref.npz';rm=save(ref,arr,{})
    alt=tmp_path/'alt.npz';aa={**arr,'pose_w2c':pose.copy()};aa['pose_w2c'][:,0,3]=2;am=save(alt,aa,{})
    current=tmp_path/'current.npz';ca={**arr,'pose_w2c':pose.copy()};ca['pose_w2c'][1,0,3]=3;save(current,ca,{})
    verifier=tmp_path/'verifier.npz';va={**aa,'selected_branch':np.ones(2,np.int8)}
    bindings=dict(candidate_order='primary_then_alternate',primary_pose_file_sha256=file_sha256(ref),alternate_pose_file_sha256=file_sha256(alt),primary_pose_content_sha256=rm['content_sha256'],alternate_pose_content_sha256=am['content_sha256'])
    save(verifier,va,bindings)
    output=tmp_path/'out.npz'
    def run():
        monkeypatch.setattr(sys,'argv',['rescue','--current',str(current),'--reference',str(ref),'--alternative',str(alt),'--verifier',str(verifier),'--output',str(output)])
        main()
    run()
    with np.load(output) as z:
        assert z['selected_branch'].tolist()==[1,0]
        assert np.array_equal(z['pose_w2c'][1],ca['pose_w2c'][1])
    output.unlink();bindings['primary_pose_file_sha256']='wrong';save(verifier,va,bindings)
    with pytest.raises(ValueError,match='binding differs'):run()
    assert not output.exists()
