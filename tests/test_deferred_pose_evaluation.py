import json
import numpy as np
from feature_extract.tools.vfm.deferred_pose_evaluation import write_deferred_evaluation
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def test_deferred_status_binds_frozen_pose_without_inventing_metrics(tmp_path):
    frozen=tmp_path/'pose.npz'
    np.savez(frozen,pose_w2c=np.eye(4)[None])
    output=tmp_path/'reports/status.json'
    write_deferred_evaluation(output,frozen,1)
    report=json.loads(output.read_text())
    assert report['evaluation_deferred']
    assert report['query_pose_or_ground_truth_opened'] is False
    assert report['frozen_pose_sha256']==file_sha256(frozen)
    assert report['query_count']==1
    assert not set(report)&{'errors','rows','recall_percent','translation_mean_m'}
