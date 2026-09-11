import numpy as np
import pytest
from feature_extract.tools.vfm.audit_goal_maplet_context_candidate_control import compare
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def artifact(path,context=False,shift=0.):
    np.savez(path,query_token=np.arange(4),prototype_world=np.ones((4,3))+shift,
             association_features=np.ones((4,5 if context else 4)))
    np.savez(path.with_suffix('.labels.npz'),labels=np.array([1,0,-1,-2]),
             supervision_valid=np.array([1,1,0,0],bool),supervision_available=np.array([1,1,1,0],bool),
             frozen_candidate_sha256=np.asarray(file_sha256(path)))


def test_context_control_rejects_geometry_change(tmp_path):
    a=tmp_path/'a.npz';b=tmp_path/'b.npz';artifact(a);artifact(b,True)
    assert compare(a,b)['identical_supervision']
    artifact(b,True,.01)
    with pytest.raises(ValueError,match='changed candidates'):compare(a,b)


def test_context_control_rejects_stale_labels(tmp_path):
    a=tmp_path/'a.npz';b=tmp_path/'b.npz';artifact(a);artifact(b,True)
    np.savez(b.with_suffix('.labels.npz'),frozen_candidate_sha256=np.asarray('stale'))
    with pytest.raises(ValueError,match='lineage'):compare(a,b)
