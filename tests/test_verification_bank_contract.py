import numpy as np
import pytest
from feature_extract.tools.vfm.verification_bank_contract import validate_bank,validate_poses
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256


def fixture():
    bank=dict(names=np.array(['query']),offsets=np.array([0,2]),prototype_rows=np.array([0,1]),query_tokens=np.array([0,2]),pixels=np.zeros((2,2)),camera_matrices=np.eye(3)[None],radial_k1=np.zeros(1))
    meta=dict(arrays_sha256=arrays_sha256(bank),source_sha256={'/tmp/map.npz':'map','/tmp/atlas.npz':'atlas'})
    return bank,meta


def check(bank,meta):
    validate_bank(bank,meta,np.ones((2,3)),'/tmp/map.npz','map','atlas',np.array(['query']),np.eye(3)[None],np.zeros(1))


def test_valid_bank_and_poses():
    check(*fixture());validate_poses(np.eye(4)[None])


@pytest.mark.parametrize('field,value',[('offsets',np.array([1,2])),('prototype_rows',np.array([0,2])),('query_tokens',np.array([0,2304])),('camera_matrices',np.zeros((1,3,3))),('names',np.array(['other']))])
def test_structural_mismatch_rejected_even_with_recomputed_array_hash(field,value):
    bank,meta=fixture();bank[field]=value;meta['arrays_sha256']=arrays_sha256(bank)
    with pytest.raises(ValueError):check(bank,meta)


def test_wrong_map_identity_rejected():
    bank,meta=fixture();meta['source_sha256']['/tmp/map.npz']='different'
    with pytest.raises(ValueError):check(bank,meta)


def test_reflection_pose_rejected():
    p=np.eye(4)[None];p[0,0,0]=-1
    with pytest.raises(ValueError):validate_poses(p)
