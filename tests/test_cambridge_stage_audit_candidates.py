import numpy as np
from feature_extract.tools.vfm.audit_cambridge_complete_stages import candidates

class Arrays(dict):
    @property
    def files(self):return list(self)

def test_ragged_candidates_respect_query_offsets_even_when_counts_match():
    poses=np.repeat(np.eye(4)[None],2,axis=0);poses[:,0,3]=[10,20]
    z=Arrays(names=np.array(['a','b']),candidate_offsets=np.array([0,0,2]),candidate_pose_w2c=poses)
    v=candidates(z,['a','b'])
    assert len(v[0])==0 and len(v[1])==2
    assert v[1][1][0,3]==20

def test_query_order_and_unusable_predictions_are_not_silently_mixed():
    z=Arrays(names=np.array(['a','b']),pose_w2c=np.repeat(np.eye(4)[None],2,axis=0),usable=np.array([True,False]))
    assert candidates(z,['b','a'])==[[],[]]
    v=candidates(z,['a','b']);assert len(v[0])==1 and not v[1]
