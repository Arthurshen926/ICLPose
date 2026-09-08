import numpy as np
from feature_extract.tools.vfm.mapping_retrieved_negatives import mine


def run(policy,query_world=None):
    features=np.array([[1.,0.],[1.,0.],[.99,.1],[.99,.1],[0.,1.],[0.,1.]],np.float32)
    features/=np.linalg.norm(features,axis=1,keepdims=True)
    return mine(np.array([10,10]),np.tile([1.,0.],(2,1)),np.array([9,9]),np.zeros(2,int),np.zeros(2,int),
                np.zeros((2,3)) if query_world is None else query_world,
                features,np.array([[0.,0.,0.]]*2+[[2.,0.,0.]]*2+[[10.,0.,0.]]*2),
                np.array([0,1]*3),np.array([0,0,1,1,2,2]),np.zeros(6,int),4,.5,policy,topk=4)


def test_hard_negative_is_retrieved_not_farthest_and_duplicates_share_choice():
    rows,valid,audit=run('radio_topk')
    assert valid.all() and np.all(rows==2)
    assert audit['negative_distance_median_m']==2.
    far,valid,audit=run('pool_far')
    assert np.all(far==4)
    assert audit['negative_distance_median_m']==10.


def test_geometry_does_not_change_retrieval_candidate_count():
    _,_,a=run('radio_topk');_,_,b=run('radio_topk',np.ones((2,3))*100)
    assert a['candidate_count_median']==b['candidate_count_median']==6


def test_source_exclusion_precedes_two_view_gate():
    rows,valid,audit=mine(np.array([0]),np.array([[1.,0.]]),np.array([1]),np.array([0]),np.array([0]),np.zeros((1,3)),
        np.array([[1.,0.],[1.,0.]]),np.ones((2,3))*3,np.array([1,2]),np.array([1,1]),np.array([0,0]),4,.5,'radio_topk')
    assert not valid.any() and audit['valid_pair_count']==0


def test_score_audit_ties_and_perfect_ranking():
    from feature_extract.tools.vfm.audit_goal_maplet_retrieved_match_scores import metrics
    assert metrics([.5,.5],[.5,.5])['sampled_auc']==.5
    assert metrics([.9,.8],[.1,.2])['sampled_auc']==1.


def test_positive_gate_does_not_fit_evaluation_scores():
    from feature_extract.tools.vfm.audit_goal_maplet_retrieved_match_scores import positive_recall_gate
    bank=dict(calibration_rows=np.arange(4),evaluation_rows=np.arange(4,8),source_views=np.arange(8),
              positive_probability=np.linspace(.2,.9,8),negative_probability=np.zeros(8),head_content_sha256=np.asarray('test'))
    threshold=positive_recall_gate(bank)['threshold']
    bank['positive_probability'][4:]=0
    assert positive_recall_gate(bank)['threshold']==threshold


def test_match_only_optimizer_cannot_change_coordinate_weights():
    import torch
    from feature_extract.tools.vfm.train_goal_maplet_mapping_canonical_subtoken_head import _freeze_coordinate_parameters
    from feature_extract.tools.vfm.train_goal_maplet_mapping_subtoken_head import MappingSurfaceCoordinateHead
    torch.manual_seed(12);model=MappingSurfaceCoordinateHead(4,8)
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    _freeze_coordinate_parameters(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    q=torch.randn(5,4);m=torch.randn(5,4);tokens=torch.zeros(5,dtype=torch.long)
    outputs=model(q,m,tokens);outputs[-1].sum().backward();optimizer.step()
    assert not torch.equal(model.match.weight,before['match.weight'])
    for key,value in model.state_dict().items():
        if not key.startswith('match.'):assert torch.equal(value,before[key])
    after=model(q,m,tokens)
    for i in range(4):assert torch.equal(outputs[i],after[i])
