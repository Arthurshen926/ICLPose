import torch
from feature_extract.tools.vfm.shared_identity_matcher import SharedIdentityMatcher, identity_objective


def inputs():
    torch.manual_seed(9)
    q=torch.randn(1,7,136);m=torch.randn(1,7,4,133);e=torch.ones(1,7,7)-torch.eye(7)[None];s=torch.randn(1,7,4,2)
    ids=torch.arange(28).reshape(1,7,4)%9
    model=SharedIdentityMatcher().eval()
    torch.nn.init.normal_(model.head[-1].weight,std=.1)
    return model,[q,m,e,s,ids]


def test_shared_identity_is_equivariant_to_token_and_candidate_permutations():
    model,x=inputs();q,m,e,s,ids=x;p=torch.tensor([2,0,5,3,1,6,4]);k=torch.tensor([2,0,3,1])
    a=model(*x);b=model(q[:,p],m[:,p][:,:,k],e[:,p][:,:,p],s[:,p][:,:,k],ids[:,p][:,:,k])
    assert torch.allclose(b,a[:,p][:,:,k],atol=2e-6)
    assert torch.allclose(a,model(q,m,e,s,100-ids),atol=2e-6)


def test_shared_identity_padding_and_hidden_ids_do_not_change_valid_outputs():
    model,x=inputs();a=model(*x);q,m,e,s,ids=x
    b=model(torch.cat([q,torch.zeros_like(q[:,:2])],1),torch.cat([m,torch.zeros_like(m[:,:2])],1),torch.nn.functional.pad(e,(0,2,0,2)),torch.cat([s,torch.randn_like(s[:,:2])],1),torch.cat([ids,torch.full_like(ids[:,:2],999)],1))
    assert torch.allclose(a,b[:,:7],atol=2e-6)
    assert torch.isfinite(model(q*0,m*0,e,s,ids)).all()


def test_unknown_is_not_a_supervised_negative():
    logits=torch.tensor([[[1.,2.,3.]]],requires_grad=True);sim=torch.zeros(1,1,3,2)
    positive=torch.tensor([[[True,False,False]]]);known=torch.tensor([[[True,True,False]]])
    _,supervised,_=identity_objective(logits,sim,positive,known,torch.ones_like(known))
    supervised.backward();assert logits.grad[0,0,2]==0


def test_reciprocal_identity_rejects_competing_query_and_preserves_physical_keys():
    import numpy as np
    from feature_extract.tools.vfm.shared_identity_matcher import reciprocal_identity
    scores=np.array([[5.,1.],[6.,2.],[1.,8.]])
    ids=np.array([[7,8],[7,9],[7,10]])
    rows,columns=reciprocal_identity(scores,ids,np.ones_like(scores),.5)
    assert rows.tolist()==[1,2] and columns.tolist()==[0,1]
    a=reciprocal_identity(scores,100-ids,np.ones_like(scores),.5)
    assert np.array_equal(rows,a[0]) and np.array_equal(columns,a[1])


def test_ray_exclusion_keeps_hidden_near_ray_and_missing_depth_unknown():
    import numpy as np
    from feature_extract.tools.vfm.complete_identity_ray_negatives import ray_excluded_known
    ids=np.tile(np.arange(3),(2,1));known=np.zeros((2,3),bool);positive=known.copy()
    uv=np.array([[1.,0.],[10.,0.],[20.,0.]])
    result=ray_excluded_known(known,positive,ids,np.zeros((2,2)),uv,np.array([True,True,False]),np.array([1.,np.nan]))
    assert result.tolist()==[[False,True,False],[False,False,False]]


def test_appearance_control_cannot_use_anchor_ids_or_absolute_features():
    from feature_extract.tools.vfm.shared_identity_matcher import AppearanceIdentityMatcher
    _,x=inputs();q,m,e,s,ids=x;model=AppearanceIdentityMatcher()
    a=model(*x);b=model(q*3,m*2,e*0,s,ids+100)
    assert torch.equal(a,b)
    assert sum(p.numel() for p in model.parameters())==2


def test_frontend_loads_shared_and_appearance_checkpoints():
    from feature_extract.tools.vfm.train_shared_identity_matcher import DecoupledMatcher
    from feature_extract.tools.vfm.run_overlap_lod_frontend import make_matcher
    _,x=inputs()
    for shared,appearance in [(True,False),(False,False),(False,True)]:
        model=DecoupledMatcher(shared=shared,appearance_only=appearance).eval()
        c=dict(metadata=dict(shared_physical_identity=shared,appearance_only=appearance),state_dict=model.state_dict())
        loaded=make_matcher(c).eval();loaded.load_state_dict(c['state_dict'])
        with torch.no_grad():
            a=model(*x);b=loaded(*x)
        assert all(torch.equal(aa,bb) for aa,bb in zip(a,b))
