import numpy as np
import torch
from feature_extract.tools.vfm.localization_lod_memory import LocalizationLoD
from feature_extract.tools.vfm.partial_overlap_matcher import PartialOverlapMatcher,masked_losses


def test_lod_identity_coverage_and_route_exclusion():
    rng=np.random.default_rng(2);world=np.c_[np.arange(48)*2.1,np.zeros((48,2))];features=rng.normal(size=(48,8));eligible=np.ones(48,bool);eligible[3]=False
    memory=LocalizationLoD(world,features,eligible);arrays=memory.arrays()
    assert set(arrays['leaf_members'])==set(np.flatnonzero(eligible))
    assert len(np.unique(arrays['leaf_members']))==47
    rows,*_=memory.nearest(features[[7,21,40]])
    assert rows.tolist()==[7,21,40]
    assert 3 not in arrays['fine_rows'] and 3 not in arrays['coarse_rows']


def test_direct_gateway_survives_corrupted_coarse_summary():
    world=np.c_[np.arange(12)*10.,np.zeros((12,2))];features=np.eye(12);memory=LocalizationLoD(world,features)
    memory.coarse_features[:]=features[0]
    rows,*_=memory.nearest(features[[11]])
    assert rows[0]==11
    assert np.array_equal(memory.world,world)


def test_matcher_is_candidate_and_query_permutation_equivariant():
    torch.manual_seed(3);model=PartialOverlapMatcher().eval();q=torch.randn(1,5,136);m=torch.randn(1,5,4,133);edges=torch.ones(1,5,5)-torch.eye(5)[None];sim=torch.randn(1,5,4,2)
    with torch.no_grad():
        o,i=model(q,m,edges,sim);perm=torch.tensor([2,0,3,1]);op,ip=model(q,m[:,:,perm],edges,sim[:,:,perm]);order=torch.tensor([4,1,3,0,2]);oq,iq=model(q[:,order],m[:,order],edges[:,order][:,:,order],sim[:,order])
    assert torch.allclose(o,op,atol=1e-5) and torch.allclose(i[:,:,perm],ip,atol=1e-5)
    assert torch.allclose(o[:,order],oq,atol=1e-5) and torch.allclose(i[:,order],iq,atol=1e-5)


def test_unknown_has_no_supervision_gradient_and_positive_modes_are_set_valued():
    overlap=torch.zeros(1,2,requires_grad=True);identity=torch.zeros(1,2,3,requires_grad=True);target=torch.tensor([[-1.,1.]]);positive=torch.tensor([[[False,False,False],[True,True,False]]]);known=positive.clone();known[0,1,2]=True
    loss,_,_=masked_losses(overlap,identity,target,positive,known);loss.backward()
    assert overlap.grad[0,0]==0 and torch.equal(identity.grad[0,0],torch.zeros(3))
    assert identity.grad[0,1,0]<0 and identity.grad[0,1,1]<0 and identity.grad[0,1,2]>0


def test_padding_does_not_create_nan_in_normal_batch():
    from feature_extract.tools.vfm.train_partial_overlap_matcher import batch
    examples=[]
    for n in [4,6]:examples.append(dict(query=torch.ones(n,136),map=torch.ones(n,3,133),edges=torch.eye(n),similarity=torch.ones(n,3,2),target=torch.zeros(n),positive=torch.zeros(n,3,dtype=torch.bool),known=torch.ones(n,3,dtype=torch.bool)))
    x=batch(examples,'cpu');model=PartialOverlapMatcher().eval();o,i=model(x['query'],x['map'],x['edges'],x['similarity'])
    assert torch.isfinite(o).all() and torch.isfinite(i).all() and (x['target'][0,4:]==-1).all()


def test_native_covariance_is_required_and_never_silently_zeroed():
    import pytest
    from feature_extract.tools.vfm.prepare_overlap_lod_training import atlas_uncertainty
    covariance=np.array([np.eye(3)*.03])
    assert np.allclose(atlas_uncertainty({'prototype_world_covariance_m2':covariance}),.03)
    with pytest.raises(KeyError):atlas_uncertainty({'world_covariance_m2':covariance})
    with pytest.raises(ValueError):atlas_uncertainty({'prototype_world_covariance_m2':np.zeros((1,3))})


def test_overlap_prior_changes_token_sampling_and_retains_exploration():
    from feature_extract.tools.vfm.token_hypothesis_ransac import guided_sample
    groups=[np.array([i]) for i in range(8)];scores=np.array([100.]*4+[.01]*4)
    rng=np.random.default_rng(11);hits=np.zeros(8,int)
    for _ in range(2000):
        sample=guided_sample(rng,groups,np.zeros((8,2)),np.zeros(8,int),scores,'overlap_prior')
        assert len(set(sample))==4
        hits[sample]+=1
    assert hits[:4].mean()>2*hits[4:].mean()
    assert hits.min()>300


def test_batched_lod_matches_reference_leaf_expansion_with_ties():
    rng=np.random.default_rng(43)
    world=rng.uniform(-12,12,(80,3));features=rng.integers(-1,2,(80,6)).astype(float);features[:,0]=1
    memory=LocalizationLoD(world,features);query=memory.features[[2,7,19,23]]
    _,_,_,_,cost=memory.nearest(query)
    coarse=query@memory.coarse_features.T;fine=query@memory.fine_features.T;expected=[]
    for a,b in zip(coarse,fine):
        parents=[]
        for j in np.argsort(-a,kind='stable'):
            p=int(memory.coarse_parent[j])
            if p not in parents:parents.append(p)
            if len(parents)==2:break
        leaves=[]
        for j in np.argsort(-b,kind='stable'):
            leaf=int(memory.fine_leaf[j])
            if memory.fine_parent[j] in parents and leaf not in leaves:leaves.append(leaf)
            if len(leaves)==2:break
        leaves.append(int(memory.fine_leaf[b.argmax()]))
        expected.append(sorted(set(leaves)))
    assert cost['opened_leaf_ids']==expected


def test_nonperiodic_depth_boundary_ignores_invalid_neighbors():
    from feature_extract.tools.vfm.partial_overlap_matcher import depth_boundary
    depth=np.zeros((36,64));depth[-1]=1;valid=np.ones_like(depth,bool)
    b=depth_boundary(depth,valid).reshape(36,64)
    assert not b[0].any() and b[-1].max()==1
    valid[-1]=False
    assert not depth_boundary(depth,valid).any()


def test_matcher_padding_batch_and_hidden_cache_invariance():
    torch.manual_seed(19);model=PartialOverlapMatcher().eval()
    q=torch.randn(1,5,136);m=torch.randn(1,5,4,133);e=torch.ones(1,5,5)-torch.eye(5)[None];s=torch.randn(1,5,4,2)
    with torch.no_grad():
        o,i=model(q,m,e,s)
        qp=torch.cat([q,torch.zeros(1,3,136)],1);mp=torch.cat([m,torch.randn(1,3,4,133)*100],1);sp=torch.cat([s,torch.randn(1,3,4,2)*100],1);ep=torch.zeros(1,8,8);ep[:,:5,:5]=e
        op,ip=model(qp,mp,ep,sp)
        qb=torch.cat([qp,torch.randn_like(qp)],0);mb=torch.cat([mp,torch.randn_like(mp)],0);eb=torch.cat([ep,torch.ones_like(ep)],0);sb=torch.cat([sp,torch.randn_like(sp)],0);ob,ib=model(qb,mb,eb,sb)
        oz,iz=model(torch.zeros_like(qp),mp,ep,sp)
    assert torch.allclose(o,op[:,:5],atol=1e-5) and torch.allclose(i,ip[:,:5],atol=1e-5)
    assert torch.allclose(o,ob[:1,:5],atol=1e-5) and torch.allclose(i,ib[:1,:5],atol=1e-5)
    assert torch.isfinite(oz).all() and torch.isfinite(iz).all()


def test_candidate_padding_does_not_change_valid_logits():
    torch.manual_seed(44);model=PartialOverlapMatcher().eval();q=torch.randn(1,4,136);m=torch.randn(1,4,3,133);s=torch.randn(1,4,3,2);e=torch.eye(4)[None]
    with torch.no_grad():
        o,i=model(q,m,e,s)
        op,ip=model(q,torch.cat([m,torch.zeros(1,4,2,133)],2),e,torch.cat([s,torch.randn(1,4,2,2)*100],2))
    assert torch.allclose(o,op,atol=1e-5) and torch.allclose(i,ip[:,:,:3],atol=1e-5)
    assert (ip[:,:,3:]<-100).all()
