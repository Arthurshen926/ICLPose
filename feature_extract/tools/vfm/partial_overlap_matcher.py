"""Candidate-conditioned overlap and set-valued anchor identity matcher.

Context attention and depth-gated measurement messages are separate. Geometry
messages do not estimate a scale from provisional image-to-map identities.
"""
import numpy as np
import torch
from torch import nn
from feature_extract.vfm.localization_goal_maplet.structured_local_memory import normalise
from feature_extract.tools.vfm.build_goal_maplet_fine_readout import sample_grid


def query_inputs(grids,points,normals,reliable,tokens):
    tokens=np.asarray(tokens,int);xy=np.c_[tokens%64*4+1.5,tokens//64*4+1.5]
    coarse=normalise(grids[0].reshape(2304,64));fine=sample_grid(grids[1],xy)
    z=np.maximum(points[:,2],1e-6);median=np.median(z[reliable]) if reliable.any() else 1.
    logz=np.log(z/median);depth=logz.reshape(36,64);boundary=np.maximum(np.abs(depth-np.roll(depth,1,0)),np.abs(depth-np.roll(depth,1,1))).ravel();boundary=np.clip(boundary,0,2)
    qgeo=np.c_[xy/np.array([128,72])-1,np.clip(logz[tokens],-4,4),normals[tokens],reliable[tokens],boundary[tokens]]
    feature=np.c_[coarse[tokens],fine,qgeo].astype(np.float32)
    pixel_distance=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
    edge=(pixel_distance<=16)&(np.abs(logz[tokens,None]-logz[tokens][None,:])<=np.log(1.25))&reliable[tokens,None]&reliable[tokens][None,:]
    np.fill_diagonal(edge,False)
    return feature,edge.astype(np.float32),xy


def make_pair(grids,points,normals,reliable,tokens,region_rows,world,coarse_map,fine_map,covariance,map_normals,available,k=16):
    q,edges,xy=query_inputs(grids,points,normals,reliable,tokens);g=np.asarray(region_rows,int)
    if len(g)<k:raise ValueError('too few region anchors')
    sim=q[:,:64]@coarse_map[g].T;rank=np.argsort(-sim,axis=1,kind='stable')[:,:k];ids=g[rank]
    # Keep view modes and identities until learned inference; no texel averaging.
    w=world[ids];center=world[g].mean(0);radius=np.linalg.norm(w-center,axis=-1)/6
    mean_n=normalise(map_normals[g].mean(0)[None])[0];normal_coherence=np.abs(map_normals[ids]@mean_n)
    geometry=np.stack([radius,np.log1p(covariance[ids]),normal_coherence,available[ids].astype(float),np.linalg.norm(w-w.mean(1,keepdims=True),axis=-1)/6],axis=-1)
    m=np.concatenate([coarse_map[ids],fine_map[ids],geometry],axis=-1).astype(np.float32)
    pair_sim=np.stack([np.einsum('nd,nkd->nk',q[:,:64],coarse_map[ids]),np.einsum('nd,nkd->nk',q[:,64:128],fine_map[ids])],axis=-1).astype(np.float32)
    reverse=np.argmax(sim,axis=0)[rank[:,0]]==np.arange(len(tokens))
    return dict(query=q,map=m,edges=edges,similarity=pair_sim,ids=ids,tokens=np.asarray(tokens),pixels=xy,mutual=reverse)


class PartialOverlapMatcher(nn.Module):
    def __init__(self,width=64):
        super().__init__();self.width=width
        self.query_encoder=nn.Sequential(nn.Linear(136,width),nn.LayerNorm(width),nn.GELU())
        self.map_encoder=nn.Sequential(nn.Linear(133,width),nn.LayerNorm(width),nn.GELU())
        self.context=nn.MultiheadAttention(width,4,batch_first=True)
        self.cross_query=nn.Linear(width,width,bias=False);self.cross_map=nn.Linear(width,width,bias=False)
        self.cross_update=nn.Sequential(nn.Linear(width*2,width),nn.GELU(),nn.LayerNorm(width))
        self.overlap=nn.Sequential(nn.Linear(width*3+4,width),nn.GELU(),nn.Linear(width,1))
        self.geometry_update=nn.Linear(width,width,bias=False)
        self.identity=nn.Sequential(nn.Linear(width*3+2,width),nn.GELU(),nn.Linear(width,1))
        nn.init.zeros_(self.identity[-1].weight);nn.init.zeros_(self.identity[-1].bias)
        self.feature_weight=nn.Parameter(torch.tensor([10.,0.]))

    def forward(self,query,map,edges,similarity):
        q=self.query_encoder(query);m=self.map_encoder(map)
        context,_=self.context(q,q,q,key_padding_mask=query.abs().sum(-1)==0,need_weights=False);q=q+context
        appearance=(similarity*self.feature_weight).sum(-1)
        logits=torch.einsum('bnd,bnkd->bnk',self.cross_query(q),self.cross_map(m))/self.width**.5+appearance
        attended=(logits.softmax(-1)[...,None]*m).sum(-2)
        q=q+self.cross_update(torch.cat([q,attended],-1))
        region=m.mean((1,2))[:,None].expand_as(q)
        stats=torch.cat([similarity.max(-2).values,similarity.mean(-2)],-1)
        overlap=self.overlap(torch.cat([q,attended,region,stats],-1)).squeeze(-1)
        gate=edges*overlap.sigmoid()[:,:,None]*overlap.sigmoid()[:,None,:]
        message=torch.bmm(gate,q)/gate.sum(-1,keepdim=True).clamp_min(1)
        refined=q+self.geometry_update(message)
        expanded=refined[:,:,None].expand_as(m)
        identity=appearance+self.identity(torch.cat([expanded,m,expanded*m,similarity],-1)).squeeze(-1)
        return overlap,identity


def masked_losses(overlap,identity,target,positive,known):
    valid=target>=0
    if valid.any():
        loss_overlap=nn.functional.binary_cross_entropy_with_logits(overlap[valid],target[valid].float())
    else:loss_overlap=overlap.sum()*0
    has_positive=positive.any(-1)
    if has_positive.any():
        denominator=identity.masked_fill(~known,-1e4).logsumexp(-1)
        numerator=identity.masked_fill(~positive,-1e4).logsumexp(-1)
        loss_identity=(denominator-numerator)[has_positive].mean()
    else:loss_identity=identity.sum()*0
    return loss_overlap+loss_identity,loss_overlap,loss_identity
