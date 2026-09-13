"""Pose-free local context retrieval and appearance-diverse query coverage.

These are deterministic controls, not learned membership or calibrated overlap.
Recognition uses all query contexts; metric correspondences still require MoGe.
"""
import numpy as np
from scipy.ndimage import uniform_filter
from feature_extract.vfm.localization_goal_maplet.structured_local_memory import normalise


def diverse_tokens(features, valid, budget=256):
    """Half uniform geometric coverage, half farthest appearance coverage."""
    features=normalise(np.asarray(features,float));ids=np.flatnonzero(valid)
    if budget<1:raise ValueError('positive budget required')
    if len(ids)<=budget:return ids
    selected=list(ids[np.linspace(0,len(ids)-1,budget//2,dtype=int)])
    similarity=(features[ids]@features[selected].T).max(1)
    available=np.ones(len(ids),bool);available[np.isin(ids,selected)]=False
    while len(selected)<budget:
        j=int(np.argmin(np.where(available,similarity,np.inf)))
        selected.append(int(ids[j]));available[j]=False
        similarity=np.maximum(similarity,features[ids]@features[ids[j]])
    return np.sort(selected)


def local_modes(world, contexts, available, voxel_m=2., modes_per_voxel=4):
    """Keep actual context exemplars; no averaging of distinct map identities."""
    world=np.asarray(world);contexts=normalise(np.asarray(contexts,float))
    ids=np.flatnonzero(available & np.isfinite(contexts).all(1)&(np.linalg.norm(contexts,axis=1)>.5))
    if not len(ids):raise ValueError('no available context members')
    _,groups=np.unique(np.floor(world[ids]/voxel_m).astype(np.int64),axis=0,return_inverse=True)
    order=np.argsort(groups,kind='stable');cuts=np.r_[0,np.flatnonzero(np.diff(groups[order]))+1,len(order)];result=[]
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        rows=ids[order[lo:hi]];f=contexts[rows];j=int(np.argmax(f@normalise(f.mean(0)[None])[0]));chosen=[j];sim=f@f[j]
        for _ in range(min(modes_per_voxel,len(rows))-1):
            score=sim.copy();score[chosen]=np.inf;j=int(np.argmin(score));chosen.append(j);sim=np.maximum(sim,f@f[j])
        result.extend(rows[chosen])
    return np.asarray(result,int)


def retrieve_local_regions(context_grid, descriptors, centers, count=4, separation_m=3.):
    """Greedy coverage of distinct local appearances at three context scales.

    Every descriptor retains its location/scale; a small query patch can propose
    a region without winning a quadrant-wide mean. Similar query patches lose
    weight after a proposal, preventing large repeated areas dominating votes.
    """
    grid=np.asarray(context_grid,float);h,w,d=grid.shape
    query=np.concatenate([normalise(uniform_filter(grid,size=(size,size,1),mode='nearest')[1::3,1::3].reshape(-1,d)) for size in (1,3,7)])
    sim=query@normalise(np.asarray(descriptors,float)).T
    active=np.ones(len(centers),bool);weight=np.ones(len(query));chosen=[];evidence=[]
    for _ in range(count):
        score=(sim+1)*weight[:,None];score[:,~active]=-np.inf
        i,j=np.unravel_index(np.argmax(score),score.shape)
        if not active[j]:break
        chosen.append(int(j));evidence.append(dict(query_patch=int(i),similarity=float(sim[i,j]),weighted_score=float(score[i,j])))
        # Continuous novelty weighting, not an overlap probability or null gate.
        weight=np.minimum(weight,np.clip(1-query@query[i],0,1))
        active &= np.linalg.norm(centers-centers[j],axis=1)>=separation_m
    return chosen,evidence


def retrieve_anchor_regions(query, map_features, world, count=4):
    """Local features vote for metric neighborhoods without context averaging.

    Standardized similarity and inverse appearance density balance query votes.
    Full-map scores are uncalibrated retrieval evidence, not probabilities.
    """
    query=normalise(query);map_features=normalise(map_features)
    tokens=diverse_tokens(query,np.ones(len(query),bool))
    sim=query[tokens]@map_features.T;rows=sim.argmax(1);best=sim.max(1)
    density=np.maximum(np.maximum(query[tokens]@query[tokens].T,0)**8 @ np.ones(len(tokens)),1)
    weights=(best-sim.mean(1))/np.maximum(sim.std(1),1e-6)/density
    centers=np.asarray(world)[rows];dist=np.linalg.norm(centers[:,None]-centers[None,:],axis=-1)
    active=np.ones(len(rows),bool);used=np.zeros(len(rows),bool);chosen=[];evidence=[]
    for _ in range(count):
        if not active.any():break
        score=(dist<=6)@(weights*~used);j=int(np.argmax(np.where(active,score,-np.inf)))
        if score[j]<=0:break
        chosen.append(int(rows[j]));evidence.append(dict(query_token=int(tokens[j]),similarity=float(best[j]),coverage_score=float(score[j])))
        used |= dist[j]<=6;active &= dist[j]>=3
    return chosen,evidence
