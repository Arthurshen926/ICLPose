"""Token-level multi-hypothesis PnP: unique sampling, support and local refinement."""
import cv2
import numpy as np
import time


def guided_sample(rng, groups, centers, planes, scores, policy, plane_groups=None, group_ids=None):
    """Retain 50% uniform exploration; guide sampling, never remove candidates."""
    if policy == 'row_uniform':
        return rng.choice(np.concatenate(groups),4,replace=False)
    if policy == 'uniform' or rng.random() < .5:
        chosen=rng.choice(len(groups),4,replace=False)
        return np.array([groups[g][rng.integers(len(groups[g]))] for g in chosen])
    if policy == 'context_prior':
        chosen = rng.choice(len(groups), 4, replace=False)
        return np.asarray([rng.choice(groups[g], p=scores[groups[g]]/scores[groups[g]].sum()) for g in chosen])
    if group_ids is None:
        group_ids=np.empty(len(planes),np.int64)
        for i,rows in enumerate(groups):group_ids[rows]=i
    if plane_groups is None:plane_groups={int(p):np.flatnonzero(planes==p) for p in np.unique(planes)}
    chosen=[];sample=[];used_planes=set()
    for _ in range(4):
        available=np.ones(len(groups),bool);available[chosen]=False
        families=np.array([p for p,rows in plane_groups.items() if np.any(available[group_ids[rows]])])
        unseen=families[~np.isin(families,list(used_planes))]
        family=int(rng.choice(unseen if len(unseen) else families))
        rows=plane_groups[family];rows=rows[available[group_ids[rows]]]
        weight=np.ones(len(rows))
        if chosen:
            distance=np.min(np.sum((centers[group_ids[rows],None]-centers[chosen][None])**2,axis=2),axis=1)
            weight=distance+1.0
        if policy=='geometry_score':
            # Scores have only within-plane training support. Never compare them
            # across planes; equal scores get equal rank and every row stays live.
            rank=np.searchsorted(np.unique(scores[rows]),scores[rows])+1
            weight*=rank
        _,inverse,counts=np.unique(group_ids[rows],return_inverse=True,return_counts=True)
        weight/=counts[inverse]  # Extra modes do not multiply token sampling mass.
        row=int(rng.choice(rows,p=weight/weight.sum()))
        chosen.append(int(group_ids[row]))
        sample.append(row);used_planes.add(int(family))
    return np.asarray(sample,np.int64)


def canonical_hypotheses(world,tokens,pixels):
    world=np.asarray(world,np.float64);tokens=np.asarray(tokens);pixels=np.asarray(pixels,np.float64)
    if (world.shape!=(len(tokens),3) or pixels.shape!=(len(tokens),2)
            or not np.isfinite(world).all() or not np.isfinite(pixels).all()):
        raise ValueError('invalid token hypothesis inventory')
    # Identical alternatives cannot change sampling probability or support.
    _,first=np.unique(np.c_[tokens,world,pixels],axis=0,return_index=True)
    return world[first],tokens[first],pixels[first]


def score_pose(pose,world,pixels,groups,K,k1,threshold=4.,group_starts=None,return_selected=True):
    camera=world@pose[:3,:3].T+pose[:3,3]
    z=camera[:,2];safe_z=np.where(np.abs(z)>1e-12,z,1.)
    xy=camera[:,:2]/safe_z[:,None]
    radial=1.+k1*np.sum(xy*xy,axis=1)
    prediction=xy*radial[:,None]*np.array([K[0,0],K[1,1]])+np.array([K[0,2],K[1,2]])
    error=np.sum((prediction-pixels)**2,axis=1)
    error[(z<=1e-8)|(~np.isfinite(error))]=np.inf
    if group_starts is not None and not return_selected:
        e=np.minimum.reduceat(error,group_starts)
        selected=np.zeros(0,np.int64)
    else:
        selected=np.array([g[np.argmin(error[g])] for g in groups],np.int64)
        e=error[selected]
    valid=e<=threshold**2
    key=(int(valid.sum()),-float(np.minimum(e,threshold**2).sum()))
    return key,selected[valid] if return_selected else selected


def solve(world,tokens,K,k1,rows,pixels=None,iterations=128,seed=260901,
          sampling_policy='uniform',planes=None,scores=None,hypothesis_budget=None,stats=None):
    started=time.perf_counter()
    if sampling_policy not in ('uniform','geometry','geometry_score','row_uniform','context_prior'):raise ValueError('invalid sampling policy')
    if hypothesis_budget is not None and hypothesis_budget<1:raise ValueError('positive hypothesis budget required')
    if iterations<1:raise ValueError('positive fixed RANSAC budget required')
    if pixels is None:pixels=np.c_[(tokens%64)*4+1.5,(tokens//64)*4+1.5]
    selected_world,selected_tokens,selected_pixels=world[rows],tokens[rows],pixels[rows]
    _,first,inverse=np.unique(np.c_[selected_tokens,selected_world,selected_pixels],axis=0,return_index=True,return_inverse=True)
    world,tokens,pixels=canonical_hypotheses(selected_world,selected_tokens,selected_pixels)
    if planes is None:planes=np.zeros(len(first),np.int64)
    else:
        plane_values=np.asarray(planes)[rows]
        planes=np.full(len(first),np.iinfo(np.int64).max,np.int64)
        np.minimum.at(planes,inverse,plane_values)
    if scores is None:scores=np.zeros(len(first))
    else:
        score_values=np.asarray(scores)[rows]
        if not np.isfinite(score_values).all():raise ValueError('nonfinite sampling scores')
        scores=np.full(len(first),-np.inf)
        np.maximum.at(scores,inverse,score_values)
    if sampling_policy == 'context_prior' and np.any(scores <= 0):
        raise ValueError('context sampling weights must be positive')
    unique=np.unique(tokens)
    if len(unique)<6:return None
    groups=[np.flatnonzero(tokens==t) for t in unique]
    centers=np.c_[(unique%64)*4+1.5,(unique//64)*4+1.5]
    starts=np.flatnonzero(np.r_[True,tokens[1:]!=tokens[:-1]])
    group_ids=np.repeat(np.arange(len(groups)),np.diff(np.r_[starts,len(tokens)]))
    plane_groups={int(p):np.flatnonzero(planes==p) for p in np.unique(planes)}
    K=np.asarray(K,np.float64);distortion=np.array([k1,0.,0.,0.,0.])
    rng=np.random.default_rng(seed);best=None;best_key=(-1,-np.inf)
    attempted=valid_models=0
    for _ in range(iterations):
        if hypothesis_budget is not None and valid_models>=hypothesis_budget:break
        attempted+=1
        sample=guided_sample(rng,groups,centers,planes,scores,sampling_policy,plane_groups,group_ids)
        singular=np.linalg.svd(world[sample]-world[sample].mean(axis=0),compute_uv=False)
        if singular[1]<=max(1e-10,singular[0]*1e-8):continue
        try:
            result=cv2.solvePnPGeneric(world[sample],pixels[sample],K,distortion,flags=cv2.SOLVEPNP_AP3P)
        except cv2.error:
            continue
        if not result[0]:continue
        for rv,tv in zip(result[1],result[2]):
            if hypothesis_budget is not None and valid_models>=hypothesis_budget:break
            if not np.isfinite(rv).all() or not np.isfinite(tv).all():continue
            valid_models+=1
            pose=np.eye(4);pose[:3,:3]=cv2.Rodrigues(rv)[0];pose[:3,3]=tv.reshape(3)
            key,_=score_pose(pose,world,pixels,groups,K,k1,group_starts=starts,return_selected=False)
            if key>best_key:best,best_key=pose,key
    if stats is not None:stats.update(attempted_samples=attempted,scored_hypotheses=valid_models,
                                    budget_reached=hypothesis_budget is None or valid_models==hypothesis_budget)
    if best is None or best_key[0]<6:
        if stats is not None:stats['seconds']=time.perf_counter()-started
        return None
    # LM is a proposal only: retain it iff the same unique-token score improves.
    for _ in range(3):
        _,selected=score_pose(best,world,pixels,groups,K,k1)
        if len(selected)<6:break
        try:
            rv,tv=cv2.solvePnPRefineLM(world[selected],pixels[selected],K,distortion,
                                      cv2.Rodrigues(best[:3,:3])[0],best[:3,3].copy())
        except cv2.error:
            break
        if not np.isfinite(rv).all() or not np.isfinite(tv).all():break
        pose=np.eye(4);pose[:3,:3]=cv2.Rodrigues(rv)[0];pose[:3,3]=tv.reshape(3)
        key,_=score_pose(pose,world,pixels,groups,K,k1)
        if key<=best_key:break
        best,best_key=pose,key
    if stats is not None:stats['seconds']=time.perf_counter()-started
    return best
