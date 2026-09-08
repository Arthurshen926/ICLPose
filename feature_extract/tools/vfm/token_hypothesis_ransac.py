"""Token-level multi-hypothesis PnP: unique sampling, support and local refinement."""
import cv2
import numpy as np


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


def solve(world,tokens,K,k1,rows,pixels=None,iterations=128,seed=260901):
    if iterations<1:raise ValueError('positive fixed RANSAC budget required')
    if pixels is None:pixels=np.c_[(tokens%64)*4+1.5,(tokens//64)*4+1.5]
    world,tokens,pixels=canonical_hypotheses(world[rows],tokens[rows],pixels[rows])
    unique=np.unique(tokens)
    if len(unique)<6:return None
    groups=[np.flatnonzero(tokens==t) for t in unique]
    starts=np.flatnonzero(np.r_[True,tokens[1:]!=tokens[:-1]])
    K=np.asarray(K,np.float64);distortion=np.array([k1,0.,0.,0.,0.])
    rng=np.random.default_rng(seed);best=None;best_key=(-1,-np.inf)
    for _ in range(iterations):
        chosen=rng.choice(len(groups),4,replace=False)
        sample=np.array([groups[g][rng.integers(len(groups[g]))] for g in chosen])
        singular=np.linalg.svd(world[sample]-world[sample].mean(axis=0),compute_uv=False)
        if singular[1]<=max(1e-10,singular[0]*1e-8):continue
        try:
            result=cv2.solvePnPGeneric(world[sample],pixels[sample],K,distortion,flags=cv2.SOLVEPNP_AP3P)
        except cv2.error:
            continue
        if not result[0]:continue
        for rv,tv in zip(result[1],result[2]):
            if not np.isfinite(rv).all() or not np.isfinite(tv).all():continue
            pose=np.eye(4);pose[:3,:3]=cv2.Rodrigues(rv)[0];pose[:3,3]=tv.reshape(3)
            key,_=score_pose(pose,world,pixels,groups,K,k1,group_starts=starts,return_selected=False)
            if key>best_key:best,best_key=pose,key
    if best is None or best_key[0]<6:return None
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
    return best
