"""Bounded averaging of three frozen frontend families, without pose labels."""
import numpy as np
from scipy.spatial.transform import Rotation


def consensus(reference,poses,method='mean'):
    if method not in ['mean','median']:raise ValueError('unknown local consensus')
    p=np.asarray(poses,float)
    if len(p)<3 or not np.isfinite(p).all():return reference.copy(),False
    centers=-np.einsum('nji,nj->ni',p[:,:3,:3],p[:,:3,3]);center=-reference[:3,:3].T@reference[:3,3]
    angular=Rotation.from_matrix(p[:,:3,:3]@reference[:3,:3].T).magnitude()
    if np.any(np.linalg.norm(centers-center,axis=1)>.5) or np.any(angular>np.deg2rad(5)):return reference.copy(),False
    # Exact duplicate outputs add no additional mass.
    _,ids=np.unique(p.reshape(len(p),-1),axis=0,return_index=True);p=p[ids];centers=centers[ids]
    weights=np.ones(len(p))/len(p);c=centers.mean(0)
    if method=='median':
        # Smoothed Euclidean geometric median, fixed metric epsilon.
        for _ in range(100):
            weights=1/np.maximum(np.linalg.norm(centers-c,axis=1),1e-5);weights/=weights.sum();new=weights@centers
            if np.linalg.norm(new-c)<1e-9:c=new;break
            c=new
    R=Rotation.from_matrix(p[:,:3,:3]).mean(weights=weights).as_matrix();out=np.eye(4);out[:3,:3]=R;out[:3,3]=-R@c
    return out,True


def interpolate_pose(a,b,fraction):
    """Interpolate camera centers and shortest relative rotation; frame equivariant."""
    if not 0<=fraction<=1:raise ValueError('fraction outside [0,1]')
    if fraction==0:return a.copy()
    if fraction==1:return b.copy()
    ca=-a[:3,:3].T@a[:3,3];cb=-b[:3,:3].T@b[:3,3]
    rot=Rotation.from_matrix(b[:3,:3]@a[:3,:3].T).as_rotvec()
    R=Rotation.from_rotvec(fraction*rot).as_matrix()@a[:3,:3];c=(1-fraction)*ca+fraction*cb
    out=np.eye(4);out[:3,:3]=R;out[:3,3]=-R@c;return out
