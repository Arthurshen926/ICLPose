"""Bounded regional matching with soft, rotation-invariant pairwise messages.

Relations are auxiliary compatibility scores, not calibrated probabilities.
Different physical regions are solved separately; world anchors never move.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import softmax


def configuration_match(unary,world,normals,query_points,query_normals,pixels,policy='joint',absolute_null=None,exclusive=False):
    u=np.asarray(unary,float);w=np.asarray(world,float);n=np.asarray(normals,float);q=np.asarray(query_points,float);qn=np.asarray(query_normals,float)
    if policy not in ['independent','joint','shuffled']:raise ValueError('unknown matching policy')
    if u.ndim!=2 or w.shape!=(*u.shape,3) or n.shape!=w.shape or len(u)<2:raise ValueError('invalid regional alternatives')
    if not all(np.isfinite(v).all() for v in [u,w,n,q,qn,pixels]):raise ValueError('nonfinite evidence')
    count,k=u.shape
    # Null means unknown, with no pairwise penalty or geometric coordinate.
    threshold=None if absolute_null is None else np.asarray(absolute_null,float)
    if threshold is not None and (threshold.shape not in [(),(count,)] or not np.isfinite(threshold).all()):raise ValueError('invalid null threshold')
    null=np.full(count,-1.) if threshold is None else (threshold-u.max(1))/.1
    logits=np.c_[(u-u.max(1,keepdims=True))/.1,null]
    _,anchor=np.unique(w.reshape(-1,3),axis=0,return_inverse=True)
    p=softmax(logits,axis=1)
    neighbor=cKDTree(pixels).query(pixels,k=min(9,count))[1][:,1:]
    edges=np.unique(np.sort(np.c_[np.repeat(np.arange(count),neighbor.shape[1]),neighbor.ravel()],axis=1),axis=0);i,j=edges.T
    if policy=='shuffled':
        order=np.random.default_rng(360).permutation(count);q=q[order];qn=qn[order]
    dq=np.linalg.norm(q[i]-q[j],axis=-1);dw=np.linalg.norm(w[i,:,None]-w[j,None,:],axis=-1)
    nearest=np.argmax(u,axis=1);ref=dw[np.arange(len(i)),nearest[i],nearest[j]];valid=(dq>1e-6)&(ref>1e-6)
    logscale=float(np.median(np.log(ref[valid]/dq[valid]))) if valid.any() else 0.
    # Compare within-domain normal angles; never compare camera and world normals directly.
    angle_q=np.arccos(np.clip(np.abs(np.sum(qn[i]*qn[j],axis=1)),0,1))
    angle_w=np.arccos(np.clip(np.abs(np.einsum('eid,ejd->eij',n[i],n[j])),0,1))
    distance_error=np.log(np.maximum(dw,1e-6)/np.maximum(dq[:,None,None],1e-6))-logscale
    normal_valid=(np.linalg.norm(n[i],axis=-1)[:,:,None]>.9)&(np.linalg.norm(n[j],axis=-1)[:,None,:]>.9)&(np.linalg.norm(qn[i],axis=-1)[:,None,None]>.9)&(np.linalg.norm(qn[j],axis=-1)[:,None,None]>.9)
    normal_cost=np.where(normal_valid,.5*np.minimum(((angle_w-angle_q[:,None,None])/np.deg2rad(20))**2,4),0.)
    cost=.5*np.minimum((distance_error/.5)**2,4)+normal_cost
    cost[~valid]=0.;degree=np.bincount(np.r_[i,j],minlength=count);iterations=5
    # All controls construct the same graph. The independent control skips messages.
    for _ in range(iterations):
        message=np.zeros((count,k));np.add.at(message,i,-np.einsum('eab,eb->ea',cost,p[j,:k]));np.add.at(message,j,-np.einsum('eab,ea->eb',cost,p[i,:k]));message/=np.maximum(degree[:,None],1)
        update=logits.copy()
        if policy!='independent':update[:,:k]+=message
        p=.5*p+.5*softmax(update,axis=1)
        if exclusive:
            column_mass=np.bincount(anchor,weights=p[:,:k].ravel());p[:,:k]/=np.maximum(1.,column_mass[anchor].reshape(count,k));p[:,k]=1-p[:,:k].sum(1)
    chosen=np.argmax(p,axis=1);selected=chosen<k
    return chosen,selected,p,dict(edges=len(edges),logscale=logscale,iterations=iterations,null_fraction=float(np.mean(~selected)),policy=policy,absolute_null=None if threshold is None else threshold.tolist(),exclusive=exclusive)
