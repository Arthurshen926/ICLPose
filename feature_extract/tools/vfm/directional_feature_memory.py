"""Select anonymous feature modes by viewing direction, without query appearance."""
import numpy as np

def directional_target(modes,directions,count,world,pose,rule):
    modes=np.asarray(modes,float);directions=np.asarray(directions,float)
    center=-pose[:3,:3].T@pose[:3,3];v=center-world;v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
    valid=np.arange(modes.shape[1])[None,:]<np.asarray(count)[:,None]
    if np.any(np.sum(valid,axis=1)==0):raise ValueError('no visible map mode')
    score=np.einsum('nkd,nd->nk',directions,v);score=np.where(valid,score,-np.inf)
    if rule=='nearest':target=modes[np.arange(len(modes)),np.argmax(score,axis=1)]
    elif rule=='kernel':
        # Fixed 15-degree angular bandwidth; no query labels or appearance used.
        weights=np.exp((score-score.max(axis=1,keepdims=True))/(1-np.cos(np.deg2rad(15))))
        weights/=weights.sum(axis=1,keepdims=True);target=np.einsum('nk,nkd->nd',weights,modes)
    else:raise ValueError('unknown direction rule')
    return target/np.maximum(np.linalg.norm(target,axis=1,keepdims=True),1e-12)
