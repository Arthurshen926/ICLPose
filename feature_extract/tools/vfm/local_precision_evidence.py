"""Paired coarse/fine appearance evidence on identical map anchors.

Supports are intersected before comparison. These are score differences, not
calibrated likelihood ratios; shared encoders preclude independence claims.
"""
import cv2
import numpy as np
from feature_extract.tools.vfm.build_goal_maplet_fine_readout import sample_grid

FEATURE_NAMES=[f'{level}_{stat}' for level in ['coarse','fine'] for stat in ['mean','median','trimmed','sign','cell_mean','standardized_mean']]


def project_support(pose,world,K,k1):
    xy=cv2.projectPoints(world,cv2.Rodrigues(pose[:3,:3])[0],pose[:3,3],K,np.array([k1,0.,0.,0.,0.]))[0].reshape(-1,2)
    z=(world@pose[:3,:3].T+pose[:3,3])[:,2]
    return xy,(z>0)&np.isfinite(xy).all(1)&(xy>=4).all(1)&(xy<=np.array([251,139])).all(1)


def paired_features(base,candidate,world,pixels,tokens,K,k1,grids,targets):
    if not len(world):return np.zeros(12),0
    a,va=project_support(base,world,K,k1);b,vb=project_support(candidate,world,K,k1)
    keep=va&vb&(np.linalg.norm(a-pixels,axis=1)<=4)&(np.linalg.norm(b-pixels,axis=1)<=4)
    rows=np.flatnonzero(keep)
    # Symmetric tie breaking; one physical observation per query token.
    e=np.linalg.norm(a-pixels,axis=1)+np.linalg.norm(b-pixels,axis=1)
    rows=rows[np.lexsort((rows,e[rows],tokens[rows]))]
    _,first=np.unique(tokens[rows],return_index=True);rows=rows[first]
    if len(rows)<6:return np.zeros(12),len(rows)
    cell=(tokens[rows]//64//4)*16+(tokens[rows]%64//4)
    features=[]
    for grid,target in zip(grids,targets):
        fa=sample_grid(grid,a[rows]);fb=sample_grid(grid,b[rows]);delta=np.einsum('nd,nd->n',fb-fa,target[rows])
        sorted_delta=np.sort(delta);trim=len(delta)//10;trimmed=sorted_delta[trim:len(delta)-trim]
        cell_mean=np.mean([delta[cell==c].mean() for c in np.unique(cell)])
        se=max(float(delta.std()/np.sqrt(len(delta))),1e-4)
        features.extend([delta.mean(),np.median(delta),trimmed.mean(),np.sign(delta).mean(),cell_mean,np.clip(delta.mean()/se,-20,20)])
    return np.array(features),len(rows)
