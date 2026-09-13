"""Pose-conditioned anonymous RADIO field and symmetric context verification.

Projection is a sparse atlas z-buffer approximation, not full surface rendering.
Map prototype selection uses geometry/view direction only, never query features.
"""
import numpy as np
from feature_extract.vfm.localization_goal_maplet.structured_local_memory import normalise,sector_descriptors


def project_field(world,features,directions,pose,K,k1,grid=(36,64),depth_band=.25,return_depth=False):
    h,w=grid;depth=np.full((h,w),np.nan);out=np.zeros((h*w,features.shape[1]),np.float32);valid=np.zeros(h*w,bool)
    if not np.isfinite(pose).all():return (out.reshape(h,w,-1),valid.reshape(h,w),depth) if return_depth else (out.reshape(h,w,-1),valid.reshape(h,w))
    cam=world@pose[:3,:3].T+pose[:3,3];positive=np.isfinite(cam).all(1)&(cam[:,2]>1e-6)
    xy=cam[:,:2]/np.where(positive,cam[:,2],1)[:,None]
    xy*=1+k1*np.sum(xy*xy,axis=1)[:,None]
    pix=xy*np.array([K[0,0],K[1,1]])+np.array([K[0,2],K[1,2]])
    center=-pose[:3,:3].T@pose[:3,3];delta=center-world;view=np.sum(normalise(delta)*directions,axis=1)
    keep=positive&np.isfinite(pix).all(1)&(pix[:,0]>=-.5)&(pix[:,0]<w*4-.5)&(pix[:,1]>=-.5)&(pix[:,1]<h*4-.5)&(view>=0)
    ids=np.flatnonzero(keep)
    if not len(ids):return (out.reshape(h,w,-1),valid.reshape(h,w),depth) if return_depth else (out.reshape(h,w,-1),valid.reshape(h,w))
    token=np.floor((pix[ids]+.5)/4).astype(int);token=token[:,1]*w+token[:,0]
    zmin=np.full(h*w,np.inf);np.minimum.at(zmin,token,cam[ids,2]);near=cam[ids,2]<=zmin[token]+depth_band;ids=ids[near];token=token[near]
    target=np.c_[(token%w)*4+1.5,(token//w)*4+1.5];dist=np.sum((pix[ids]-target)**2,axis=1)
    order=np.lexsort((ids,-view[ids],dist,token));ids=ids[order];token=token[order];_,first=np.unique(token,return_index=True)
    ids=ids[first];token=token[first];out[token]=features[ids];valid[token]=True
    depth.ravel()[token]=cam[ids,2]
    result=(normalise(out).reshape(h,w,-1),valid.reshape(h,w))
    return (*result,depth) if return_depth else result


def neighbour_valid(mask):
    """Require every in-image radius-one neighbour (and centre) to be mapped."""
    mask=np.asarray(mask,bool);h,w=mask.shape;out=mask.copy()
    for dy in [-1,0,1]:
        for dx in [-1,0,1]:
            y0,y1=max(0,-dy),min(h,h-dy);x0,x1=max(0,-dx),min(w,w-dx)
            out[y0:y1,x0:x1]&=mask[y0+dy:y1+dy,x0+dx:x1+dx]
    return out


def exclude_generation(mask,tokens,radius=1):
    out=np.asarray(mask,bool).copy();h,w=out.shape
    for t in np.unique(tokens):
        y,x=divmod(int(t),w)
        if not (0<=y<h):raise ValueError('generation token outside query grid')
        out[max(0,y-radius):min(h,y+radius+1),max(0,x-radius):min(w,x+radius+1)]=False
    return out


def budget_indices(mask,disagreement,budget,uniform=False):
    ids=np.flatnonzero(np.asarray(mask).ravel());d=np.asarray(disagreement).ravel()
    if budget<1 or not np.isfinite(d[ids]).all():raise ValueError('invalid verification budget or disagreement')
    if len(ids)<=budget:return ids
    if uniform:return ids[np.linspace(0,len(ids)-1,budget,dtype=int)]
    return ids[np.lexsort((ids,-d[ids]))[:budget]]


def score_pair(query,fields,masks,generation_tokens,budget=128,query_mask=None,instability=None,include_centered=False):
    """Freeze five explicit controls; no learned weights or query labels.

    Common mapped domains are fixed for both candidates. The holdout context
    cannot include any token present in either generator correspondence bank.
    This does not imply statistical independence from the shared VFM backbone.
    """
    q=normalise(query);f=[normalise(x) for x in fields]
    common=masks[0]&masks[1]
    if query_mask is not None:common=common&query_mask
    local_dis=1-np.sum(f[0]*f[1],axis=-1)
    qc=sector_descriptors(q,1);fc=[sector_descriptors(x,1) for x in f]
    cm=neighbour_valid(masks[0])&neighbour_valid(masks[1])
    if query_mask is not None:cm=cm&neighbour_valid(query_mask)
    dis=1-np.mean(np.sum(fc[0]*fc[1],axis=-1),axis=-1)
    if instability is not None:
        noise=instability[0]+instability[1]
        local_dis=local_dis/(noise+1e-3)
        # Neighbourhood mean uncertainty with no query labels or fitted weight.
        from scipy.ndimage import uniform_filter
        dis=dis/(uniform_filter(noise,size=3,mode="nearest")+1e-3)
    configs=[('local_disagreement',common,local_dis,False,False,False),('context_uniform',cm,dis,True,True,False),('context_disagreement',cm,dis,False,True,False),('context_reflected',cm,dis,False,True,True),('context_spatial_holdout',exclude_generation(cm,generation_tokens),dis,False,True,False)]
    result={}
    for name,mask,d,uniform,context,reflected in configs:
        ids=budget_indices(mask,d,budget,uniform);scores=[]
        for j in range(2):
            if context:
                pred=fc[j][..., [1,0,2,3],:] if reflected else fc[j]
                per=np.mean(np.sum(qc*pred,axis=-1),axis=-1)
            else:per=np.sum(q*f[j],axis=-1)
            scores.append(float(per.ravel()[ids].mean()) if len(ids) else None)
        selected=int(len(ids)>0 and scores[1]>scores[0])
        result[name]=dict(scores=scores,selected=selected,selected_tokens=ids.tolist(),available_tokens=int(mask.sum()),insufficient_evidence=not len(ids))
    if include_centered:
        per=np.stack([centered_similarity(qc,x) for x in fc])
        for name,uniform,guard in [('structure_uniform',True,False),('structure_disagreement',False,False),('structure_guarded',True,True)]:
            ids=budget_indices(cm,dis,budget,uniform);scores=[float(v.ravel()[ids].mean()) if len(ids) else None for v in per]
            delta=(per[1]-per[0]).ravel()[ids]
            supported=spatial_agreement(ids,delta,q.shape[1])
            selected=int(len(ids)>0 and scores[1]>scores[0] and (not guard or supported['allow']))
            result[name]=dict(scores=scores,selected=selected,selected_tokens=ids.tolist(),available_tokens=int(cm.sum()),insufficient_evidence=not len(ids) or (guard and supported['cells']<4),spatial_agreement=supported)
    return result


def centered_similarity(query_sectors,map_sectors):
    q=np.asarray(query_sectors);m=np.asarray(map_sectors)
    # Equals aligned similarity minus the mean over all sector permutations.
    return np.mean(np.sum((q-q.mean(axis=-2,keepdims=True))*(m-m.mean(axis=-2,keepdims=True)),axis=-1),axis=-1)


def spatial_agreement(tokens,differences,width=64):
    tokens=np.asarray(tokens,int);d=np.asarray(differences,float)
    cells=(tokens//width//8)*((width+7)//8)+(tokens%width//8)
    means=[float(d[cells==c].mean()) for c in np.unique(cells) if np.sum(cells==c)>=4]
    fraction=float(np.mean(np.array(means)>0)) if means else 0.
    return dict(cells=len(means),positive_fraction=fraction,allow=len(means)>=4 and fraction>=.75)


def phase_averaged_field(world,features,directions,pose,K,k1):
    """Sensitivity to +/-1 coarse-pixel projection phase, not a calibrated PDF."""
    fields=[];masks=[]
    for dx,dy in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
        camera=K.copy();camera[0,2]+=dx;camera[1,2]+=dy
        f,m=project_field(world,features,directions,pose,camera,k1);fields.append(f);masks.append(m)
    mean=np.mean(fields,axis=0);valid=np.logical_and.reduce(masks)
    instability=np.maximum(0,1-np.sum(mean*mean,axis=-1))
    return normalise(mean),valid,instability


def foreground_visibility(depths,masks,query_depth,query_valid,ratio=1.5,min_support=32):
    """Symmetric approximate foreground exclusion after robust common-scale fitting.

    Each pose fits its own scale on the same valid domain. This is not a
    calibrated visibility probability and can fail when foreground dominates.
    Missing/insufficient depth provides no permission to replace the reference.
    """
    if ratio<=1 or min_support<1:raise ValueError('invalid visibility parameters')
    q=np.asarray(query_depth,float)
    common=np.asarray(masks[0],bool)&np.asarray(masks[1],bool)&np.asarray(query_valid,bool)&np.isfinite(q)&(q>0)
    for z in depths:common &= np.isfinite(z)&(z>0)
    scales=[];visible=common.copy()
    if common.sum()<min_support:return np.zeros_like(common),dict(sufficient=False,scales=[],fit_tokens=int(common.sum()))
    for z in depths:
        scale=float(np.exp(np.median(np.log(np.asarray(z)[common]/q[common]))));scales.append(scale)
        # If observed depth is substantially closer than predicted mapped depth,
        # the mapped feature may be behind a query foreground occluder.
        visible &= q*scale*ratio>=z
    return visible,dict(sufficient=True,scales=scales,fit_tokens=int(common.sum()),visible_tokens=int(visible.sum()),ratio=ratio)
