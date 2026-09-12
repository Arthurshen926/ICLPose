"""Small, bounded native fine measurement operations; no pose labels at inference."""
import numpy as np
from feature_extract.tools.vfm.build_goal_maplet_fine_readout import sample_grid
from feature_extract.vfm.localization_goal_maplet.structured_local_memory import normalise


def same_surface_sample(grid, pixels, labels, surface_ids):
    """Bilinear aggregation excluding feature centers outside the observed surface.

    This only changes descriptor readout; it cannot undo mixing inside the frozen
    backbone. Missing support returns a zero descriptor and an explicit mask.
    """
    grid=np.asarray(grid);p=np.asarray(pixels,float);labels=np.asarray(labels)
    if labels.shape!=(144,256) or p.shape[-1]!=2 or grid.ndim!=3:
        raise ValueError('surface sample dimensions differ')
    ids=np.broadcast_to(np.asarray(surface_ids),p.shape[:-1])
    h,w,d=grid.shape
    x=np.clip((p[...,0]+.5)*w/256-.5,0,w-1);y=np.clip((p[...,1]+.5)*h/144-.5,0,h-1)
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);wx=x-x0;wy=y-y0
    result=np.zeros(p.shape[:-1]+(d,),float);mass=np.zeros(p.shape[:-1],float)
    for dx,dy,weight in [(0,0,(1-wx)*(1-wy)),(1,0,wx*(1-wy)),(0,1,(1-wx)*wy),(1,1,wx*wy)]:
        xx=np.minimum(x0+dx,w-1);yy=np.minimum(y0+dy,h-1)
        px=np.clip(np.floor((xx+.5)*256/w).astype(int),0,255)
        py=np.clip(np.floor((yy+.5)*144/h).astype(int),0,143)
        weight=weight*((labels[py,px]==ids)&(ids>=0))
        mass+=weight;result+=weight[...,None]*grid[yy,xx]
    centerx=np.clip(np.floor(p[...,0]+.5).astype(int),0,255);centery=np.clip(np.floor(p[...,1]+.5).astype(int),0,143)
    valid=(mass>1e-8)&(labels[centery,centerx]==ids)&(ids>=0)
    return normalise(result),valid


def eligible_rows(corr,lo,hi,project,camera,available,plane,require_groups=True):
    """Replay v264's fixed mask, including row-order tie breaking."""
    tok=corr['query_tokens'][lo:hi];pr=corr['prototype_atlas_row'][lo:hi];xy=corr['query_measurements_xy'][lo:hi]
    residual=np.linalg.norm(project-xy,axis=1)
    valid=(camera[:,2]>0)&np.isfinite(residual)&(residual<=4)&available[pr]
    rows=np.flatnonzero(valid);order=np.lexsort((rows,residual[rows],tok[rows]));rows=rows[order]
    _,first=np.unique(tok[rows],return_index=True);rows=rows[first]
    if len(rows)>128:rows=rows[np.linspace(0,len(rows)-1,128,dtype=int)]
    provenance=corr['provenance_region_plane_atlas_row'][lo:hi]
    def make_groups(rows):
        keys=np.c_[plane[pr[rows]],provenance[rows,0],tok[rows]//64//8,tok[rows]%64//8]
        _,labels=np.unique(keys,axis=0,return_inverse=True)
        return [np.flatnonzero(labels==v) for v in np.unique(labels)]
    if not require_groups:return rows,make_groups(rows)
    groups=make_groups(rows);keep=[g for g in groups if len(g)>=3]
    rows=rows[np.sort(np.concatenate(keep))] if keep else np.array([],int)
    return rows,make_groups(rows)


def fit_update_scale(original, update, target, variance, weights=None):
    """Constrained least squares update and isotropic Gaussian variance scaling.

    Call only on mapping supervision. The scalar alpha stays in [0,1]; no
    extrapolation beyond a token is possible. Weights can balance source views.
    """
    original=np.asarray(original,float);delta=np.asarray(update,float)-original
    target=np.asarray(target,float);v=np.asarray(variance,float)
    w=np.ones(len(v)) if weights is None else np.asarray(weights,float)
    if original.shape!=target.shape or original.shape!=delta.shape or original.shape!=(len(v),2) or not len(v) or np.any(v<=0) or np.any(w<=0) or not all(np.isfinite(x).all() for x in [original,delta,target,v,w]):
        raise ValueError('invalid mapping calibration rows')
    denom=np.sum(w*np.sum(delta*delta,axis=1))
    alpha=float(np.clip(np.sum(w*np.sum(delta*(target-original),axis=1))/denom,0,1)) if denom>1e-12 else 0.
    residual=original+alpha*delta-target
    scale=float(np.sum(w*np.sum(residual**2,axis=1)/(2*v))/w.sum())
    return dict(alpha=alpha,variance_scale=max(scale,1e-6))


def compound_whiten(residual, group_ids, rho):
    """Whiten standardized residuals for C=(1-rho)I+rho*11^T per group.

    Each group is independent; negative IDs are independent individual rows.
    The common mode is downweighted without treating n copies as n independent
    votes. Singleton groups remain exactly unchanged.
    """
    residual=np.asarray(residual,float);ids=np.asarray(group_ids)
    if residual.ndim!=2 or residual.shape[1]!=2 or ids.shape!=(len(residual),) or ids.dtype.kind not in 'iu' or not np.isfinite(rho) or not 0<=rho<1:
        raise ValueError('invalid compound measurement covariance')
    out=residual.copy()
    if rho==0:return out
    for g in np.unique(ids[ids>=0]):
        rows=np.flatnonzero(ids==g);n=len(rows)
        if n<2:continue
        mean=residual[rows].mean(0)
        out[rows]=(residual[rows]-mean)/np.sqrt(1-rho)+mean/np.sqrt(1+(n-1)*rho)
    return out


def load_measurement_groups(path,correspondence_path,corr):
    import json
    from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256
    with np.load(path) as z:meta=json.loads(str(z['metadata_json']));arrays={k:z[k] for k in z.files if k!='metadata_json'}
    if (meta.get('artifact_type')!='goal_maplet_correlated_measurement_groups_v1' or meta.get('query_ground_truth_read') is not False
            or arrays_sha256(arrays)!=meta.get('arrays_sha256') or canonical_json_sha256({k:v for k,v in meta.items() if k!='content_sha256'})!=meta.get('content_sha256')
            or meta.get('correspondence_file_sha256')!=file_sha256(correspondence_path)
            or not np.array_equal(arrays['names'],corr['names']) or not np.array_equal(arrays['correspondence_offsets'],corr['correspondence_offsets'])
            or arrays['group_ids'].shape!=corr['query_tokens'].shape or arrays['group_ids'].dtype.kind not in 'iu'
            or not np.isfinite(meta.get('rho',np.nan)) or not 0<=meta['rho']<1):
        raise ValueError('correlated measurement group lineage differs')
    # Groups may not mix queries, physical planes, or alternative identities of a token.
    seen=set()
    for lo,hi in zip(corr['correspondence_offsets'][:-1],corr['correspondence_offsets'][1:]):
        ids=arrays['group_ids'][lo:hi]
        for g in np.unique(ids[ids>=0]):
            rows=np.flatnonzero(ids==g)+lo
            if int(g) in seen or len(np.unique(corr['query_tokens'][rows]))!=len(rows) or len(np.unique(corr['provenance_region_plane_atlas_row'][rows,1]))!=1:
                raise ValueError('correlated groups mix physical evidence')
            seen.add(int(g))
    return arrays['group_ids'],meta
