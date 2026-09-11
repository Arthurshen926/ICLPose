"""Pose-free appearance arrangement and relative-shape reads on fixed evidence."""
import numpy as np


def normalise(x):
    x=np.asarray(x,np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)


def sector_descriptors(grid, radius=2):
    """Four disjoint cardinal sectors, excludes centre; count-normalised at edges.

    Returns H,W,4,C. The pooling is in observed image coordinates: no invented
    geometry, image wraparound, interpolation-derived high resolution, or pose.
    """
    grid=np.asarray(grid,np.float32)
    if grid.ndim!=3 or radius<1 or not np.isfinite(grid).all():raise ValueError('invalid feature grid')
    h,w,c=grid.shape
    total=np.zeros((h,w,4,c),np.float32);count=np.zeros((h,w,4,1),np.float32)
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            if dx==0 and dy==0:continue
            # Horizontal owns exact diagonals; sectors partition all neighbours.
            sector=(0 if dx<0 else 1) if abs(dx)>=abs(dy) else (2 if dy<0 else 3)
            y0,y1=max(0,-dy),min(h,h-dy);x0,x1=max(0,-dx),min(w,w-dx)
            total[y0:y1,x0:x1,sector]+=grid[y0+dy:y1+dy,x0+dx:x1+dx]
            count[y0:y1,x0:x1,sector]+=1
    return normalise(total/np.maximum(count,1))


def arrangement_scores(query, mapping):
    """Separate unordered context from the extra information in arrangement."""
    query=np.asarray(query,np.float32);mapping=np.asarray(mapping,np.float32)
    if query.shape!=mapping.shape or query.shape[-2]!=4:raise ValueError('sector shapes differ')
    aligned=np.mean(np.sum(query*mapping,axis=-1),axis=-1)
    unordered=np.sum(normalise(query.mean(axis=-2))*normalise(mapping.mean(axis=-2)),axis=-1)
    # A pre-fixed reflection is a negative control, not a selectable alignment.
    reflected=np.mean(np.sum(query*mapping[..., [1,0,2,3], :],axis=-1),axis=-1)
    return np.stack([unordered,aligned,aligned-reflected],axis=-1)


def relative_shape_features(tokens, qpoints, qnormals, qvalid, world, normals, cosine, neighbours=8):
    """Scale/rotation/sign-invariant soft relations; one vote per neighbour token.

    Choose spatial neighbours without map/labels; marginalise their candidate
    explanations with descriptor-only weights. No hard rejection or pose input.
    """
    tokens=np.asarray(tokens,np.int64);world=np.asarray(world,float);normals=normalise(normals)
    unique=np.unique(tokens);xy=np.c_[unique%64,unique//64]
    by_token={int(t):np.flatnonzero(tokens==t) for t in unique}
    out=np.zeros((len(tokens),3),np.float32)
    for k,t in enumerate(unique):
        rows=by_token[int(t)]
        if not qvalid[t]:continue
        d2=np.sum((xy-xy[k])**2,axis=1)
        near=np.flatnonzero((d2>=4)&(d2<=144)&qvalid[unique])
        near=near[np.lexsort((unique[near],d2[near]))[:neighbours]]
        if not len(near):continue
        normal_votes=[];incidence_votes=[]
        for ni in near:
            u=int(unique[ni]);partners=by_token[u]
            _,first=np.unique(np.c_[world[partners],normals[partners],cosine[partners]],axis=0,return_index=True)
            partners=partners[first]
            dq=qpoints[u]-qpoints[t];qlength=np.linalg.norm(dq)
            if qlength<1e-6:continue
            dq=dq/qlength
            dm=world[partners][None]-world[rows][:,None];length=np.linalg.norm(dm,axis=-1)
            direction=dm/np.maximum(length[...,None],1e-8)
            qdot=abs(float(qnormals[t]@qnormals[u]))
            ndiff=np.abs(np.abs(normals[rows]@normals[partners].T)-qdot)
            idiff=.5*(np.abs(np.abs(np.einsum('rsc,rc->rs',direction,normals[rows]))-abs(dq@qnormals[t]))+
                       np.abs(np.abs(np.einsum('rsc,sc->rs',direction,normals[partners]))-abs(dq@qnormals[u])))
            weight=np.exp(10*(cosine[partners]-np.max(cosine[partners])));weight/=weight.sum()
            valid=length>1e-4
            normal_votes.append((np.exp(-4*ndiff)*valid)@weight)
            incidence_votes.append((np.exp(-4*idiff)*valid)@weight)
        if normal_votes:
            out[rows,0]=np.mean(normal_votes,axis=0);out[rows,1]=np.mean(incidence_votes,axis=0)
            out[rows,2]=len(normal_votes)/neighbours
    return out
