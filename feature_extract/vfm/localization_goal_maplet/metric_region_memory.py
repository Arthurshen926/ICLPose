"""Fixed physical region membership and partial, unique-token activation."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit


def build_regions(world, radius=3.):
    world=np.asarray(world,float)
    if radius<=0 or world.ndim!=2 or world.shape[1]!=3 or not np.isfinite(world).all():raise ValueError('invalid metric map')
    origin=world.min(0);cell=np.floor((world-origin)/radius).astype(int)
    _,inverse=np.unique(cell,axis=0,return_inverse=True)
    count=np.bincount(inverse);centers=np.column_stack([np.bincount(inverse,weights=world[:,i])/count for i in range(3)])
    tree=cKDTree(world);members=[np.asarray(x,int) for x in tree.query_ball_point(centers,radius)]
    valid=[i for i,rows in enumerate(members) if len(np.unique(world[rows],axis=0))>=12]
    return centers[valid],[members[i] for i in valid]


def region_features(world,tokens,scores,rows,total_tokens):
    rows=np.asarray(rows,int);tok=tokens[rows]
    unique=np.unique(tok)
    best=np.array([rows[np.flatnonzero(tok==t)[np.argmax(scores[rows[tok==t]])]] for t in unique],int)
    prob=expit(scores[best]);xy=np.c_[tokens[best]%64/64.,tokens[best]//64/36.]
    cov=np.cov(xy.T) if len(best)>1 else np.zeros((2,2))
    singular=np.linalg.svd(world[best]-world[best].mean(0),compute_uv=False)
    singular=np.pad(singular,(0,max(0,3-len(singular))))
    return np.array([np.log1p(len(best)),len(best)/max(total_tokens,1),prob.mean(),np.percentile(prob,75),prob.std(),
        np.sqrt(max(np.linalg.det(cov),0)),singular[1]/max(singular[0],1e-8),singular[2]/max(singular[0],1e-8),np.log1p(singular[0]/np.sqrt(len(best)))])


def activate_regions(world,tokens,scores,centers,radius=3.,number=4,diverse=True,model=None,
                     prototype_ids=None,allowed_members=None):
    """Actual group membership for PnP; each region can be partially observed."""
    world=np.asarray(world);tokens=np.asarray(tokens);scores=np.asarray(scores)
    tree=cKDTree(world);members=tree.query_ball_point(centers,radius)
    if (prototype_ids is None)!=(allowed_members is None):raise ValueError('explicit membership requires prototype identities and map members')
    if allowed_members is not None:
        if len(prototype_ids)!=len(world) or len(allowed_members)!=len(centers):raise ValueError('explicit member inventory differs')
        members=[np.asarray(rows,int)[np.isin(np.asarray(prototype_ids)[rows],allowed_members[i])].tolist()
                 for i,rows in enumerate(members)]
    _,inv=np.unique(tokens,return_inverse=True);nt=int(inv.max()+1) if len(inv) else 0
    evidence=[];eligible=[];features=[]
    for i,group in enumerate(members):
        rows=np.asarray(group,int)
        if len(np.unique(tokens[rows]))<6:continue
        weight=np.zeros(nt);np.maximum.at(weight,inv[rows],expit(scores[rows]))
        evidence.append(weight);eligible.append(i)
        if model is not None:features.append(region_features(world,tokens,scores,rows,nt))
    if not eligible:return [],[]
    evidence=np.asarray(evidence);covered=np.zeros(nt);chosen=[];groups=[]
    prediction=None
    if model is not None:
        x=(np.asarray(features)-np.asarray(model['mean']))/np.asarray(model['std'])
        beta=np.asarray(model['coefficients']);prediction=expit(beta[0]+x@beta[1:])
    for _ in range(min(number,len(eligible))):
        gain=np.maximum(evidence-covered[None],0).sum(1) if diverse else evidence.sum(1)
        if prediction is not None:
            novelty=np.maximum(evidence-covered[None],0).sum(1)/np.maximum(evidence.sum(1),1e-8)
            gain=prediction*(.25+.75*novelty)
        gain[chosen]=-np.inf;pick=int(np.argmax(gain));chosen.append(pick)
        groups.append(np.asarray(members[eligible[pick]],int));covered=np.maximum(covered,evidence[pick])
    return groups,[int(eligible[i]) for i in chosen]
