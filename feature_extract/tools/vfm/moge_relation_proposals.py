"""Scale-free within-quartet geometry for soft correspondence proposals."""
import numpy as np


def distance_dispersion(world_quartets,query_quartet):
    """Variance of six log distance ratios; translation/rotation/scale invariant.

    Reflection and repeated congruent structures are not disambiguated. This
    is a sampling proxy under approximate common MoGe scale, not a likelihood.
    """
    w=np.asarray(world_quartets,float);q=np.asarray(query_quartet,float)
    i,j=np.triu_indices(4,1);dw=np.linalg.norm(w[:,i]-w[:,j],axis=-1);dq=np.linalg.norm(q[i]-q[j],axis=-1)
    valid=(dw>1e-6).all(1)&np.isfinite(dw).all(1)&bool((dq>1e-6).all() and np.isfinite(dq).all())
    ratio=np.log(np.maximum(dw,1e-12)/np.maximum(dq,1e-12));out=np.mean((ratio-ratio.mean(1,keepdims=True))**2,axis=1);out[~valid]=np.inf
    return out


class RelationSampler:
    def __init__(self,points,valid,policy='relation',seed=314):
        self.points=np.asarray(points,float).copy();self.valid=np.asarray(valid,bool).copy();self.policy=policy
        if policy not in ['uniform_control','relation','shuffled']:raise ValueError('unknown relation proposal policy')
        if policy=='shuffled':
            ids=np.flatnonzero(self.valid);self.points[ids]=self.points[np.random.default_rng(seed).permutation(ids)]
        self.attempts=0;self.available=0;self.guided=0;self.quartet_proposals=0
    def __call__(self,rng,groups,world,tokens):
        chosen=rng.choice(len(groups),4,replace=False)
        # All controls draw and score the same number of candidate quartets.
        draws=np.array([[groups[g][rng.integers(len(groups[g]))] for g in chosen] for _ in range(8)])
        qt=tokens[draws[0]];explore=rng.random()<.5;self.attempts+=1;self.quartet_proposals+=8
        if not self.valid[qt].all():return draws[0]
        self.available+=1;error=distance_dispersion(world[draws],self.points[qt])
        if self.policy!='uniform_control' and not explore and np.isfinite(error).any():
            self.guided+=1;return draws[int(np.argmin(error))]
        return draws[0]
    def summary(self):return dict(relation_attempts=self.attempts,valid_relation_attempts=self.available,guided_attempts=self.guided,quartet_proposals=self.quartet_proposals,policy=self.policy)
