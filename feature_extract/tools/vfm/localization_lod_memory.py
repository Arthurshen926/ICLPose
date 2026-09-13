"""Feature LoD over immutable metric anchors, with a direct local gateway.

L0: 8m parent blocks with multiple appearance representatives.
L1: 2m children with actual appearance representatives (not mean geometry).
L2: original anchor identities, read per query token after opening children.
No camera pose or distance-to-camera is needed for initial expansion.
"""
import numpy as np
from feature_extract.tools.vfm.partial_visibility_memory import local_modes,diverse_tokens
from feature_extract.vfm.localization_goal_maplet.structured_local_memory import normalise


class LocalizationLoD:
    def __init__(self,world,features,eligible=None):
        self.world=np.asarray(world);self.features=normalise(features)
        self.eligible=np.ones(len(world),bool) if eligible is None else np.asarray(eligible,bool)
        self.rows=np.flatnonzero(self.eligible)
        if not len(self.rows):raise ValueError('empty map')
        leaf_key=np.floor(self.world[self.rows]/2).astype(np.int64)
        keys,groups=np.unique(leaf_key,axis=0,return_inverse=True)
        self.leaf_members=[self.rows[groups==i] for i in range(len(keys))]
        self.anchor_leaf=np.full(len(world),-1,int);self.anchor_leaf[self.rows]=groups
        parent_key=np.floor_divide(keys,4)
        _,self.leaf_parent=np.unique(parent_key,axis=0,return_inverse=True)
        self.fine_rows=local_modes(self.world,self.features,self.eligible,2.,4)
        self.coarse_rows=local_modes(self.world,self.features,self.eligible,8.,8)
        self.fine_leaf=self.anchor_leaf[self.fine_rows]
        self.fine_parent=self.leaf_parent[self.fine_leaf]
        self.coarse_parent=self.leaf_parent[self.anchor_leaf[self.coarse_rows]]
        self.fine_features=self.features[self.fine_rows];self.coarse_features=self.features[self.coarse_rows]
        # Representatives are grouped in spatial-key order by local_modes.
        assert np.all(np.diff(self.coarse_parent)>=0) and np.all(np.diff(self.fine_leaf)>=0)
        self.parent_starts=np.r_[0,np.flatnonzero(np.diff(self.coarse_parent))+1]
        self.leaf_starts=np.r_[0,np.flatnonzero(np.diff(self.fine_leaf))+1]

    def nearest(self,query,flat=False):
        query=normalise(query);n=len(query)
        if flat:
            sim=query@self.features[self.rows].T
            return self.rows[sim.argmax(1)],sim.max(1),sim.mean(1),sim.std(1),dict(descriptor_comparisons=n*len(self.rows),leaf_anchor_reads=n*len(self.rows),flat=True)
        coarse=query@self.coarse_features.T;fine=query@self.fine_features.T
        parent_scores=np.maximum.reduceat(coarse,self.parent_starts,axis=1)
        parent_ids=self.coarse_parent[self.parent_starts]
        parents=parent_ids[np.argsort(-parent_scores,axis=1,kind='stable')[:,:2]]
        leaf_scores=np.maximum.reduceat(fine,self.leaf_starts,axis=1)
        leaf_ids=self.fine_leaf[self.leaf_starts]
        allowed=(self.fine_parent[self.leaf_starts][None,:,None]==parents[:,None,:]).any(-1)
        chosen_leaf=np.argsort(-np.where(allowed,leaf_scores,-np.inf),axis=1,kind='stable')[:,:2]
        gateway=leaf_ids[leaf_scores.argmax(1)]
        rows=[];best=[];mean=[];std=[];reads=0;opened=[]
        for i in range(n):
            leaves=list(leaf_ids[chosen_leaf[i,allowed[i,chosen_leaf[i]]]])
            # Direct gateway cannot be vetoed by an incorrect coarse parent.
            leaves.append(int(gateway[i]))
            opened_ids=np.unique(leaves);ids=np.concatenate([self.leaf_members[j] for j in opened_ids]);s=query[i]@self.features[ids].T
            rows.append(int(ids[np.argmax(s)]));best.append(float(s.max()));mean.append(float(s.mean()));std.append(float(s.std()));reads+=len(ids);opened.append(opened_ids.tolist())
        return np.array(rows),np.array(best),np.array(mean),np.array(std),dict(descriptor_comparisons=n*(len(self.coarse_rows)+len(self.fine_rows))+reads,leaf_anchor_reads=reads,opened_leaf_ids=opened,flat=False)

    def proposals(self,query,count=4,flat=False):
        query=normalise(query);tokens=diverse_tokens(query,np.ones(len(query),bool))
        rows,best,mean,std,cost=self.nearest(query[tokens],flat)
        # Common weights in both arms: map-wide moments from an identical fixed
        # representative bank, so LoD does not change the score's population.
        ref=query[tokens]@self.fine_features.T;cost['descriptor_comparisons']+=len(tokens)*len(self.fine_rows)
        density=np.maximum(np.maximum(query[tokens]@query[tokens].T,0)**8 @ np.ones(len(tokens)),1)
        weights=np.maximum(best-ref.mean(1),0)/np.maximum(ref.std(1),1e-6)/density
        xyz=self.world[rows];distance=np.linalg.norm(xyz[:,None]-xyz[None,:],axis=-1);active=np.ones(len(rows),bool);used=np.zeros(len(rows),bool);chosen=[]
        for _ in range(count):
            if not active.any():break
            score=(distance<=6)@(weights*~used);j=int(np.argmax(np.where(active,score,-np.inf)))
            if score[j]<=0:break
            chosen.append(int(rows[j]));used|=distance[j]<=6;active&=distance[j]>=3
        return chosen,cost

    def arrays(self):
        return dict(eligible_rows=self.rows,coarse_rows=self.coarse_rows,fine_rows=self.fine_rows,coarse_parent=self.coarse_parent,fine_leaf=self.fine_leaf,leaf_parent=self.leaf_parent,leaf_offsets=np.r_[0,np.cumsum([len(x) for x in self.leaf_members])],leaf_members=np.concatenate(self.leaf_members))
